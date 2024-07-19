#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import gc
import os
import sys
import torch.distributed
import tqdm
import json
import yaml
import glob
from dataclasses import asdict, dataclass, field

from datasets import concatenate_datasets, load_dataset, load_from_disk

from accelerate import Accelerator
from accelerate.state import PartialState

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from lionalign.arguments import (
    H4ArgumentParser,
    ModelArguments,
    DataArguments,
    LoggingArguments,
)
from lionalign.data.dpo_data_processor import DPODatasetProcessor
from lionalign.trainer.precompute_dpo_trainer import PrecomputeDPOTrainer
from lionalign.trainer.dpo_config import DPOConfig
from lionalign.data.utils import (
    get_datasets,
    apply_chat_template,
    get_dataset_cache_hash,
)
from lionalign.model_utils import (
    get_checkpoint,
    get_tokenizer,
    get_peft_config,
    remove_optimizer_weights,
    fix_deepspeed_model_save,
)
from lionalign.logging_utils import init_logger, is_main

logger = logging.getLogger(__name__)


def load_precomputed_dpo_dataset(model_args, data_args, training_args, tokenizer, eval=False):
    mixer = (
        data_args.eval_dataset_mixer if eval else data_args.train_dataset_mixer
    )
    split = (
        data_args.eval_dataset_split if eval else data_args.train_dataset_split
    )
    shuffle = False if eval else training_args.shuffle_train_dataloader

    # compute the hash of the train dataset
    dataset_cache_hash = get_dataset_cache_hash(
        mixer,
        split,
        tokenizer.chat_template,
        data_args.auto_insert_empty_system_msg,
        shuffle,
        training_args.seed,
        model_name_or_path=model_args.model_name_or_path,
    )
    dataset_cache_path = os.path.join(
        training_args.dataset_cache_dir, dataset_cache_hash
    )

    # Load train dataset from cache if it exists
    filenames = glob.glob(dataset_cache_path + "_dpo_precompute_*")
    if len(filenames) > 0:
        output_dataset = []
        for filename in filenames:
            output_dataset.append(load_from_disk(filename))
        output_dataset = concatenate_datasets(output_dataset)
    else:
        raise ValueError(f"DPO Dataset cache not found at {dataset_cache_path}")

    return output_dataset, dataset_cache_path


def main():
    parser = H4ArgumentParser(
        (ModelArguments, DataArguments, LoggingArguments, DPOConfig)
    )
    model_args, data_args, logging_args, training_args = parser.parse()

    all_run_args = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(training_args),
        **asdict(logging_args),
    }
    if "wandb" in training_args.report_to and is_main():
        import wandb

        wandb_all_args = {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "training_args": asdict(training_args),
        }
        run = wandb.init(
            project=logging_args.wandb_project,
            name=training_args.output_dir.split("/")[-1] or None,
            group=logging_args.wandb_group,
            config=wandb_all_args,
        )
        run_id = run.id
        all_run_args["wandb_id"] = run_id

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Logging Setup
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    #################
    # Find checkpoint
    #################
    last_checkpoint = get_checkpoint(training_args)

    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    ###############
    # Load datasets
    ###############
    tokenizer = get_tokenizer(model_args, data_args)

    data_processor = DPODatasetProcessor(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        num_proc=data_args.preprocessing_num_workers,
        pad_to_multiple=1,
        use_fast_model=training_args.use_fast_model,
    )

    with PartialState().local_main_process_first():
        os.makedirs(training_args.dataset_cache_dir, exist_ok=True)

        train_dataset, train_dataset_cache_path = load_precomputed_dpo_dataset(
            model_args, data_args, training_args, tokenizer, eval=False
        )
        if data_args.eval_dataset_mixer:
            eval_dataset, eval_dataset_cache_path = load_precomputed_dpo_dataset(
                model_args, data_args, training_args, tokenizer, eval=True
            )
        else:
            eval_dataset, eval_dataset_cache_path = None, None

    ###############################
    # Instantiate model and trainer
    ###############################
    # TODO: Support adapter models and Peft training
    model = model_args.model_name_or_path
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2",  # TODO: support other attention implementations
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None,
    )

    if training_args.use_fast_model:
        model_config = transformers.PretrainedConfig.from_pretrained(
            model_args.model_name_or_path
        )

        if model_config.model_type == "llama":
            from lionalign.fastmodels.llama.modeling_llama import LlamaForCausalLM

            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )
        elif model_config.model_type == "gemma":
            from lionalign.fastmodels.gemma.modeling_gemma import GemmaForCausalLM

            model = GemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )
        else:
            raise ValueError(
                f"Model type {model.config.model_type} not supported for fast model."
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )

    logger.info("*** Model loaded! ***")

    if training_args.mask_embed_grad:
        # check if it is llama 3 model
        if model.config.model_type == "llama" and (
            130000 > model.config.vocab_size > 128000
        ):
            from lionalign.fastmodels.llama.embed_grad_mask import (
                apply_llama3_embed_grad_mask,
            )

            model = apply_llama3_embed_grad_mask(model)
        elif model.config.model_type == "gemma":
            from lionalign.fastmodels.gemma.embed_grad_mask import (
                apply_gemma_embed_grad_mask,
            )

            model = apply_gemma_embed_grad_mask(model)
        else:
            raise ValueError(
                f"Model type {model.config.model_type} and model name"
                f" {model_args.model_name_or_path} not supported for masking special tokens."
            )

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = PrecomputeDPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_processor.data_collator,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    logger.info("*** Training complete ***")

    # Save the run args
    if trainer.accelerator.is_main_process:
        yaml_path = os.path.join(training_args.output_dir, "run_args.yaml")
        with open(yaml_path, "w", encoding="utf-8") as fwrite:
            yaml.dump(all_run_args, fwrite, default_flow_style=False)

    # fix the deepspeed model save
    if (
        trainer.accelerator.is_main_process
        and trainer.accelerator.state.deepspeed_plugin is not None
        and trainer.accelerator.state.deepspeed_plugin.zero_stage != 3
    ):
        fix_deepspeed_model_save(training_args.output_dir)


if __name__ == "__main__":
    main()
