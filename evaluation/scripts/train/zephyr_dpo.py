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
"""
modified DPO training script from HuggingFace's zephyr repo
"""

import logging
import random
import torch
import os
import yaml
from transformers import set_seed
from src.trainers.configs import (
    DataArguments, DPOConfig, LoggingArguments, ModelArguments,
    H4ArgumentParser
)
from src.dataloads.data import apply_chat_template, get_datasets
from src.dataloads.decontaminate import decontaminate_humaneval
from src.utils.model_utils import (
    get_checkpoint,
    get_tokenizer,
    remove_optimizer_weights
)
from src.utils.utils import init_logger, is_main
from src.trainers.dpo_fixed import tokenize_row, DPOEnhancedTrainer
from src.constants import DPO_DATA_COLUMNS_TO_REMOVE, DPO_DATA_MIX_COLUMNS
from dataclasses import asdict


logger: logging.Logger


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, LoggingArguments, DPOConfig))
    model_args, data_args, logging_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    #######
    # Setup
    #######
    log_level = training_args.get_process_log_level()
    logger = init_logger(is_main=is_main(), log_level=log_level, is_distributed=True)

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    column_names = DPO_DATA_COLUMNS_TO_REMOVE
    column_names.remove("prompt_id")
    column_names.remove("other_info")
    column_to_keep = DPO_DATA_MIX_COLUMNS
    
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits, col_to_mix=column_to_keep)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    if 'stablelm' in tokenizer.name_or_path:
        logger.warning("Setting pad token id to 100288 assuming you are using StableLM tokenizer")
        tokenizer.pad_token_id = 100288

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter(
        decontaminate_humaneval,
        fn_kwargs={"text_column": "text_chosen"},
        batched=True,
        batch_size=10_000,
        keep_in_memory=True,
        num_proc=1,
        desc="Decontaminating HumanEval samples",
    )
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    ######################################################
    # pre-tokenize data used originally inside DPO trainer
    ######################################################
    raw_datasets = raw_datasets.map(
        tokenize_row,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": training_args.max_length,
            "truncation_mode": "keep_end",
            "max_prompt_length": training_args.max_prompt_length,
            "label_pad_token_id": -100,
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="(Map) Truncating prompt and responses"
    )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None
    )

    model = model_args.model_name_or_path
    # ref_model = model
    ref_model = model_args.ref_model_name_or_path or model
    ref_model_kwargs = model_kwargs

    #######################################
    # Initialize external logger, if needed
    #######################################
    all_run_args = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(training_args),
        **asdict(logging_args),
    }
    if 'wandb' in training_args.report_to and is_main():
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
        all_run_args['wandb_id'] = run_id

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = DPOEnhancedTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    
    if trainer.accelerator.is_main_process:
        ### save into yaml
        yaml_path = os.path.join(training_args.output_dir, "run_args.yaml")
        with open(yaml_path, "w", encoding='utf-8') as fwrite:
            yaml.dump(all_run_args, fwrite, default_flow_style=False)
    
    #############################################################
    # Save last model IF --save_strategy=no --save_total_limit=-1
    #############################################################
    if training_args.save_strategy == "no" and training_args.save_total_limit == -1:
        logger.info("*** Save LAST model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    # also remove the optimizer weights as its taking too much disk space
    if trainer.accelerator.is_main_process:
        remove_optimizer_weights(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("*** Training complete! ***")
    return


if __name__ == "__main__":
    main()