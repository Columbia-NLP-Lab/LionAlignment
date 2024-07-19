"""
Supervised fine-tuning script for decoder language models.
"""

from typing import Any, Dict, Optional
import gc
import os
import logging
import random
import sys
import tqdm

import datasets
from accelerate.state import PartialState
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
from datasets import concatenate_datasets

from lionalign.arguments import H4ArgumentParser, ModelArguments, DataArguments
from lionalign.trainer.sft_trainer import SFTTrainer
from lionalign.trainer.sft_config import SFTConfig
from lionalign.data.sft_data_processor import SFTDatasetProcessor
from lionalign.data.utils import (
    get_datasets,
    apply_chat_template,
    get_dataset_cache_hash,
)
from lionalign.model_utils import get_peft_config, get_checkpoint, get_tokenizer

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
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
    def process_chat_template(datasets):
        remove_column_names = list(datasets.features)
        if "dataset_mix_source" in remove_column_names:
            remove_column_names.remove("dataset_mix_source")

        datasets = datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "sft",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=remove_column_names,
            desc="Applying chat template",
        )
        return datasets

    data_processor = SFTDatasetProcessor(
        tokenizer=tokenizer,
        assistant_bos=training_args.assistant_bos,
        assistant_eos=training_args.assistant_eos,
        max_seq_length=training_args.max_seq_length,
        num_proc=data_args.preprocessing_num_workers,
        mask_user_labels=training_args.mask_user_labels,
        compute_cu_seqlens=training_args.use_fast_model,
    )

    with PartialState().local_main_process_first():
        os.makedirs(training_args.dataset_cache_dir, exist_ok=True)

        # compute the hash of the train dataset
        train_dataset_cache_hash = get_dataset_cache_hash(
            data_args.train_dataset_mixer,
            data_args.train_dataset_split,
            data_args.chat_template,
            data_args.auto_insert_empty_system_msg,
            training_args.shuffle_train_dataloader,
            training_args.seed,
        )
        train_dataset_cache_path = os.path.join(
            training_args.dataset_cache_dir, train_dataset_cache_hash
        )

        # Load train dataset from cache if it exists
        if os.path.exists(train_dataset_cache_path):
            logger.info(f"Loading dataset from cache {train_dataset_cache_path}")
            train_dataset = datasets.load_from_disk(train_dataset_cache_path)

            if training_args.num_train_epochs > 1:
                logger.info(
                    f"Loading dataset from cache {train_dataset_cache_path} and setting num_train_epochs to 1"
                )
                training_args.num_train_epochs = 1
        else:
            train_dataset = get_datasets(
                data_args.train_dataset_mixer,
                data_args.train_dataset_split,
                columns_to_keep=["messages", "dataset_mix_source"],
                dedup_key="messages",
                shuffle=training_args.shuffle_train_dataloader,
                process_fn=process_chat_template,
                seed=training_args.seed,
                num_proc=data_args.preprocessing_num_workers,
            )

            ##########################
            # Print dataset statistics
            ##########################
            if training_args.local_rank in [-1, 0]:
                dataset_names = set(train_dataset["dataset_mix_source"])
                dataset_numbers = {dataset_name: 0 for dataset_name in dataset_names}

                for item in tqdm.tqdm(train_dataset, desc="Counting dataset examples"):
                    dataset_numbers[item["dataset_mix_source"]] += 1

                # print percentage of different sources and number of examples
                total = len(train_dataset)
                print(f"Total number of examples: {total}")
                for dataset_name in dataset_numbers:
                    dataset_number = dataset_numbers[dataset_name]
                    dataset_percentage = dataset_number / total * 100
                    print(
                        f'"{dataset_name}": {dataset_number} examples, {dataset_percentage:.2f}%'
                    )

            # Repeat the training dataset if num_train_epochs > 1
            if training_args.num_train_epochs > 1:
                train_dataset = concatenate_datasets(
                    [train_dataset for _ in range(training_args.num_train_epochs)]
                )
                if training_args.shuffle_train_dataloader:
                    train_dataset = train_dataset.shuffle(seed=training_args.seed)
                training_args.num_train_epochs = 1

            # Process train dataset
            train_dataset = data_processor(train_dataset)
            logger.info(
                f"Train dataset processed. Number of samples: {len(train_dataset)}"
            )

            if training_args.local_rank in [-1, 0]:
                logger.info(f"Saving dataset to cache {train_dataset_cache_path}")
                train_dataset.save_to_disk(train_dataset_cache_path)

        # Load eval dataset
        eval_dataset = None
        if data_args.eval_dataset_mixer is not None:
            eval_dataset = get_datasets(
                data_args.eval_dataset_mixer,
                data_args.eval_dataset_split,
                columns_to_keep=["messages", "dataset_mix_source"],
                dedup_key="messages",
                shuffle=False,
                process_fn=process_chat_template,
                seed=training_args.seed,
                num_proc=data_args.preprocessing_num_workers,
            )

        logger.info(f"Process datasets")
        eval_dataset = (
            data_processor(eval_dataset) if eval_dataset is not None else None
        )

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None,
        attn_implementation="flash_attention_2",
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
                f"Model type {model_config.model_type} not supported for fast model."
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )

    logger.info("*** Model loaded! ***")

    if training_args.mask_embed_grad:
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
            logger.warning(
                f"Model type {model.config.model_type} and model name"
                f" {model_args.model_name_or_path} not supported for masking special tokens."
            )

    ########################
    # Initialize the Trainer
    ########################

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        data_collator=data_processor.data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Start training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
