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
import torch
import os
import yaml
import pandas as pd
import gc
from transformers import set_seed, AutoModelForCausalLM
from datasets import DatasetDict, Dataset
from src.trainers.configs import (
    DataArguments, EfficientDPOConfig, LoggingArguments, ModelArguments,
    H4ArgumentParser
)
from src.dataloads.data import apply_chat_template, get_datasets
from src.dataloads.decontaminate import decontaminate_humaneval
from src.utils.data_utils import add_full_id
from src.utils.model_utils import (
    get_checkpoint,
    get_tokenizer,
    remove_optimizer_weights,
    fix_deepspeed_model_save
)
from src.utils.utils import init_logger, is_main
from src.trainers.dpo_fixed_v2 import tokenize_row, EfficientDPOTrainer
from src.constants import DPO_DATA_COLUMNS_TO_REMOVE, DPO_DATA_MIX_COLUMNS
from dataclasses import asdict, dataclass, field


logger: logging.Logger


@dataclass
class DataSizeControlledDataArguments(DataArguments):
    """
    modified from DataArguments to include max_data_size
    """
    
    max_data_size: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples to use for training."},
    )


def control_data_size(dataset: DatasetDict, max_data_size: int = 2000):
    if max_data_size == -1:
        return dataset
    if max_data_size > len(dataset):
        raise ValueError(f"max_data_size {max_data_size} is greater than the dataset size {len(dataset)}")
    
    dataset = dataset.shuffle(seed=42)
    subset_dataset = dataset.select(range(max_data_size))
    return subset_dataset


def prepare_datasets(
    data_args: DataSizeControlledDataArguments,
    training_args: EfficientDPOConfig,
    raw_datasets: DatasetDict,
    tokenizer,
):
    global logger
    column_names = DPO_DATA_COLUMNS_TO_REMOVE
    column_names.remove("prompt_id")
    column_names.remove("other_info")

    raw_datasets['train'] = control_data_size(
        dataset=raw_datasets['train'],
        max_data_size=data_args.max_data_size
    )

    raw_datasets = raw_datasets.map(
        add_full_id,
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        desc="Adding full_id to dataset",
    )

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
    return raw_datasets


def maybe_precompute_all_ref_probs(
    training_args: EfficientDPOConfig,
    ref_model: str,
    ref_model_kwargs: dict,
    datasets,
    tokenizer,
):
    if training_args.ref_update_steps != -1:
        return datasets['train'], datasets['test']
    
    # precompute all when ref_update_steps == -1
    # doing it here allows using a different ref model than the main model
    logger.info(f"Precomputing reference log probabilities using {ref_model}")
    model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_kwargs)

    assert training_args.loss_type == 'sigmoid'  # i.e., dpo
    trainer = EfficientDPOTrainer(
        model,
        args=training_args,
        train_dataset=datasets["train"],
        tokenizer=tokenizer,
    )
    
    train_logps = trainer.compute_all_reference_log_probs(datasets['train'])
    train_logps_df = pd.DataFrame(train_logps).drop_duplicates(subset='full_id')

    # update the datasets
    train_df: pd.DataFrame = datasets['train'].to_pandas()
    train_df.index = train_df['full_id'].values
    train_logps_df.index = train_logps_df['full_id'].values
    train_logps_df = train_logps_df.drop(columns=['full_id'])
    train_df = train_df.join(
        train_logps_df,
        on='full_id',
        how='inner'
    )

    test_logps = trainer.compute_all_reference_log_probs(datasets['test'])
    test_logps_df = pd.DataFrame(test_logps).drop_duplicates(subset='full_id')

    # update the datasets
    test_df: pd.DataFrame = datasets['test'].to_pandas()
    test_df.index = test_df['full_id'].values
    test_logps_df.index = test_logps_df['full_id'].values
    test_logps_df = test_logps_df.drop(columns=['full_id'])
    test_df = test_df.join(
        test_logps_df,
        on='full_id',
        how='inner'
    )
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def train_eval_save(
    model_args: ModelArguments,
    data_args: DataSizeControlledDataArguments,
    training_args: EfficientDPOConfig,
    logging_args: LoggingArguments,
    model_kwargs: dict,
    raw_datasets: DatasetDict,
    tokenizer,
    last_checkpoint
):
    global logger
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
    

    model = model_args.model_name_or_path
    ref_model = model_args.ref_model_name_or_path or model

    has_precomputed_ref_probs = 'reference_chosen_logps' in raw_datasets["train"].column_names
    if ref_model != model and not has_precomputed_ref_probs:
        raise ValueError("Reference model must be the same as the main model, unless you are running DPO with fully precomputed ref log probs.")

    model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

    #########################
    # Instantiate DPO trainer
    #########################    
    training_args.remove_unused_columns = False
    assert not training_args.precompute_ref_log_probs, "DPOEnhancedV2Trainer requires precompute_ref_log_probs to be False to work correctly"
    trainer = EfficientDPOTrainer(
        model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
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

        if trainer.accelerator.is_main_process and trainer.accelerator.state.deepspeed_plugin is not None:
            fix_deepspeed_model_save(training_args.output_dir)

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


def main():
    global logger
    parser = H4ArgumentParser((ModelArguments, DataSizeControlledDataArguments, LoggingArguments, EfficientDPOConfig))
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

    raw_datasets = prepare_datasets(
        data_args=data_args,
        training_args=training_args,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
    )


    ############################
    # prepare model loading args
    ############################
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

    ####################################################################################
    # pre-compute all reference. Used when you want to do DPO with a different ref model
    ####################################################################################
    train_dset, test_dset = maybe_precompute_all_ref_probs(
        training_args=training_args,
        ref_model=model_args.ref_model_name_or_path,
        ref_model_kwargs=model_kwargs,
        datasets=raw_datasets,
        tokenizer=tokenizer,
    )
    torch.cuda.empty_cache()
    gc.collect()

    ####################
    # train and evaluate
    ####################
    precomputed_datasets = DatasetDict({
        "train": train_dset,
        "test": test_dset,
    })
    train_eval_save(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        logging_args=logging_args,
        model_kwargs=model_kwargs,
        raw_datasets=precomputed_datasets,
        tokenizer=tokenizer,
        last_checkpoint=last_checkpoint
    )
    return


if __name__ == "__main__":
    main()