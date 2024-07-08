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
import jsonlines
import hashlib
import math
import pandas as pd
import numpy as np
import csv
from transformers import set_seed
from datasets import DatasetDict
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
from src.constants import DPO_DATA_COLUMNS_TO_REMOVE
from dataclasses import asdict, dataclass, field
from trl import DPOTrainer
from typing import Optional


logger: logging.Logger


@dataclass
class RewardImportranceControlledDataArguments(DataArguments):
    """
    modified from DataArguments. Data idx is sorted from HIGHEST importance to LOWEST importance
    """
    
    importance_file_path: str = field(
        default="data/analysis/capybara/dpo_v_sft_importance/importance.jsonl",
        metadata={"help": "Path to the importance file."},
    )
    filter_threshold: Optional[float] = field(
        default=0.1,
        metadata={"help": "Data with importance less than this will be filtered out."},
    )
    # consider bi-modal as two normal distributions
    mean_1: Optional[float] = field(
        default=0.3,
        metadata={"help": "Mean of the first normal distribution."},
    )
    std_1: Optional[float] = field(
        default=0.1,
        metadata={"help": "Standard deviation of the first normal distribution."},
    )
    mean_2: Optional[float] = field(
        default = 0.75,
        metadata={"help": "Mean of the second normal distribution."},
    )
    std_2: Optional[float] = field(
        default = 0.1,
        metadata={"help": "Standard deviation of the second normal distribution."},
    )
    max_data_size: Optional[int] = field(
        default = -1,
        metadata={"help": "Maximum data size to be considered. Use None for all data"},
    )


def get_full_id(data_dict: dict):
    text_prompt = data_dict['prompt']
    text_chosen = data_dict['chosen']
    full_encoded = f"{text_prompt} {text_chosen}"
    full_encoded_id = hashlib.sha256(full_encoded.encode("utf-8")).hexdigest()
    return full_encoded_id


def normpdf(x, mean, sd):
    var = sd**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def add_weight_sample_prob(
    data_dict,
    weight_df: pd.DataFrame,
    all_bins: np.ndarray,
    density_per_bin: np.ndarray,
    data_args: RewardImportranceControlledDataArguments
):
    full_id = get_full_id(data_dict)
    if isinstance(weight_df.loc[full_id], pd.DataFrame):
        # there are duplicates, we skip them
        data_dict['weight'] = -1.0
        data_dict['prob'] = 0.0
        data_dict['full_id'] = full_id
        return data_dict
    
    weight = weight_df.loc[full_id]['weight']
    data_dict['full_id'] = full_id
    data_dict['weight'] = weight

    bin_dix = np.digitize([weight], all_bins, right=False)[0] - 1
    dist_1 = normpdf(weight, data_args.mean_1, data_args.std_1)
    dist_2 = normpdf(weight, data_args.mean_2, data_args.std_2)
    prob = (dist_1 + dist_2) / density_per_bin[bin_dix]
    data_dict['prob'] = prob
    return data_dict


def control_reward_importance(dataset: DatasetDict, data_args: RewardImportranceControlledDataArguments):
    ## read the importance file
    with jsonlines.open(data_args.importance_file_path) as reader:
        data = list(reader)
    
    weight_df = pd.DataFrame(data)
    weight_df.index = weight_df['idx'].values
    all_weights = weight_df['weight'].values

    num_bins = 25
    bins = np.linspace(0, 1.0001, num_bins+1)
    binned_weights_idx = np.digitize(all_weights, bins, right=False) - 1
    density_per_bin = np.histogram(binned_weights_idx, bins=num_bins)[0]

    augmented_dataset = dataset.map(
        add_weight_sample_prob,
        fn_kwargs={
            "weight_df": weight_df,
            "all_bins": bins,
            "density_per_bin": density_per_bin,
            "data_args": data_args
        },
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        desc="Adding weight to the dataset",
    )

    ### step 1. filter
    augmented_dataset = augmented_dataset.filter(
        lambda x: x['weight'] >= data_args.filter_threshold,
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        desc=f"Filtering data with importance <= {data_args.filter_threshold}",
    )

    ### step 2. sample
    if data_args.max_data_size == -1:
        return augmented_dataset, augmented_dataset['full_id']
    
    num_to_sample = min(data_args.max_data_size, len(augmented_dataset))
    all_probs = np.array(augmented_dataset['prob'])
    all_probs_normalized = all_probs / all_probs.sum()
    sampled_idx = np.random.choice(len(all_probs), num_to_sample, replace=False, p=all_probs_normalized)

    augmented_dataset = augmented_dataset.select(sampled_idx, keep_in_memory=True)
    return augmented_dataset


def main():
    parser = H4ArgumentParser((ModelArguments, RewardImportranceControlledDataArguments, LoggingArguments, DPOConfig))
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
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = DPO_DATA_COLUMNS_TO_REMOVE
    column_names.remove("prompt_id")  # used for data analysis

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    if len(tokenizer) > 100000 or 'stablelm' in tokenizer.name_or_path:
        logger.warning("Setting pad token id to 100288 assuming you are using StableLM tokenizer")
        tokenizer.pad_token_id = 100288

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        keep_in_memory=True,  # otherwise it will read from cache and ignore the control_data_size fn
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

    #####################
    # control_reward_importance
    #####################
    raw_datasets['train'] = control_reward_importance(
        dataset=raw_datasets['train'],
        data_args=data_args
    )  # train_data_ids will be saved for analysis
    logger.info(f"raw_datasets['train'] size: {len(raw_datasets['train'])} after control_reward_importance")


    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

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
    ref_model = model
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
    trainer = DPOTrainer(
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
        with open(yaml_path, "w", encoding="utf-8") as fwrite:
            yaml.dump(all_run_args, fwrite, default_flow_style=False)
        
        # save train_data_ids for analysis
        train_data_full_ids = raw_datasets['train']['full_id']
        train_data_prompt_ids = raw_datasets['train']['prompt_id']

        train_data_ids_save_path = os.path.join(training_args.output_dir, "train_data_ids.csv")
        with open(train_data_ids_save_path, "w", newline="\n") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["full_id", "prompt_id"])
            writer.writerows([[fid, pid] for fid, pid in zip(train_data_full_ids, train_data_prompt_ids)])

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

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
    
    logger.info("*** Training complete! ***")
    return


if __name__ == "__main__":
    main()