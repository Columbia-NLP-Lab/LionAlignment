#!/usr/bin/env python
# coding=utf-8
# modified DPO training script from HuggingFace's zephyr repo to be more memory efficient


import logging
import torch
from pathlib import Path
from datasets import concatenate_datasets
from dataclasses import dataclass, field
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
)
from src.utils.utils import init_logger, is_main, create_dir_if_not_exists
from src.utils.data_utils import add_full_id
from src.constants import DPO_DATA_COLUMNS_TO_REMOVE, DPO_DATA_MIX_COLUMNS
from trl import DPOTrainer


logger: logging.Logger



@dataclass
class PrecomputeDataArguments(DataArguments):
    """
    data arguments for precomputing reference log probs
    """

    precompute_file_path: str = field(
        default="data/precompute/tmp.csv",
        metadata={"help": "Path to save precomputed reference log probs"}
    )
    ## compatibility with other configs. Not used in this script
    id_file_path: str = field(
        default="",
        metadata={"help": "Path to the ID file containing the prompt_ids to use for training."},
    )
    id_column_name: str = field(
        default="idx_w_both_resp",
        metadata={"help": "Column name in the ID file that contains the prompt_ids."},
    )
    max_data_size: int = field(
        default=-1,
        metadata={"help": "Maximum number of samples to use for training. -1 for all data."},
    )
    use_avg_logps: bool = field(
        default=False,
        metadata={"help": "Use average log probabilities or sum log probabilities."},
    )


def main():
    parser = H4ArgumentParser((ModelArguments, PrecomputeDataArguments, LoggingArguments, DPOConfig))
    model_args, data_args, _, training_args = parser.parse()

    if is_main():
        fpath = Path(data_args.precompute_file_path)
        dirpath = fpath.parent
        create_dir_if_not_exists(dirpath)

        ## check if file already exists
        if fpath.exists():
            msg = f"""
            File {fpath} already exists. You are currently computing for:
            - dataset_mixer: {data_args.dataset_mixer}
            - model_name_or_path: {model_args.model_name_or_path}
            - max_length: {training_args.max_length}
            """.replace(" "*4, "")
            print(msg)
            if input("Enter [y] to overwrite, [n] to exit: ").strip().lower() != 'y':
                print("Exiting precompute_ref_logprobs.py...")
                exit(0)

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

    ############## NEW!!! add idx. This is so that precomputed log probs can be read later
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

    # concatenate train and test since we are precomputing
    raw_datasets['train'] = concatenate_datasets([raw_datasets['train'], raw_datasets['test']])

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map='auto'
    )

    model = model_args.model_name_or_path
    ref_model = None
    ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer in order to precompute reference log probs
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
        precompute_ref_log_probs=True,
    )

    ###############
    # Training loop
    ###############
    print("[[data_args.use_avg_logps]]", data_args.use_avg_logps)
    trainer.loss_type = 'ipo' if data_args.use_avg_logps else 'dpo'  # trl uses average log prob when loss type is ipo
    _= trainer.get_train_dataloader()

    train_dset = trainer.train_dataset
    train_dset_df = train_dset.to_pandas()

    # save it to disk
    if trainer.accelerator.is_main_process:
        train_dset_df = train_dset_df[['full_id', 'prompt_id', 'reference_chosen_logps', 'reference_rejected_logps']]
        train_dset_df.to_csv(data_args.precompute_file_path, index=False)
    
    logger.info("*** Precompute complete! ***")
    return


if __name__ == "__main__":
    main()