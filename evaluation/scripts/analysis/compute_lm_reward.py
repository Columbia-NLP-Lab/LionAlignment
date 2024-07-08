from src.utils.utils import create_dir_if_not_exists
from src.trainers.configs import (
    DataArguments, DPOConfig, ModelArguments,
    H4ArgumentParser
)
from src.dataloads.data import apply_chat_template, get_datasets
from src.constants import DPO_DATA_COLUMNS_TO_REMOVE
from src.utils.model_utils import get_tokenizer
from src.utils.utils import init_logger, is_main
from transformers import set_seed
from transformers.integrations.deepspeed import deepspeed_init
from trl import DPOTrainer
from tqdm.auto import tqdm
from dataclasses import dataclass, field, asdict
import torch
import random
import yaml
import jsonlines
import os
import hashlib


@dataclass
class CustModelArguments(ModelArguments):
    ref_model_name_or_path: str = field(
        default="alignment-handbook/zephyr-7b-dpo-full",
    )


### this is not useful as trainer.concatenated_forward(reference_model, batch) still does not perform gather
def wrap_model(trainer: DPOTrainer, model_to_wrap, dataloader):
    args = trainer.args

    if trainer.is_deepspeed_enabled and trainer.deepspeed is None:
        _, _ = deepspeed_init(trainer, num_training_steps=0, inference=True)

    model = trainer._wrap_model(model_to_wrap, training=False, dataloader=dataloader)

    if len(trainer.accelerator._models) == 0 and model is model_to_wrap:
        model = (
            trainer.accelerator.prepare(model)
            if trainer.is_deepspeed_enabled
            else trainer.accelerator.prepare_model(model, evaluation_mode=True)
        )

    # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    # while ``train`` is running, cast it to the right dtype first and then put on device
    if not trainer.is_in_train:
        if args.fp16_full_eval:
            model = model.to(dtype=torch.float16, device=args.device)
        elif args.bf16_full_eval:
            model = model.to(dtype=torch.bfloat16, device=args.device)
    return model


def add_full_id(data_dict: dict):
    text_prompt = data_dict['prompt']
    text_chosen = data_dict['chosen']
    full_encoded = f"{text_prompt} {text_chosen}"
    full_encoded_id = hashlib.sha256(full_encoded.encode("utf-8")).hexdigest()
    data_dict['full_id'] = full_encoded_id
    return data_dict


def main():
    #### TODO: multi-gpu inference not yet supported
    parser = H4ArgumentParser((CustModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    log_level = training_args.get_process_log_level()
    logger = init_logger(is_main=is_main(), log_level=log_level, is_distributed=False)

    ###########
    # load data
    ###########
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits, shuffle=False)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = DPO_DATA_COLUMNS_TO_REMOVE


    # Load tokenizer and process datasets
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    if len(tokenizer) > 100000 or 'stablelm' in tokenizer.name_or_path:
        logger.warn("Setting pad token id to 100288 assuming you are using StableLM tokenizer")
        tokenizer.pad_token_id = 100288


    # Apply chat template
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
        logger.info(f"Loaded {len(raw_datasets[split])} samples for {split}")
    ############## NEW!!! add idx
    raw_datasets['train'] = raw_datasets['train'].map(
        add_full_id,
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        desc="Adding full_id to train",
    )


    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")


    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    #################
    # configure model
    #################

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None
    )

    model = model_args.model_name_or_path
    ref_model = model_args.ref_model_name_or_path
    ref_model_kwargs = model_kwargs

    #############################################
    # configure trainer for distributed inference
    #############################################

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

    ### save run args
    if trainer.accelerator.is_main_process:
        create_dir_if_not_exists(training_args.output_dir)
        all_run_args = {
            **asdict(model_args),
            **asdict(data_args),
            **asdict(training_args),
        }

        yaml_path = os.path.join(training_args.output_dir, "run_args.yaml")
        with open(yaml_path, "w", encoding="utf-8") as fwrite:
            yaml.dump(all_run_args, fwrite, default_flow_style=False)


    ####################
    # compute importance
    ####################
    data_idx_to_weights = []
    loader = trainer.get_train_dataloader()
    save_path = os.path.join(training_args.output_dir, "importance.jsonl")

    # to_check_model = wrap_model(trainer, trainer.model, loader)
    # reference_model = wrap_model(trainer, trainer.ref_model, loader)
    to_check_model = trainer.model
    reference_model = trainer.ref_model
    
    pbar = tqdm(total=len(loader), desc="Calculating importance")
    for i, batch in enumerate(loader):
        with torch.no_grad():
            (
                policy_chosen_logps,
                policy_rejected_logps,
                _,
                _,
            ) = trainer.concatenated_forward(to_check_model, batch)

            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = trainer.concatenated_forward(reference_model, batch)

            policy_chosen_logps = policy_chosen_logps.to('cpu')
            policy_rejected_logps = policy_rejected_logps.to('cpu')
            reference_chosen_logps = reference_chosen_logps.to('cpu')
            reference_rejected_logps = reference_rejected_logps.to('cpu')

            chosen_rewards = (
                trainer.beta
                * (
                    policy_chosen_logps - reference_chosen_logps
                )
            )
            rejected_rewards = (
                trainer.beta
                * (
                    policy_rejected_logps
                    - reference_rejected_logps
                )
            )

            gradient_weight = torch.nn.functional.sigmoid(rejected_rewards - chosen_rewards)

            ### save result
            batch_idx = batch['full_id']
            for i_, idx in enumerate(batch_idx):
                data_idx_to_weights.append({
                    'idx': idx,
                    'chosen_rewards': chosen_rewards[i_].item(),
                    'rejected_rewards': rejected_rewards[i_].item(),
                    'policy_chosen_logps': policy_chosen_logps[i_].item(),
                    'policy_rejected_logps': policy_rejected_logps[i_].item(),
                    'reference_chosen_logps': reference_chosen_logps[i_].item(),
                    'reference_rejected_logps': reference_rejected_logps[i_].item(),
                    'weight': gradient_weight[i_].item()
                })
        
        pbar.update(1)
        if trainer.accelerator.is_main_process:
            if i % 10 == 0:
                with jsonlines.open(save_path, 'w') as writer:
                    writer.write_all(data_idx_to_weights)

    if trainer.accelerator.is_main_process:
        with jsonlines.open(save_path, 'w') as writer:
            writer.write_all(data_idx_to_weights)
    pbar.close()

    ## printing the last example to check if index is right

    return


if __name__ == "__main__":
    main()