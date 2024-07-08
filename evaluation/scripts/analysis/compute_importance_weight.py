from src.utils.utils import create_dir_if_not_exists
from src.trainers.configs import (
    DataArguments, DPOConfig, ModelArguments,
    H4ArgumentParser
)
from src.dataloads.data import maybe_insert_system_message, get_datasets
from src.constants import (
    DPO_DATA_COLUMNS_TO_REMOVE,
    NO_SPECIAL_TOKEN_CHAT_TEMPLATE,
    VICUNA_CHAT_TEMPLATE, ULTRALM_CHAT_TEMPLATE, STAR_CHAT_TEMPLATE, ALPACA_CHAT_TEMPLATE
)
from src.utils.model_utils import get_tokenizer
from src.utils.utils import init_logger, is_main
from transformers import set_seed
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from trl import DPOTrainer
from typing import Literal
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
    load_in_8bit: bool = field(
        default=False,
    )
    load_in_4bit: bool = field(
        default=False,
    )
    llm_int8_enable_fp32_cpu_offload: bool = field(
        default=False,
    )
    do_quantization: bool = field(
        default=False,
    )
    
    def __post_init__(self):
        if self.load_in_8bit or self.load_in_4bit:
            self.do_quantization = True
        return


def add_full_id(data_dict: dict):
    text_chosen = data_dict['chosen']
    text_rejected = data_dict['rejected']
    full_encoded = f"{text_chosen} {text_rejected}"
    full_encoded_id = hashlib.sha256(full_encoded.encode("utf-8")).hexdigest()
    data_dict['full_id'] = full_encoded_id
    return data_dict


def maybe_swap_system_and_assistant(messages, tokenizer):
    # check the last
    tmp_formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    if messages[-1]['content'] in tmp_formatted:
        # we are probably fine
        return messages
    else:
        chat_template = tokenizer.chat_template
        if chat_template is None:
            chat_template = tokenizer.default_chat_template  # get_tokenizer will set this
        
        assert 'system' in chat_template or 'assistant' in chat_template
        # we need to swap the system with assistant, or vice versa
        for idx, message in enumerate(messages[1:]):
            if message['role'] == 'system':
                messages[idx+1]['role'] = 'assistant'
            elif message['role'] == 'assistant':
                messages[idx+1]['role'] = 'system'
    return messages


def subtract_prompt(prompt_text, text):
    assert prompt_text in text, f"Prompt text {prompt_text} not found in text {text}"
    return text.replace(prompt_text, '').strip()


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt

            # Problem: some tokenizer CANNOT tokenize message = [{'role': 'assistant', 'content': ''}] alone
            # Solution: we format the full string first, then we subtract the prompt to extract the chosen and rejected.
            prompt_messages = example["chosen"][:-1]
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            maybe_swap_system_and_assistant(prompt_messages, tokenizer)  # some data uses 'system' role instead of 'assistant', and vice versa
            maybe_swap_system_and_assistant(chosen_messages, tokenizer)
            maybe_swap_system_and_assistant(rejected_messages, tokenizer)
            
            # things are complicated because trainer.build_tokenized_answer cuts prompt + answer
            full_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            full_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            full_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

            example["text_chosen"] = subtract_prompt(full_prompt, full_chosen)
            example["text_rejected"] = subtract_prompt(full_prompt, full_rejected)
            example["text_prompt"] = full_prompt
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def main():
    #### TODO: multi-gpu inference not yet supported
    parser = H4ArgumentParser((CustModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    log_level = training_args.get_process_log_level()
    log_file_path = os.path.join(training_args.output_dir, "log.txt")
    if is_main():
        create_dir_if_not_exists(training_args.output_dir)
    logger = init_logger(is_main=is_main(), log_level=log_level, is_distributed=False, filename=log_file_path)

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
    if 'falcon' in tokenizer.name_or_path:
        logger.warn("Setting bos token id to >>QUESTION<< assuming you are using Falcon tokenizer")
        tokenizer.bos_token_id = 6
        tokenizer.bos_token = tokenizer._convert_id_to_token(6)
    print('tokenizer.pad_token_id', tokenizer.pad_token_id)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad token id for training")
    
    ### set chat template correctly
    if 'WizardLM' in model_args.model_name_or_path or 'vicuna' in tokenizer.name_or_path:
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    elif 'UltraLM' in tokenizer.name_or_path:
        tokenizer.chat_template = ULTRALM_CHAT_TEMPLATE
    elif 'starchat' in tokenizer.name_or_path:
        tokenizer.chat_template = STAR_CHAT_TEMPLATE
    elif 'Alpaca' in model_args.model_name_or_path:
        tokenizer.chat_template = ALPACA_CHAT_TEMPLATE
    elif 'falcon' in tokenizer.name_or_path or 'pythia' in tokenizer.name_or_path or 'mpt' in tokenizer.name_or_path:
        tokenizer.chat_template = NO_SPECIAL_TOKEN_CHAT_TEMPLATE

    logger.info('Chat template:\n' + tokenizer.chat_template)


    ############## NEW!!! add idx
    raw_datasets['train'] = raw_datasets['train'].map(
        add_full_id,
        num_proc=data_args.preprocessing_num_workers,
        keep_in_memory=True,
        desc="Adding full_id to train",
    )

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
    if model_args.do_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            llm_int8_enable_fp32_cpu_offload=model_args.llm_int8_enable_fp32_cpu_offload
        )
    else:
        quantization_config = None

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map='auto',
        quantization_config=quantization_config,
    )

    real_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # used to just tokenize the dataset
    dummy_trainer = DPOTrainer(
        "sshleifer/tiny-gpt2",
        None,
        model_init_kwargs=None,
        ref_model_init_kwargs=None,
        precompute_ref_log_probs=False,
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
    if dummy_trainer.accelerator.is_main_process:
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
    loader = dummy_trainer.get_train_dataloader()
    save_path = os.path.join(training_args.output_dir, "importance.jsonl")

    dummy_trainer.model = real_model
    to_check_model = real_model
    
    pbar = tqdm(total=len(loader), desc="Calculating importance")
    for i, batch in enumerate(loader):
        # log a few examples from the batch
        with torch.no_grad():
            concatenated_batch = dummy_trainer.concatenated_inputs(
                batch,
                is_encoder_decoder=dummy_trainer.is_encoder_decoder,
                label_pad_token_id=dummy_trainer.label_pad_token_id,
                padding_value=dummy_trainer.padding_value,
                device=dummy_trainer.accelerator.device,
            )

            (
                policy_chosen_logps,
                policy_rejected_logps,
                _,
                _,
            ) = dummy_trainer.concatenated_forward(to_check_model, batch)

            policy_chosen_logps = policy_chosen_logps.to('cpu')
            policy_rejected_logps = policy_rejected_logps.to('cpu')

            ### save result
            batch_idx = batch['full_id']
            for i_, idx in enumerate(batch_idx):
                data_idx_to_weights.append({
                    'idx_w_both_resp': idx,
                    'policy_chosen_logps': policy_chosen_logps[i_].item(),
                    'policy_rejected_logps': policy_rejected_logps[i_].item(),
                })
        
        pbar.update(1)
        if dummy_trainer.accelerator.is_main_process:
            if i % 200 == 0:
                with jsonlines.open(save_path, 'w') as writer:
                    writer.write_all(data_idx_to_weights)
                # print a few text
                logger.info("logging a few encoded text")
                decoded = tokenizer.batch_decode(concatenated_batch['concatenated_input_ids'])
                logger.info("========\n".join(decoded))

    if dummy_trainer.accelerator.is_main_process:
        with jsonlines.open(save_path, 'w') as writer:
            writer.write_all(data_idx_to_weights)
    return


if __name__ == "__main__":
    main()