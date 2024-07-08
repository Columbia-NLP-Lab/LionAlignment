import logging
import random
import gc
import os
import sys
import torch.distributed
import tqdm
import json

from dataclasses import dataclass, field
import datasets
from datasets import concatenate_datasets

from accelerate import Accelerator
from accelerate.state import PartialState
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from lionalign.trainer.dpo_config import DPOConfig
from lionalign.data.utils import get_datasets
from lionalign.data.dpo_data_processor import DPODatasetProcessor
from lionalign.fastmodels.llama.modeling_llama import LlamaForCausalLM

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


from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from lionllm.trainer.utils import move_to_device

logger = logging.getLogger(__name__)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    label_pad_token_id: int = -100,
    padding_value: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        label_pad_token_id: The label pad token id.
        padding_value: The padding value to use for the concatenated inputs_ids.
        device: The device for the concatenated inputs.

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = defaultdict(list)
    # append all chosen
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key].extend([item for item in batch[k]])

    # append all rejected
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key].extend([item for item in batch[k]])

    # pad the concatenated inputs
    for k in concatenated_batch:
        if "labels" in k:
            pad_value = label_pad_token_id
        elif k.endswith("_input_ids"):
            pad_value = padding_value
        elif k.endswith("_attention_mask"):
            pad_value = False
        concatenated_batch[k] = pad_sequence(
            concatenated_batch[k], batch_first=True, padding_value=pad_value
        ).to(device=device)
    return concatenated_batch


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_forward(model, batch):
    device = batch["chosen_labels"].device
    concatenated_batch = concatenated_inputs(
        batch,
        label_pad_token_id=-100,
        padding_value=0,
        device=device,
    )
    len_chosen = batch["chosen_labels"].shape[0]

    all_logits = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
    ).logits

    all_logps = get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        average_log_prob=False,
        label_pad_token_id=-100,
    )

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)


# @dataclass
# class PrecomputeDPOConfig(DPOConfig):
#     world_size: int = 4
#     rank: int = 0
#     save_name = "data"


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = get_tokenizer(model_args, data_args)

    data_processor = DPODatasetProcessor(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        num_proc=data_args.preprocessing_num_workers,
        pad_to_multiple=1,
        use_fast_model=training_args.use_fast_model,
    )

    def process_chat_template(datasets):
        remove_column_names = list(datasets.features)
        # We don't want to remove the dataset_mix_source column
        if "dataset_mix_source" in remove_column_names:
            remove_column_names.remove("dataset_mix_source")

        datasets = datasets.map(
            apply_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "task": "dpo",
                "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            },
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=remove_column_names,
            desc="Formatting comparisons with prompt template",
        )
        # Process the datasets tokenize and label the data
        datasets = data_processor(datasets)
        return datasets

    with PartialState().local_main_process_first():
        os.makedirs(training_args.dataset_cache_dir, exist_ok=True)

        def load_and_process_dpo_dataset(data_args, training_args, tokenizer, eval=False):
            mixer = data_args.eval_dataset_mixer if eval else data_args.train_dataset_mixer
            split = data_args.eval_dataset_split if eval else data_args.train_dataset_split
            shuffle = False if eval else training_args.shuffle_train_dataloader

            # compute the hash of the train dataset
            dataset_cache_hash = get_dataset_cache_hash(
                mixer,
                split,
                tokenizer.chat_template,
                data_args.auto_insert_empty_system_msg,
                shuffle,
                training_args.seed,
            )
            dataset_cache_path = os.path.join(
                training_args.dataset_cache_dir, dataset_cache_hash
            )
                
            # Load train dataset from cache if it exists
            if os.path.exists(dataset_cache_path):
                logger.info(f"Loading dataset from cache {dataset_cache_path}")
                output_dataset = datasets.load_from_disk(dataset_cache_path)
            else:
                output_dataset = get_datasets(
                    mixer,
                    split,
                    dedup_key="chosen",  # we use the chosen message as dedup key
                    columns_to_keep=[
                        "messages",
                        "chosen",
                        "rejected",
                        "prompt",
                        "completion",
                        "label",
                    ],
                    process_fn=process_chat_template,
                    num_proc=data_args.preprocessing_num_workers,
                    seed=training_args.seed,
                    shuffle=shuffle,
                )

                ##########################
                # Get dataset statistics
                ##########################
                if training_args.local_rank in [-1, 0]:
                    dataset_names = set(output_dataset["dataset_mix_source"])
                    dataset_numbers = {dataset_name: 0 for dataset_name in dataset_names}

                    for item in tqdm.tqdm(output_dataset, desc="Counting dataset examples"):
                        dataset_numbers[item["dataset_mix_source"]] += 1

                    # print percentage of different sources and number of examples
                    total = len(output_dataset)
                    dataset_ratio = {
                        dataset_name: dataset_numbers[dataset_name] / total
                        for dataset_name in dataset_numbers
                    }

                    print(f"Total number of examples: {total}")
                    for dataset_name in dataset_ratio.keys():
                        print(
                            f"Dataset {dataset_name} has {dataset_numbers[dataset_name]} examples "
                            f"({dataset_ratio[dataset_name]*100:.2f}%)"
                        )

                output_dataset = data_processor(output_dataset)
                
                if eval:
                    logger.info(
                        f"Eval dataset processed. Number of samples: {len(output_dataset)}"
                    )
                else:
                    logger.info(
                        f"Training Dataset processed. Number of samples: {len(output_dataset)}"
                    )

                if training_args.local_rank in [-1, 0]:
                    logger.info(f"Saving dataset to cache {dataset_cache_path}")
                    output_dataset.save_to_disk(dataset_cache_path)

                    # save dataset statistics
                    dataset_statistics_path = os.path.join(dataset_cache_path, "dataset_statistics.json")
                    dataset_stats = {"total": total, "dataset_numbers": dataset_numbers, "dataset_ratio": dataset_ratio}
                    with open(dataset_statistics_path, "w") as f:
                        json.dump(dataset_stats, f, indent=4)

            return output_dataset, dataset_cache_path
        
        train_dataset, train_dataset_cache_path = load_and_process_dpo_dataset(data_args, training_args, tokenizer, eval=False)
        if data_args.eval_dataset_mixer:
            eval_dataset, eval_dataset_cache_path = load_and_process_dpo_dataset(data_args, training_args, tokenizer, eval=True)
        else:
            eval_dataset, eval_dataset_cache_path = None, None

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
        model_config = transformers.PretrainedConfig.from_pretrained(model_args.model_name_or_path)

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

    accelerator = Accelerator()

    model = model.to(training_args.device)
    model = accelerator.prepare(model)
    model.eval()

    ############################
    #  Prepare train dataloader
    ############################

    # Shard the train dataset
    world_size = training_args.world_size
    rank = training_args.local_rank if training_args.local_rank != -1 else 0
    
    
    train_dataset.name = "Train"

    if eval_dataset is not None:
        eval_dataset.name = "Eval"


    for dataset, dataset_cache_path in zip([train_dataset, eval_dataset], [train_dataset_cache_path, eval_dataset_cache_path]):
        if dataset is None:
            continue

        dataset = dataset.shard(world_size, rank)

        dataloader_params = {
            "batch_size": training_args.eval_batch_size,
            "collate_fn": data_processor.data_collator,
            "num_workers": data_args.preprocessing_num_workers,
            "pin_memory": training_args.dataloader_pin_memory,
            "shuffle": False,
        }

        # prepare dataloader
        dataloader = DataLoader(dataset, **dataloader_params)

        ############################
        #  Compute reference logps
        ############################
        all_reference_chosen_logps = []
        all_reference_rejected_logps = []
        for padded_batch in tqdm.tqdm(
            iterable=dataloader, desc=f"Calculating dataset reference log probs"
        ):
            padded_batch = move_to_device(padded_batch, training_args.device)

            # compute reference logps
            with torch.no_grad():
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = concatenated_forward(model, padded_batch)

            all_reference_chosen_logps.append(reference_chosen_logps.cpu())
            all_reference_rejected_logps.append(reference_rejected_logps.cpu())

        all_reference_chosen_logps = torch.cat(all_reference_chosen_logps).float().numpy()
        all_reference_rejected_logps = (
            torch.cat(all_reference_rejected_logps).float().numpy()
        )

        dataset = dataset.add_column(
            name="reference_chosen_logps", column=all_reference_chosen_logps
        )
        dataset = dataset.add_column(
            name="reference_rejected_logps", column=all_reference_rejected_logps
        )

        # save train_dataset
        rank = training_args.local_rank if training_args.local_rank != -1 else 0
        world_size = training_args.world_size
        dataset_cache_basename = os.path.basename(dataset_cache_path)
        save_path = os.path.join(
            training_args.dataset_cache_dir,
            f"{dataset_cache_basename}_dpo_precompute_rank_{rank}_world_{world_size}",
        )
        dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()
