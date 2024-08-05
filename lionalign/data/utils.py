from typing import List, Tuple, Optional, Literal, Union, Dict, Any
import os
import copy
import collections
import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import pathlib
from torch.nn.utils.rnn import pad_sequence

import hashlib
import pickle

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from datasets.builder import DatasetGenerationError


def create_dir_if_not_exists(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def maybe_insert_system_message(messages, tokenizer, prob=0.8):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        # since it is maybe, adds a prob not always insert
        if np.random.rand() < prob:
            messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = example["chosen"][:-1]
            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_datasets(
    dataset_mixer: dict,
    splits: Union[str, Dict[str, str]] = "train",
    shuffle: bool = True,
    columns_to_keep: Optional[List[str]] = None,
    dedup_key: str = "messages",
    process_fn: Optional[callable] = None,
    num_proc: int = 64,
    seed: int = 42,
) -> Dataset:
    """
    Get datasets based on the provided dataset mixer.

    Args:
        dataset_mixer (dict): A dictionary mapping dataset names to their corresponding fractions.
        splits (Union[str, Dict[str, str]], optional): The dataset splits to load. Defaults to "train".
        shuffle (bool, optional): Whether to shuffle the datasets. Defaults to True.
        columns_to_keep (List[str], optional): List of columns to keep in the datasets. Defaults to None.
        dedup_key (str, optional): The key to use for deduplication. Defaults to "messages".
        process_fn (callable, optional): A function to process the datasets. Defaults to None.
        num_proc (int, optional): The number of processes to use for parallel processing. Defaults to 64.
        seed (int, optional): The seed value for shuffling. Defaults to 42.

    Returns:
        Dataset: The final combined dataset.
    """
    all_dataset_names = list(dataset_mixer.keys())
    all_dataset_names = list(set(all_dataset_names))
    # show that the following datasets are being loaded
    print(f"Loading datasets: {all_dataset_names}...")

    all_raw_datasets = []
    fracs = []
    for idx, dataset_name in enumerate(
        tqdm.tqdm(all_dataset_names, desc="Loading datasets")
    ):
        # Check if splits are provided for each dataset
        if isinstance(splits, dict):
            dataset_split = splits[dataset_name]
        elif isinstance(splits, str):
            dataset_split = splits
        else:
            raise ValueError(
                f"Invalid split type {splits}. Please provide a string or dictionary."
            )
        
        frac = dataset_mixer[dataset_name]
        fracs.append(frac)

        if dataset_name.startswith("data/") or dataset_name.startswith("dataset/"):
            # local dataset
            dataset = load_from_disk(os.path.join(dataset_name, dataset_split))
        else:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(dataset_name, split=dataset_split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(dataset_name, dataset_split))

        # Remove redundant columns to avoid schema conflicts on load
        if columns_to_keep is not None:
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in columns_to_keep]
            )

        if dedup_key is not None and dedup_key not in dataset.column_names:
            raise ValueError(f"Key {dedup_key} not found in dataset columns.")

        dataset = dataset.add_column(
            "dataset_mix_source", [dataset_name] * len(dataset)
        )

        all_raw_datasets.append(dataset)

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if dedup_key is not None:
        all_raw_datasets = concatenate_datasets(all_raw_datasets)

        # remove duplicates
        all_raw_datasets = all_raw_datasets.map(
            lambda x: {"temp_text": str(x[dedup_key])}, num_proc=num_proc
        )

        # load everything into memory
        texts = all_raw_datasets["temp_text"]

        uniques = collections.defaultdict(list)
        not_dups = []

        # hash all texts and find duplicates
        for i in tqdm.tqdm(range(len(texts)), desc="Finding duplicates"):
            text = texts[i]
            if hash(text) in uniques:
                uniques[hash(text)].append(i)
                not_dups.append(False)
            else:
                uniques[hash(text)].append(i)
                not_dups.append(True)

        # remove duplicates based on hash
        all_raw_datasets = all_raw_datasets.filter(
            lambda example, idx: not_dups[idx], with_indices=True, num_proc=num_proc
        )
        all_raw_datasets = all_raw_datasets.remove_columns(["temp_text"])

        # print the ratio of duplicates
        print(f"Ratio of duplicates: {1 - sum(not_dups) / len(texts)}")

        # split all_raw_data back to different datasets based on source
        datasets_map = {}
        for dataset_name in tqdm.tqdm(
            all_dataset_names, desc="Splitting datasets back to original sources"
        ):
            datasets_map[dataset_name] = all_raw_datasets.filter(
                lambda example: example["dataset_mix_source"] == dataset_name,
                num_proc=num_proc,
            )

        # transform back to list
        all_raw_datasets = list(datasets_map.values())

    if process_fn is not None:
        all_raw_datasets = [process_fn(dataset) for dataset in all_raw_datasets]

    # Combine all datasets based on the mixer fractions
    final_datasets = []
    for idx, dataset_name in enumerate(all_dataset_names):
        dataset = all_raw_datasets[idx]
        frac = dataset_mixer[dataset_name]
        # Repeat dataset frac int times
        repeat = int(frac)
        for _ in range(repeat):
            if shuffle:
                dataset = dataset.shuffle(seed=seed)
            final_datasets.append(dataset)

        # Add the remaining float fraction
        frac = frac % 1
        if frac > 0:
            if shuffle:
                dataset = dataset.shuffle(seed=seed)
            train_subset = dataset.select(range(int(frac * len(dataset))))
            final_datasets.append(train_subset)

    if shuffle:
        final_datasets = concatenate_datasets(final_datasets).shuffle(seed=seed)
    else:
        final_datasets = concatenate_datasets(final_datasets)

    if len(final_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return final_datasets


def get_datasets_xy(
    data_config,
    splits: List[str] = ["train", "test"],
    col_to_mix: List[str] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        col_to_mix:
            if not None, the column to mix after loading the dataset
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    dataset_mixer = data_config.dataset_mixer

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, col_to_mix=col_to_mix, shuffle=shuffle)
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[Union[List[str], Dict[str, list]]] = None, col_to_mix=None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = {}
    raw_val_datasets = []  # this will not be subsampled, so we can just append to it
    # if its a list, we assume the same split names are used for all datasets
    if isinstance(splits, dict):
        for ds, splits_ in splits.items():
            for split in splits_:
                if ds.startswith("data/"):
                    # local dataset
                    dataset = load_from_disk(os.path.join(ds, split))
                else:
                    try:
                        # Try first if dataset on a Hub repo
                        dataset = load_dataset(ds, split=split)
                    except (DatasetGenerationError, ValueError):
                        # If not, check local dataset
                        dataset = load_from_disk(os.path.join(ds, split))
                if col_to_mix is not None:
                    dataset = dataset.select_columns(col_to_mix)

                if "train" in split:
                    if ds not in raw_train_datasets:
                        raw_train_datasets[ds] = []
                    raw_train_datasets[ds].append(dataset)
                elif "test" in split:
                    raw_val_datasets.append(dataset)
                else:
                    raise ValueError(f"Split type {split} not recognized as one of test or train.")
    else:
        raise ValueError(f"Split type {splits} not recognized as one of list or dict.")

    # if any(frac < 0 for frac in fracs):
    #     raise ValueError("Dataset fractions cannot be negative.")
    if any(frac < 0 for frac in dataset_mixer.values()):
        raise ValueError("Dataset fractions cannot be negative.")

    ### now we mix them
    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dsname, dss in raw_train_datasets.items():
            frac = dataset_mixer[dsname]
            for ds in dss:
                train_subset = ds.select(range(int(frac * len(ds))))
                train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


def add_full_id(data_dict: dict):
    text_chosen = data_dict['chosen']
    text_rejected = data_dict['rejected']
    full_encoded = f"{text_chosen} {text_rejected}"
    full_encoded_id = hashlib.sha256(full_encoded.encode("utf-8")).hexdigest()
    data_dict['full_id'] = full_encoded_id
    return data_dict


def _extract_stage_datasets(
    dataset_mixer: dict,
    all_datasets: dict,
    shuffle: bool = True,
    stage_name: str = "",
    columns_to_keep: List[str] = ["messages", "source", "text"],
) -> DatasetDict:
    new_datasets = []
    fracs = []
    for ds, frac in tqdm.tqdm(dataset_mixer.items()):
        fracs.append(frac)
        dataset = all_datasets[ds]

        # Remove redundant columns to avoid schema conflicts on load
        if columns_to_keep is not None:
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in columns_to_keep]
            )
        if stage_name:
            dataset = dataset.add_column("stage", [stage_name] * len(dataset))
        new_datasets.append(dataset)

    if len(new_datasets) > 0:
        dataset_subsets = []
        for dataset, frac in zip(new_datasets, fracs):
            # Repeat dataset frac times
            repeat = int(frac)
            for i in range(repeat):
                dataset_subsets.append(dataset)

            # Add the remaining fraction
            frac = frac % 1
            if frac > 0.01:
                if shuffle:
                    dataset = dataset.shuffle(seed=42)
                train_subset = dataset.select(range(int(frac * len(dataset))))
                dataset_subsets.append(train_subset)

        new_datasets = concatenate_datasets(dataset_subsets)
    else:
        new_datasets = new_datasets[0]

    if shuffle:
        new_datasets = new_datasets.shuffle(seed=42)

    return new_datasets


def get_stage_datasets(
    dataset_mixer: dict,
    columns_to_keep: List[str] = ["messages", "text"],
    num_proc: int = 64,
    process_fn: Optional[callable] = None,
) -> DatasetDict:
    # TODO: currently only support SFT datasets
    # get all dataset names
    all_dataset_names = []
    for ds in dataset_mixer:
        all_dataset_names.extend(list(dataset_mixer[ds].keys()))

    all_dataset_names = list(set(all_dataset_names))
    all_dataset_names.sort()

    print(all_dataset_names)

    # load all datasets
    all_raw_data = []

    for ds in tqdm.tqdm(all_dataset_names):
        dataset = load_dataset(ds, split="train_sft")

        if columns_to_keep is not None:
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in columns_to_keep]
            )
        dataset = dataset.add_column("source", [ds] * len(dataset))
        all_raw_data.append(dataset)

    all_raw_data = concatenate_datasets(all_raw_data)

    # remove duplicates
    all_raw_data = all_raw_data.map(
        lambda x: {"temp_text": str(x["messages"])}, num_proc=num_proc
    )

    # load everything into memory
    texts = all_raw_data["temp_text"]

    uniques = collections.defaultdict(list)
    not_dups = []

    # hash all texts and find duplicates
    for i in tqdm.tqdm(range(len(texts))):
        text = texts[i]
        if hash(text) in uniques:
            uniques[hash(text)].append(i)
            not_dups.append(False)
        else:
            uniques[hash(text)].append(i)
            not_dups.append(True)

    # remove duplicates based on hash
    all_raw_data = all_raw_data.filter(
        lambda example, idx: not_dups[idx], with_indices=True, num_proc=num_proc
    )
    all_raw_data = all_raw_data.remove_columns(["temp_text"])

    if process_fn is not None:
        all_raw_data = process_fn(all_raw_data)

    # print the ratio of duplicates
    print(f"Ratio of duplicates: {1 - sum(not_dups) / len(texts)}")

    # split all_raw_data back to different datasets based on source
    all_datasets = {}
    for ds in tqdm.tqdm(all_dataset_names):
        all_datasets[ds] = all_raw_data.filter(
            lambda example: example["source"] == ds, num_proc=num_proc
        )

    # combine all datasets based on the mixer
    staged_datasets = {}

    for stage in dataset_mixer:
        staged_datasets[stage] = _extract_stage_datasets(
            dataset_mixer[stage], all_datasets, shuffle=True, stage_name=stage
        )

    # Combine all staged datasets
    all_datasets = concatenate_datasets(list(staged_datasets.values()))
    all_datasets_dict = DatasetDict({"train": all_datasets})

    return all_datasets_dict


def adjust_attention_mask(attention_mask, pad_to_multiple: int = 8):
    """
    Adjusts the attention mask by adding ones to make the sum a multiple of a given number.

    Args:
        attention_mask (torch.Tensor): The attention mask tensor.
        pad_to_multiple (int, optional): The number to make the sum of the attention mask a multiple of. Defaults to 8.

    Returns:
        torch.Tensor: The adjusted attention mask tensor.
    """
    # Calculate the current sum of the attention mask
    current_sum = attention_mask.sum().item()

    # Determine how many 1's we need to add to make the sum a multiple of 8
    remainder = current_sum % pad_to_multiple
    if remainder == 0:
        return attention_mask  # Already a multiple of 8

    # Calculate how many 1's need to be added
    to_add = (pad_to_multiple - remainder) % pad_to_multiple

    # Flatten the mask to make it easier to iterate
    flat_mask = attention_mask.flatten()

    # Get the indices of the zeros in the flattened mask
    zero_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]

    # Check if we have enough zeros to replace
    if len(zero_indices) < to_add:
        return attention_mask

    # Replace the required number of zeros with ones
    flat_mask[zero_indices[:to_add]] = 1

    # Reshape the mask back to its original shape
    new_attention_mask = flat_mask.view(attention_mask.shape)

    return new_attention_mask


def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1
) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def hash_obj(obj, length=16):
    # Serialize the dictionary to a bytes-like object
    dict_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    # Create a SHA-256 hash object
    hash_obj = hashlib.sha256()

    # Update the hash object with the serialized bytes
    hash_obj.update(dict_bytes)

    # Get the hexadecimal digest of the hash
    hex_digest = hash_obj.hexdigest()

    # Truncate the hex digest to the specified length
    truncated_hex_digest = str(hex_digest[:length])

    return truncated_hex_digest


def get_dataset_cache_hash(*args, **kwargs):
    """
    Calculate the hash value for caching a dataset.

    Args:
        *args: Variable length arguments.
        **kwargs: Keyword arguments.

    Returns:
        int: The hash value for caching the dataset.
    """
    params = {}
    for i, arg in enumerate(args):
        params[f"arg_{i}"] = arg
    params.update(kwargs)

    return hash_obj(params)