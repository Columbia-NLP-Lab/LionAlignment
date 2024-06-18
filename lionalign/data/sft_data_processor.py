from typing import List, Tuple, Optional, Literal, Union, Dict, Any
import os
import copy
import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .utils import get_datasets, get_stage_datasets

def combine_cu_seqlens(batch_cu_seqlens, attention_mask=None):
    # TODO: handle attention mask
    cu_seqlens = [0]

    for i in range(len(batch_cu_seqlens)):
        if i == 0:
            start_idx = 0
        else:
            start_idx = cu_seqlens[-1]
        for j in range(len(batch_cu_seqlens[i])):
            cu_seqlens.append(start_idx + batch_cu_seqlens[i][j])

    return cu_seqlens


class LabelsMasking:
    """
    Mask the labels of the model output to ignore the user input and system messages.
    """

    def __init__(
        self,
        tokenizer,
        assistant_bos: Union[str, List[int]],
        assistant_eos: Union[str, List[int]],
        ignore_index: int = -100,
        prob_no_mask: float = 0.0,
    ):
        self.prob_no_mask = prob_no_mask
        self.tokenizer = tokenizer
        self.assistant_bos = assistant_bos
        if isinstance(assistant_bos, str):
            # The user provides a string, must tokenize
            self.assistant_bos_token_ids = self.tokenizer.encode(self.assistant_bos, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.assistant_bos_token_ids = assistant_bos

        self.assistant_eos = assistant_eos
        if isinstance(assistant_eos, str):
            # The user provides a string, must tokenize
            self.assistant_eos_token_ids = self.tokenizer.encode(self.assistant_eos, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.assistant_eos_token_ids = assistant_eos

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def __call__(self, labels: Union[torch.Tensor, List[int]]) -> torch.LongTensor:
        if self.prob_no_mask > 0 and np.random.rand() < self.prob_no_mask:
            return labels

        if isinstance(labels, list):
            labels = torch.tensor(labels)

        if isinstance(labels, torch.Tensor):
            assert labels.dim() == 1, "LabelsProcessorForChat expects a 1D tensor of token ids."

        labels = labels.clone()

        assistant_bos_idxs = []
        eos_idxs = []

        for assistant_idx in np.where(labels == self.assistant_bos_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (
                self.assistant_bos_token_ids
                == labels[assistant_idx : assistant_idx + len(self.assistant_bos_token_ids)].tolist()
            ):
                assistant_bos_idxs.append(assistant_idx + len(self.assistant_bos_token_ids))

        if len(assistant_bos_idxs) == 0:
            warnings.warn(
                f"Could not find response key `{self.assistant_bos}` in the "
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            labels[:] = self.ignore_index

        for eos_idx in np.where(labels == self.assistant_eos_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if self.assistant_eos_token_ids == labels[eos_idx : eos_idx + len(self.assistant_eos_token_ids)].tolist():
                eos_idxs.append(eos_idx + len(self.assistant_eos_token_ids))

        indices = []
        for bos_idx in assistant_bos_idxs:
            indices.append((bos_idx, "start"))
        for eos_idx in eos_idxs:
            indices.append((eos_idx, "end"))

        indices = sorted(indices, key=lambda x: x[0])

        # find start, end pairs and set the labels outside to -100
        idx = 0
        prev_end = 0
        while idx < len(indices):
            if indices[idx][1] == "start":
                start = indices[idx][0]
                labels[prev_end:start] = self.ignore_index

                if idx + 1 < len(indices) and indices[idx + 1][1] == "end":
                    end = indices[idx + 1][0]
                    prev_end = end

            idx += 1

        labels[prev_end:] = self.ignore_index
        return labels


class SFTDatasetProcessor:
    def __init__(
        self,
        tokenizer,
        assistant_bos: Union[str, List[int]] = "<|assistant|>:",
        assistant_eos: Union[str, List[int]] = "<eos>",
        max_seq_length: int =4096,
        mask_user_labels: bool =True,
        num_proc: int =8,
        seq_buffer_size: int =1024,
        pad_to_multiple: int = 1,
        ignore_index: int = -100,
        compute_cu_seqlens: bool = True,
    ):
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.max_seq_length = max_seq_length
        self.num_proc = num_proc
        self.seq_buffer_size = seq_buffer_size
        self.pad_to_multiple = pad_to_multiple
        self.ignore_index = ignore_index
        self.mask_user_labels = mask_user_labels
        self.compute_cu_seqlens = compute_cu_seqlens

        self.labels_masking = LabelsMasking(
            tokenizer=tokenizer,
            assistant_bos=assistant_bos,
            assistant_eos=assistant_eos,
        )

    def __call__(self, dataset: Dataset) -> Dataset:
        # check if "text" is in the dataset
        if "text" not in dataset.column_names:
            raise ValueError("Column 'text' not found in dataset. Please ensure the dataset has a 'text' column.")
        
        # first tokenize the data, must remove the "text" column to avoid schema conflicts
        dataset = dataset.map(self.batch_tokenize_data, num_proc=self.num_proc, remove_columns=["text"], batched=True)

        # split long data
        remove_column_names = dataset.column_names
        remove_column_names.remove("input_ids")
        dataset = dataset.remove_columns(remove_column_names)
        dataset = dataset.map(self.batch_split_long_data, batched=True, num_proc=self.num_proc, batch_size=self.seq_buffer_size)

        if self.mask_user_labels:
            # process labels
            dataset = dataset.map(self.process_labels, num_proc=self.num_proc)

        # to tensor
        dataset = dataset.map(self.to_tensor, num_proc=self.num_proc)

        # filter out examples with all labels as -100
        dataset = dataset.filter(self.filter_element, num_proc=self.num_proc)

        return dataset

    def to_tensor(self, element):
        return {k: torch.as_tensor(v, dtype=torch.long) for k, v in element.items()}

    def batch_tokenize_data(self, examples):
        texts = examples["text"]
        return {"input_ids": self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]}

    def process_labels(self, element):
        # TODO: if labels at right side is -100, we can truncate the inputs to improve efficiency
        if "labels" not in element:
            element["labels"] = copy.deepcopy(element["input_ids"])
        element["labels"] = self.labels_masking(element["labels"])
        return element

    def filter_element(self, element):
        # if 90% labels are -100, we filter out the example
        labels = torch.as_tensor(element["labels"])
        if torch.sum(labels.eq(self.ignore_index)).item() > 0.9 * len(labels):
            return False
        return True

    def batch_split_long_data(self, examples):
        all_input_ids = []
        input_ids_buffer = []
        all_cu_seqlens = []
        cu_seqlens = []

        for i in range(len(examples["input_ids"])):
            if len(input_ids_buffer) > 0:
                # add the previous buffer length to cu_seqlens
                cu_seqlens.append(len(input_ids_buffer))

            input_ids = examples["input_ids"][i]
            input_ids_buffer.extend(input_ids)

            while len(input_ids_buffer) > self.max_seq_length:
                all_input_ids.append(input_ids_buffer[: self.max_seq_length])
                input_ids_buffer = input_ids_buffer[self.max_seq_length :]

                # append seqlen in the end
                if not cu_seqlens or cu_seqlens[-1] != self.max_seq_length:
                    cu_seqlens.append(self.max_seq_length)

                all_cu_seqlens.append(cu_seqlens)
                cu_seqlens = []

        return {"input_ids": all_input_ids, "labels": copy.copy(all_input_ids), "cu_seqlens": all_cu_seqlens}

    def data_collator(self, features):
        batch = {}
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        if "attention_mask" not in features[0]:
            attention_mask = [torch.tensor([1] * len(f["input_ids"]), dtype=torch.bool) for f in features]
        else:
            attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.bool) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # TODO: support attention mask
        # batch["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # batch["attention_mask"] = pad_sequence(attention_mask, batch_first=True, padding_value=False)
        # batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

        # force stack to avoid padding
        batch["input_ids"] = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        # ensure attention mask does not include padding tokens
        assert torch.all(attention_mask == 1), "Attention mask should not include padding tokens for fast training."
        batch["labels"] = torch.stack(labels, dim=0)

        # if using fast model, we need to compute cu_seqlens
        if self.compute_cu_seqlens:
            # get cu_seqlens
            all_cu_seqlens = [f["cu_seqlens"] for f in features]
            all_cu_seqlens = combine_cu_seqlens(all_cu_seqlens)
            batch["cu_seqlens"] = torch.tensor(all_cu_seqlens, dtype=torch.int32)

        return batch


