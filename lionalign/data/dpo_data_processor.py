from typing import List, Tuple, Optional, Literal, Union, Dict, Any
from dataclasses import dataclass
import os
import copy
import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset

from .utils import adjust_attention_mask, pad_to_length

class DPODatasetProcessor:
    def __init__(
        self,
        tokenizer,
        max_length=2048,
        max_prompt_length=1024,
        num_proc=8,
        pad_to_multiple: int = 1,
        ignore_index: int = -100,
        truncation_mode: str = "keep_end",
        use_fast_model: bool = True,
    ):
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.num_proc = num_proc
        self.pad_to_multiple = pad_to_multiple
        self.label_pad_token_id = ignore_index
        self.pad_token_id = tokenizer.pad_token_id
        self.truncation_mode = truncation_mode
        self.use_fast_model = use_fast_model

    def __call__(self, dataset: Dataset) -> Dataset:
        dataset = self.rename_columns(dataset)
        dataset = dataset.map(self.tokenize_row, num_proc=self.num_proc)
        return dataset

    def rename_columns(self, dataset: Dataset):
        # rename columns
        if "text_prompt" in dataset.column_names:
            dataset = dataset.rename_column("text_prompt", "prompt")

        if "text_chosen" in dataset.column_names:
            dataset = dataset.rename_column("text_chosen", "chosen")

        if "text_rejected" in dataset.column_names:
            dataset = dataset.rename_column("text_rejected", "rejected")

        return dataset

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][
            len(prompt_input_ids) :
        ]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError(
                "Prompt input ids and answer input ids should have the same length."
            )

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if (
            prompt_input_ids
            != full_tokenized["input_ids"][:response_token_ids_start_idx]
        ):
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][
            :response_token_ids_start_idx
        ]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError(
                "Prompt input ids and attention mask should have the same length."
            )

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][
            response_token_ids_start_idx:
        ]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        # Check issues below for more details
        #  1. https://github.com/huggingface/trl/issues/907
        #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        #  3. https://github.com/LianjiaTech/BELLE/issues/337

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(
            chosen_prompt_len_input_ids, rejected_prompt_len_input_ids
        )

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most an
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [
                a != b
                for a, b in zip(
                    chosen_tokens["prompt_input_ids"],
                    rejected_tokens["prompt_input_ids"],
                )
            ]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt. Avoid adding if it's already there
        bos_token_id = self.tokenizer.bos_token_id
        if (
            prompt_len_input_ids == 0
            or bos_token_id != prompt_tokens["prompt_input_ids"][0]
        ):
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens[
                "prompt_input_ids"
            ]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens[
                "prompt_attention_mask"
            ]
        if (
            chosen_prompt_len_input_ids == 0
            or bos_token_id != chosen_tokens["prompt_input_ids"][0]
        ):
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens[
                "prompt_input_ids"
            ]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens[
                "prompt_attention_mask"
            ]
        if (
            rejected_prompt_len_input_ids == 0
            or bos_token_id != rejected_tokens["prompt_input_ids"][0]
        ):
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens[
                "prompt_input_ids"
            ]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens[
                "prompt_attention_mask"
            ]

        # add EOS token to end of answer. Avoid adding if it's already there
        eos_token_id = self.tokenizer.eos_token_id
        if (
            len(chosen_tokens["input_ids"]) == 0
            or eos_token_id != chosen_tokens["input_ids"][-1]
        ):
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)
        if (
            len(rejected_tokens["input_ids"]) == 0
            or eos_token_id != rejected_tokens["input_ids"][-1]
        ):
            rejected_tokens["input_ids"].append(eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(
            len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
        )

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if (
                len(answer_tokens["prompt_input_ids"]) + longer_response_length
                > self.max_length
            ):
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if (
                len(answer_tokens["prompt_input_ids"]) + longer_response_length
                > self.max_length
            ):
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][
                        : self.max_length - self.max_prompt_length
                    ]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k]
            for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k]
            for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][
            : len(rejected_tokens["prompt_input_ids"])
        ] = [self.label_pad_token_id] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                if type_key in ["input_ids", "labels"]:
                    batch[f"{k}{type_key}"] = torch.LongTensor(tokens)
                elif type_key in ["attention_mask"]:
                    batch[f"{k}{type_key}"] = torch.BoolTensor(tokens)
                else:
                    raise ValueError(f"Unknown type_key: {type_key}")

        return batch

    def data_collator(self, features):
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if "prompt" in k:
                continue

            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):

                if k.endswith("_attention_mask"):
                    dtype = torch.bool
                else:
                    dtype = torch.long
                
                to_pad = [torch.as_tensor(ex[k], dtype=dtype) for ex in features]

                if k.endswith("_input_ids"):
                    if self.pad_token_id is None:
                        raise ValueError(
                            "Padding is enabled, but the tokenizer is not configured with a padding token."
                            " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                            " before calling the trainer."
                        )
                    padding_value = self.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = False
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                # pad to multiple of pad_to_multiple
                if self.pad_to_multiple != 1:
                    length = padded_batch[k].shape[-1]
                    if length % self.pad_to_multiple != 0:
                        padding_length = self.pad_to_multiple - length % self.pad_to_multiple
                        padded_batch[k] = F.pad(
                            padded_batch[k], (0, padding_length), value=padding_value
                        )
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        if self.use_fast_model:
            for k in features[0].keys():
                if k.endswith("_attention_mask"):
                    padded_batch[k] = adjust_attention_mask(
                        padded_batch[k],
                        pad_to_multiple=self.pad_to_multiple
                    )

        return padded_batch