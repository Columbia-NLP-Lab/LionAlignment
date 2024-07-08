# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
from pathlib import Path
from typing import Dict, Union

import torch
import shutil
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer,
    AutoModelForCausalLM, GemmaForCausalLM
)
from transformers.trainer_utils import get_last_checkpoint

from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError

from src.constants import DEFAULT_CHAT_TEMPLATE
from src.trainers.configs import DataArguments, DPOConfig, ModelArguments, SFTConfig


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Union[Dict[str, int], None]:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_quantization_config(model_args: ModelArguments) -> Union[BitsAndBytesConfig, None]:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_tokenizer(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None and tokenizer.default_chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files


def get_checkpoint(training_args: Union[SFTConfig, DPOConfig]) -> Union[Path, None]:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def remove_optimizer_weights(save_dir):
    for checkpoint_dirs in os.listdir(save_dir):
        checkpoint_dir = os.path.join(save_dir, checkpoint_dirs)
        if os.path.isdir(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith('global_step'):
                    optimizer_dir = os.path.join(checkpoint_dir, file)
                    # remove the entire folder. This is used by deepspeed to store optimizer states
                    print('removing global_step', optimizer_dir)
                    shutil.rmtree(optimizer_dir)
                elif file.startswith("optimizer.pt"):
                    optimizer_file = os.path.join(checkpoint_dir, file)
                    print('removing optimizer', optimizer_file)
                    os.remove(optimizer_file)
    return


def fix_deepspeed_model_save(model_saved_path, dtype=torch.bfloat16):
    # for some model, we need to manually fix something when deepspeed save the model
    model = AutoModelForCausalLM.from_pretrained(model_saved_path, torch_dtype=dtype, device_map='cpu')
    if isinstance(model, GemmaForCausalLM):
        print("Fixing Gemma model")
        state_dict = model.state_dict()
        true_type = type(state_dict)
        fixed_state_dict = true_type({
            k: v.clone().cpu()
            for k, v in state_dict.items()
            if k != "lm_head.weight"
        })
        model.save_pretrained(
            model_saved_path, state_dict=fixed_state_dict, safe_serialization=True
        )
    return