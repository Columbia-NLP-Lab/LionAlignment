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
import dataclasses
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    if 'typing.Dict' in str(base_type):
                        inputs[arg] = json.loads(val)

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={"help": ("The base model checkpoint for weights initialization with PEFT adatpers.")},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    ref_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The reference model for DPO."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    model_code_revision: str = field(default=None, metadata={"help": "The branch of the IFT model"})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")
        return


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_mixer: Dict[str, float] = field(
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    dataset_splits: Dict[str, List] = field(
        metadata={"help": ("dictionary of dset: [splits] to use for training and testing")},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )


@dataclass
class LoggingArguments:
    """
    Arguments pertaining to remote logging
    """
    wandb_group: Optional[str] = field(
        default='default',
        metadata={"help": ("The wandb group to use for logging.")},
    )
    wandb_project: Optional[str] = field(
        default='when2rl',
        metadata={"help": ("The wandb project to use for logging.")},
    )


@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")


@dataclass
class SFTExtendedConfig(SFTConfig):
    """
    Modified SFT to include:
    - KL divergence
    - sequence level loss #TODO
    """

    kl_coeff: float = field(
        default=0.1,
        metadata={"help": "The KL divergence coefficient."},
    )


@dataclass
class DPOConfig(transformers.TrainingArguments):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": ("For DPO, the maximum length of the prompt to use for conditioning the model.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": ("The loss type for DPO.")})
    precompute_ref_log_probs: bool = field(default=False, metadata={"help": ("Precompute reference log probs.")})


@dataclass
class EfficientDPOConfig(DPOConfig):
    """
    Arguments related to the DPO + TR-DPO (i.e., DPO with moving reference model)
    """
    ref_update_steps: int = field(
        default=-1,
        metadata={"help": "Number of steps to update the reference log probabilities. -1 means no update."}
    )
    # other stuff from the new DPO version
    label_smoothing: Optional[float] = field(
        default=0,
        metadata={"help": "The label smoothing factor in DPO loss."},
    )
    label_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={"help": ("The label padding token id.")}
    )
    padding_value: Optional[int] = field(
        default=0,
        metadata={"help": ("The padding value.")}
    )
    truncation_mode: str = "keep_end"
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False


@dataclass
class WISDPOConfig(DPOConfig):
    """
    Arguments related to the DPO with IS training process itself
    """
    gamma: float = field(
        default=0.01,
        metadata={"help": "The gamma factor to reshape WIS."},
    )
    replace_ref_w_actual: bool = field(
        default=False,
        metadata={"help": "Whether to replace the reference model with the actual model."},
    )
    add_wis: bool = field(
        default=True,
        metadata={"help": "Whether to add WIS scaling to the loss."},
    )
    add_token_level_reward: bool = field(
        default=False,
        metadata={"help": "Whether to add token level reward to the loss."},
    )
    discount_factor: float = field(
        default=0.99,
        metadata={"help": "The discount factor for token level reward."},
    )


@dataclass
class SelectiveDPOConfig(DPOConfig):
    """
    Arguments related to the DPO with selective data during training
    """
    window_size: int = field(
        default=10,
        metadata={"help": "The window size for selective DPO."},
    )
    selective_freq: float = field(
        default=1.0,
        metadata={"help": "The frequency of running the selective data step. Input [0.0, 1.0]."},
    )
    selective_warmup: int = field(
        default=0,
        metadata={"help": "The number of warmup steps before starting selecting data."},
    )
    scoring_fn: str = field(
        default="grad_weight",
        metadata={"help": "The scoring function for selecting data."},
    )
    score_threshold: float = field(
        default=2.0,
        metadata={"help": "The threshold for selecting data."},
    )

    def __post_init__(self):
        if self.scoring_fn not in ["grad_weight", "lose_win_ratio"]:
            raise ValueError(f"Invalid scoring function: {self.scoring_fn}")
        return super().__post_init__()



@dataclass
class FocusedDPOConfig(DPOConfig):
    """
    Arguments related to the DPO with focused learning
    """
    modified_loss: str = field(
        default="dpo",
        metadata={"help": "The loss type for DPO."},
    )

    def __post_init__(self):
        if self.modified_loss not in [
            "dpo",
            "dpo_swapped", "dpo_mean_incorrect",
            "direct_p_diff", "direct_p_diff_kl",
            "token_p_diff_linear", "token_p_diff_ref__linear",
            "seq_p_diff_ref_linear", "rescaled_seq_p_diff_ref_linear",
            "reference_free"
        ]:
            raise ValueError(f"Invalid modified loss: {self.modified_loss}")
        return super().__post_init__()