from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple
import transformers


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
    # precompute save path
    precompute_train_ref_file_path: str = field(
        default="",
        metadata={"help": "The csv fille path to the precomputed reference log probs for training. Use '' to compute on the fly."},
    )
    precompute_test_ref_file_path: str = field(
        default="",
        metadata={"help": "The csv fille path to the precomputed reference log probs for testing."},
    )