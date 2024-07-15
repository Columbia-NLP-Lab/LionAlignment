import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import tqdm, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from huggingface_hub.utils._deprecation import _deprecate_arguments
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
# from transformers.trainer import *

from trl.import_utils import is_peft_available
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    pad_to_length,
    disable_dropout_in_model,
    peft_module_casting_to_bf16,
)
from lionalign.trainer.efficient_dpo_config import EfficientDPOConfig


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


def move_to_device(data, device, exclude_keys=None):
    """
    Args:
        data: a list, dict, or torch.Tensor
        device: the target torch.device
        exclude_keys: remove unwanted keys
    """
    if exclude_keys is None:
        exclude_keys = []

    # send data to device
    if isinstance(data, list):
        new_data = []
        for item in data:
            if isinstance(item, torch.Tensor):
                new_data.append(item.to(device, non_blocking=True))
            else:
                new_data.append(move_to_device(item, device, exclude_keys))
        data = new_data

    elif isinstance(data, tuple):
        new_data = ()
        for item in data:
            if isinstance(item, torch.Tensor):
                new_data = new_data + (item.to(device, non_blocking=True), )
            else:
                new_data = new_data + (move_to_device(item, device, exclude_keys), )
        data = new_data

    elif isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and all([key not in k for key in exclude_keys]):
                new_data[k] = v.to(device, non_blocking=True)
            else:
                new_data[k] = move_to_device(v, device, exclude_keys)
        data = new_data

    elif isinstance(data, torch.Tensor) or isinstance(data, torch.nn.Module):
        data = data.to(device, non_blocking=True)
    elif isinstance(data, int) or isinstance(data, float):
        data = data
    else:
        # logger.warning(f"{type(data)} cannot be sent to device")
        data = data
    return data


def build_tokenized_answer(tokenizer, prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )
    

def tokenize_row(
    feature,
    tokenizer,
    max_length: int = 1024,
    truncation_mode: str = "keep_end",
    max_prompt_length: int = 512,
    label_pad_token_id: int = -100,
) -> Dict:
    """modified from the original tokenize_row to
    1. avoid adding redundant tokens (BOS, EOS) to the prompt and answer
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
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    if not isinstance(chosen, str):
        raise ValueError(f"chosen should be an str but got {type(chosen)}")
    chosen_tokens = build_tokenized_answer(tokenizer, prompt, chosen)

    if not isinstance(rejected, str):
        raise ValueError(f"rejected should be an str but got {type(rejected)}")
    rejected_tokens = build_tokenized_answer(tokenizer, prompt, rejected)

    # Last prompt token might get merged by tokenizer and
    # it should not be included for generation if that happens
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    # Make sure prompts only have one different token at most an
    # and length only differs by 1 at most
    num_diff_tokens = sum(
        [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    )
    num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the "
            "last token due to tokenizer merge ops."
        )

    # add BOS token to head of prompt. Avoid adding if it's already there
    if tokenizer.bos_token_id != prompt_tokens["prompt_input_ids"][0]:
        prompt_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
    if tokenizer.bos_token_id != chosen_tokens["prompt_input_ids"][0]:
        chosen_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
    if tokenizer.bos_token_id != rejected_tokens["prompt_input_ids"][0]:
        rejected_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]
    

    # add EOS token to end of answer. Avoid adding if it's already there
    if tokenizer.eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

    if tokenizer.eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            if truncation_mode == "keep_start":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: max_prompt_length]
            elif truncation_mode == "keep_end":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            for k in ["input_ids", "attention_mask"]:
                answer_tokens[k] = answer_tokens[k][: max_length - max_prompt_length]

    # Create labels
    chosen_sequence_tokens = {
        k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    rejected_sequence_tokens = {
        k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(chosen_tokens["prompt_input_ids"])
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(rejected_tokens["prompt_input_ids"])

    for k, toks in {
        "chosen_": chosen_sequence_tokens,
        "rejected_": rejected_sequence_tokens,
        "": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}{type_key}"] = tokens
    return batch


class EfficientDPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        args (`DPOConfig`):
            The DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "dpo"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "beta",
            "label_smoothing",
            "label_pad_token_id",
            "padding_value",
            "truncation_mode",
            "max_length",
            "max_prompt_length",
            "max_target_length",
            "is_encoder_decoder",
            "disable_dropout",
            "generate_during_eval",
            "dataset_num_proc",
            "model_init_kwargs",
            "model_adapter_name",
        ],
        custom_message="Deprecated positional argument(s) used in DPOTrainer, please use the DPOConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair"] = "sigmoid",
        args: Optional[EfficientDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        dataset_num_proc: Optional[int] = None,
        model_adapter_name: Optional[str] = None,
    ):

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        # if force_use_ref_model:
        #     warnings.warn(
        #         "You passed `force_use_ref_model` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
        #     )
        #     args.force_use_ref_model = force_use_ref_model

        if peft_config is not None:
            raise ValueError("PEFT is currently not supported.")
        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the DPOTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")


        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = args.max_length
        # self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = args.padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.max_target_length = args.max_target_length
        self.tokenizer = tokenizer


        if loss_type != "sigmoid":
            warnings.warn(
                "You passed `loss_type` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.loss_type = loss_type
        if label_smoothing != 0:
            warnings.warn(
                "You passed `label_smoothing` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_smoothing = label_smoothing
        if args.loss_type in ["hinge", "ipo", "kto_pair", "bco_pair"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        if beta != 0.1:
            warnings.warn(
                "You passed `beta` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.beta = beta

        self.beta = args.beta
        self.ref_update_steps = args.ref_update_steps
        self.num_ref_updates = 0
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # if dataset_num_proc is not None:
        #     warnings.warn(
        #         "You passed `dataset_num_proc` to the DPOTrainer, the value you passed will override the one in the `DPOConfig`."
        #     )
        #     args.dataset_num_proc = dataset_num_proc
        # self.dataset_num_proc = args.dataset_num_proc

        # # Compute that only on the main process for faster data processing.
        # # see: https://github.com/huggingface/trl/pull/1255
        # with PartialState().local_main_process_first():
        #     # tokenize the dataset
        #     train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
        #     if eval_dataset is not None:
        #         eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)
        return

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    def compute_all_reference_log_probs(self, dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "shuffle": False,
        }

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        # manually move model to device as sometimes accelerator might not initialize this properly
        if self.accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage'] == 3:
            print('Detected zero 3 optimization')
            self.model = Accelerator().prepare(self.model)
        else:
            self.model = self.model.to(self.accelerator.device)

        data_samples = []
        for padded_batch in tqdm(iterable=data_loader, desc="calculate dataset reference log probs"):
            reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
            num_samples = len(padded_batch["full_id"])
            if num_samples == 1:
                full_id = padded_batch["full_id"][0]
            
                data_samples.append({
                    "full_id": full_id,
                    "reference_chosen_logps": reference_chosen_logp.cpu().item(),
                    "reference_rejected_logps": reference_rejected_logp.cpu().item(),
                })
            else:
                for i in range(num_samples):
                    full_id = padded_batch["full_id"][i]
                    data_samples.append({
                        "full_id": full_id,
                        "reference_chosen_logps": reference_chosen_logp[i].cpu().item(),
                        "reference_rejected_logps": reference_rejected_logp[i].cpu().item(),
                    })
        
        # gather all
        gathered_data_samples = gather_object(data_samples)
        return gathered_data_samples

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
        ) = self.concatenated_forward(model, batch)

        # assert reference_chosen_logps is in batch
        reference_chosen_logps = batch["reference_chosen_logps"]
        reference_rejected_logps = batch["reference_rejected_logps"]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        gradient_scale = torch.sigmoid(rejected_rewards - chosen_rewards)
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}rewards/scale"] = gradient_scale.mean().cpu()
        metrics[f"{prefix}rewards/scale_max"] = gradient_scale.max().cpu()
        metrics[f"{prefix}rewards/scale_min"] = gradient_scale.min().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        # metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
        return

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)