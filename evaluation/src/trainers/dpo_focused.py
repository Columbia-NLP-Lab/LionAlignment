from src.trainers.dpo_fixed import DPOEnhancedTrainer
from src.trainers.configs import FocusedDPOConfig
from typing import Dict, Union, Tuple, Any, List, Literal
from transformers import PreTrainedModel
from contextlib import nullcontext
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOFocusedTrainer(DPOEnhancedTrainer):
    """extending from DPOEnhancedTrainer, but each training step selects from a window
    The goal is to actually modify the DPO loss of make it focus on on learning the wrong ones

    Args:
        DPOTrainer (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args: FocusedDPOConfig = self.args
        self.modified_loss = args.modified_loss
        
        if self.modified_loss in ["direct_p_diff_kl", "token_p_diff_linear"]:
            self.return_token_logps = True
        else:
            self.return_token_logps = False
        
        if self.modified_loss in ["dpo", "reference_free"]:
            self.average_log_prob = False
        else:
            self.average_log_prob = True
        return

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        return_token_logps: bool = False,
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

        if return_token_logps:
            # return per_token_logps * loss_mask / loss_mask.sum(-1, keepdim=True)
            return per_token_logps, loss_mask

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), loss_mask
        else:
            return (per_token_logps * loss_mask).sum(-1), loss_mask
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]],
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

        all_logps, loss_mask = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.average_log_prob or self.loss_type == "ipo",
            return_token_logps=self.return_token_logps,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_mask = loss_mask[:len_chosen]
        rejected_mask = loss_mask[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_mask, rejected_mask)

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
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_loss_mask,
            rejected_loss_mask,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_loss_mask,
            rejected_loss_mask,
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
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_loss_mask: torch.BoolTensor,
        rejected_loss_mask: torch.BoolTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            loss_mask: Mask indicating which tokens to include in the loss when token level loss is used. Shape: (batch_size, sequence_length)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.modified_loss == "dpo":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.modified_loss == "token_p_diff_linear":
            # inspired from graphically how DPO worked
            # 1. calculate difference
            # 2. calculate loss
            policy_chosen_seq_logps = (policy_chosen_logps * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1)
            policy_rejected_seq_logps = (policy_rejected_logps * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1)
            
            diff = torch.exp(policy_chosen_seq_logps) - torch.exp(policy_rejected_seq_logps)
            diff = diff.detach()

            chosen_nonmasked_logps = policy_chosen_logps[chosen_loss_mask]
            rejected_nonmasked_logps = policy_rejected_logps[rejected_loss_mask]

            rejected_nonmasked_logps_clipped = rejected_nonmasked_logps.clip(max=-1e-6)
            rejected_nonmasked_one_minus_logps = torch.log(1.0 - torch.exp(rejected_nonmasked_logps_clipped))  # more stable

            chosen_scale = (1.0 - diff).unsqueeze(-1)
            rejected_scale = (1.0 + diff).unsqueeze(-1)
            
            chosen_loss = (-1.0 * chosen_nonmasked_logps * chosen_scale).mean(axis=-1)
            rejected_loss = (-1.0 * rejected_nonmasked_one_minus_logps * rejected_scale).mean(axis=-1)

            losses = chosen_loss + rejected_loss

            prefix = "train" if self.is_in_train else "eval"
            metrics = {
                f"{prefix}_losses/chosen_loss": chosen_loss.mean().cpu(),
                f"{prefix}_losses/rejected_loss": rejected_loss.mean().cpu(),
                f"{prefix}_losses/diff": diff.mean().cpu()
            }
            self.store_metrics(metrics, train_eval=prefix)
        elif self.modified_loss == "token_p_diff_ref_linear":
            # inspired from graphically how DPO worked
            # 1. calculate difference
            # 2. calculate loss
            reference_chosen_seq_logps = (reference_chosen_logps * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1)
            reference_rejected_seq_logps = (reference_rejected_logps * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1)
            
            diff = torch.exp(reference_chosen_seq_logps) - torch.exp(reference_rejected_seq_logps)
            diff = diff.detach()

            chosen_nonmasked_logps = policy_chosen_logps[chosen_loss_mask]
            rejected_nonmasked_logps = policy_rejected_logps[rejected_loss_mask]

            rejected_nonmasked_logps_clipped = rejected_nonmasked_logps.clip(max=-1e-6)
            rejected_nonmasked_one_minus_logps = torch.log(1.0 - torch.exp(rejected_nonmasked_logps_clipped))  # more stable

            chosen_scale = (1.0 - diff).unsqueeze(-1)
            rejected_scale = (1.0 + diff).unsqueeze(-1)
            
            chosen_loss = (-1.0 * chosen_nonmasked_logps * chosen_scale).mean(axis=-1)
            rejected_loss = (-1.0 * rejected_nonmasked_one_minus_logps * rejected_scale).mean(axis=-1)

            losses = chosen_loss + rejected_loss

            prefix = "train" if self.is_in_train else "eval"
            metrics = {
                f"{prefix}_losses/chosen_loss": chosen_loss.mean().cpu(),
                f"{prefix}_losses/rejected_loss": rejected_loss.mean().cpu(),
                f"{prefix}_losses/diff": diff.mean().cpu(),
            }
            self.store_metrics(metrics, train_eval=prefix)
        elif self.modified_loss == "seq_p_diff_ref_linear":
            # inspired from graphically how DPO worked
            # 1. calculate difference
            # 2. calculate loss
            # PROBLEM: this is sensitive to the absolute value of the prob
            # consider the case of:
            # - chosen_ref = 0.24
            # - rejected_ref = 0.21
            # - chosen = 0.25
            # - rejected = 0.20
            # the chosen loss will be 1.3447055302862938, and the rejected loss will be 0.229837857853636
            # it will STILL prefer to raise the chosen when even though its already correct
            policy_rejected_logps_clipped = policy_rejected_logps.clip(max=-1e-6)
            policy_rejected_one_minus_logps = torch.log(1.0 - torch.exp(policy_rejected_logps_clipped))  # more stable

            # TODO: alternatively, this can be computed with reference logps
            diff = torch.exp(reference_chosen_logps) - torch.exp(reference_rejected_logps)
            diff = diff.detach()

            chosen_scale = (1.0 - diff).unsqueeze(-1)
            rejected_scale = (1.0 + diff).unsqueeze(-1)
            
            chosen_loss = -1.0 * policy_chosen_logps * chosen_scale
            rejected_loss = -1.0 * policy_rejected_one_minus_logps * rejected_scale

            losses = chosen_loss + rejected_loss

            prefix = "train" if self.is_in_train else "eval"
            metrics = {
                f"{prefix}_losses/chosen_loss": chosen_loss.mean().cpu(),
                f"{prefix}_losses/rejected_loss": rejected_loss.mean().cpu(),
                f"{prefix}_losses/diff": diff.mean().cpu(),
                f"{prefix}_losses/chosen_seq_prob": torch.exp(policy_chosen_logps).mean().cpu(),
                f"{prefix}_losses/rejected_seq_prob": torch.exp(policy_rejected_logps).mean().cpu(),
                f"{prefix}_losses/ref_chosen_seq_prob": torch.exp(reference_chosen_logps).mean().cpu(),
                f"{prefix}_losses/ref_rejected_seq_prob": torch.exp(reference_rejected_logps).mean().cpu(),
            }
            self.store_metrics(metrics, train_eval=prefix)
        elif self.modified_loss == "rescaled_seq_p_diff_ref_linear":
            # inspired from graphically how DPO worked
            # 1. calculate difference
            # 2. calculate loss
            # idea: rescale the prob w.r.t. reference mean prob
            diff = torch.exp(reference_chosen_logps) - torch.exp(reference_rejected_logps)
            diff = diff.detach()

            chosen_scale = (1.0 - diff).unsqueeze(-1)
            rejected_scale = (1.0 + diff).unsqueeze(-1)

            ### rescale chosen and rejected logps
            mean_prob = (torch.exp(reference_chosen_logps) + torch.exp(reference_rejected_logps)) / 2
            shift = 0.5 - mean_prob
            shift = shift.detach()
            
            epsilon = 1e-6
            policy_chosen_logps_rescaled = torch.log(torch.clip(torch.exp(policy_chosen_logps) + shift, min=epsilon))

            policy_rejected_logps_clipped = policy_rejected_logps.clip(max=-1e-6)
            policy_rejected_one_minus_logps = torch.log(1.0 - torch.clip((torch.exp(policy_rejected_logps_clipped) + shift), max=1.0-epsilon))  # more stable
            
            chosen_loss = -1.0 * policy_chosen_logps_rescaled * chosen_scale
            rejected_loss = -1.0 * policy_rejected_one_minus_logps * rejected_scale

            losses = chosen_loss + rejected_loss

            prefix = "train" if self.is_in_train else "eval"
            metrics = {
                f"{prefix}_losses/chosen_loss": chosen_loss.mean().cpu(),
                f"{prefix}_losses/rejected_loss": rejected_loss.mean().cpu(),
                f"{prefix}_losses/diff": diff.mean().cpu(),
                f"{prefix}_losses/chosen_seq_prob": torch.exp(policy_chosen_logps).mean().cpu(),
                f"{prefix}_losses/chosen_seq_prob_rescaled": torch.exp(policy_chosen_logps_rescaled).mean().cpu(),
                f"{prefix}_losses/rejected_seq_prob": torch.exp(policy_rejected_logps).mean().cpu(),
                f"{prefix}_losses/rejected_seq_prob_rescaled": (torch.exp(policy_rejected_logps_clipped) + shift).mean().cpu(),
                f"{prefix}_losses/ref_chosen_seq_prob": torch.exp(reference_chosen_logps).mean().cpu(),
                f"{prefix}_losses/ref_rejected_seq_prob": torch.exp(reference_rejected_logps).mean().cpu(),
            }
            self.store_metrics(metrics, train_eval=prefix)
        elif self.modified_loss == "direct_p_diff":
            # causes degeneration
            tau = 0.1
            diff = torch.exp(policy_chosen_logps) - torch.exp(policy_rejected_logps)
            diff /= tau
            losses = -F.logsigmoid(diff)
        elif self.modified_loss == "direct_p_diff_kl":
            policy_chosen_seq_logps = (policy_chosen_logps * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1)
            policy_rejected_seq_logps = (policy_rejected_logps * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1)

            tau = 0.1
            diff = torch.exp(policy_chosen_seq_logps) - torch.exp(policy_rejected_seq_logps)
            diff /= tau
            sft_loss = -F.logsigmoid(diff)

            # calculate kl divergence (see PPOTrainer for reference)
            beta = 0.1
            kl_pos_div = (policy_chosen_logps - reference_chosen_logps).abs().sum(-1)
            kl_neg_div = (policy_rejected_logps - reference_rejected_logps).abs().sum(-1)
            # kl_neg_div = F.kl_div(
            #     policy_rejected_logps,
            #     target=reference_rejected_logps,
            #     log_target=True,
            #     reduction="batchmean"
            # )
            kl_div = (kl_pos_div + kl_neg_div) / 2.0
            losses = sft_loss + beta * kl_div

            prefix = "train" if self.is_in_train else "eval"
            metrics = {
                f"{prefix}_losses/sft_loss": sft_loss.mean().cpu(),
                f"{prefix}_losses/kl_div": kl_div.mean().cpu()
            }
            self.store_metrics(metrics, train_eval=prefix)
        elif self.modified_loss == "dpo_swapped":
            # check if chosen_prob > rejected_prob. If not, flipped the reference
            with torch.no_grad():
                is_already_correct = policy_chosen_logps > policy_rejected_logps

                new_reference_chosen_logps = reference_chosen_logps.clone()
                new_reference_chosen_logps[~is_already_correct] = reference_rejected_logps[~is_already_correct]

                new_reference_rejected_logps = reference_rejected_logps.clone()
                new_reference_rejected_logps[~is_already_correct] = reference_chosen_logps[~is_already_correct]
            
            ref_logratios = new_reference_chosen_logps - new_reference_rejected_logps
            logits = pi_logratios - ref_logratios

            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.modified_loss == "dpo_mean_incorrect":
            # if we use mean reference for all samples, then its equivalent to reference_free actually
            # here we do reference free only on the pairs that are not correct
            with torch.no_grad():
                is_already_correct = policy_chosen_logps > policy_rejected_logps

                new_reference_chosen_logps = reference_chosen_logps.clone()
                new_reference_chosen_logps[~is_already_correct] = 0.0

                new_reference_rejected_logps = reference_rejected_logps.clone()
                new_reference_rejected_logps[~is_already_correct] = 0.0
            
            ref_logratios = new_reference_chosen_logps - new_reference_rejected_logps
            logits = pi_logratios - ref_logratios

            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.modified_loss == "reference_free":
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            logits = pi_logratios - ref_logratios

            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}."
            )

        if self.return_token_logps:
            policy_chosen_logps = (policy_chosen_logps * chosen_loss_mask).sum(-1)
            reference_chosen_logps = (reference_chosen_logps * chosen_loss_mask).sum(-1)
            policy_rejected_logps = (policy_rejected_logps * rejected_loss_mask).sum(-1)
            reference_rejected_logps = (reference_rejected_logps * rejected_loss_mask).sum(-1)

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

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