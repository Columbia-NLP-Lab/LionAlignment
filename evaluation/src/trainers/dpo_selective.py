from src.trainers.dpo_fixed import (
    DPOEnhancedTrainer, DPOTrainer,
    build_tokenized_answer
)
from src.trainers.configs import SelectiveDPOConfig
from torch.utils.data import DataLoader
from typing import Dict, Union, Tuple, Any, List
from transformers import PreTrainedModel
from tqdm.auto import tqdm
from contextlib import nullcontext
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import accelerate


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
    batch['full_id'] = feature["full_id"]   # used to check for duplicates later
    batch['score_chosen'] = feature["score_chosen"]
    batch['score_rejected'] = feature["score_rejected"]
    return batch


class DPOSelectiveTrainer(DPOEnhancedTrainer):
    """extending from DPOEnhancedTrainer, but each training step selects from a window
    The goal is to show that the most important samples are the ones that are most relevant

    Args:
        DPOTrainer (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args: SelectiveDPOConfig = self.args
        self.window_size = args.window_size
        self.selective_freq = args.selective_freq
        self.selective_warmup = args.selective_warmup
        self.scoring_fn = args.scoring_fn

        self.score_threshold = args.score_threshold
        self.selectable_data_idx = []
        for idx, data in enumerate(self.train_dataset):
            score_diff = data['score_chosen'] - data['score_rejected']
            if score_diff > self.score_threshold:
                self.selectable_data_idx.append(idx)

        ## to keep track of things
        self._full_idx_to_index = {}
        for idx, data in enumerate(self.train_dataset):
            self._full_idx_to_index[data['full_id']] = idx

        self._selection_history = []
        return

    def save_selection_history(self, force_save=False):
        """save the selection history to a csv file"""
        ## save selective_history
        selection_history = self._selection_history

        if self.accelerator.is_main_process or force_save:
            sh_df = pd.DataFrame(selection_history)
            save_path = os.path.join(self.args.output_dir, "selection_history.csv")
            sh_df.to_csv(save_path, index=False)
        return

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "drop_last": True,  # need this to avoid hanging when gathering
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super(DPOTrainer, self).get_train_dataloader()
    
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], average_log_prob=False
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
            average_log_prob=average_log_prob or self.loss_type == "ipo",  # assume is DPO
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def compute_grad_weight(self, model, resampled_input, device):
        """score the sampled based on the scaling term in DPO loss"""
        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
        ) = self.concatenated_forward(model, resampled_input)

        if "reference_chosen_logps" in resampled_input and "reference_rejected_logps" in resampled_input:
            reference_chosen_logps = resampled_input["reference_chosen_logps"]
            reference_rejected_logps = resampled_input["reference_rejected_logps"]
        else:
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, resampled_input)

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(device)
                - reference_chosen_logps.to(device)
            )
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(device)
                - reference_rejected_logps.to(device)
            )
        )
        scale = torch.sigmoid(rejected_rewards - chosen_rewards).to(device)
        
        flattened_metadata = []
        for i in range(policy_chosen_logps.shape[0]):
            metadata = {
                'chosen_logps': policy_chosen_logps[i].cpu().numpy().tolist(),
                'rejected_logps': policy_rejected_logps[i].cpu().numpy().tolist(),
                'chosen_ref_logps': reference_chosen_logps[i].cpu().numpy().tolist(),
                'rejected_ref_logps': reference_rejected_logps[i].cpu().numpy().tolist(),
            }
            flattened_metadata.append(metadata)
        return scale.flatten(), flattened_metadata

    def compute_lose_win_ratio(self, model, resampled_input, device):
        """score the sampled based on:
        
        log p_{rejected} - log p_{chosen} = log frac{p_{rejected}}{p_{chosen}}

        to prevent longer response automatically losing, we use average log prob as sequence log prob
        """
        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
        ) = self.concatenated_forward(model, resampled_input, average_log_prob=True)

        # reference probs are technically not necessary, but just to keep track of things
        if "reference_chosen_logps" in resampled_input and "reference_rejected_logps" in resampled_input:
            reference_chosen_logps = resampled_input["reference_chosen_logps"]
            reference_rejected_logps = resampled_input["reference_rejected_logps"]
        else:
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, resampled_input, average_log_prob=True)

        score = policy_rejected_logps - policy_chosen_logps
        scaled_score = torch.sigmoid(score).to(device)
        
        flattened_metadata = []
        for i in range(policy_chosen_logps.shape[0]):
            metadata = {
                'chosen_logps': policy_chosen_logps[i].cpu().numpy().tolist(),
                'rejected_logps': policy_rejected_logps[i].cpu().numpy().tolist(),
                'chosen_ref_logps': reference_chosen_logps[i].cpu().numpy().tolist(),
                'rejected_ref_logps': reference_rejected_logps[i].cpu().numpy().tolist(),
            }
            flattened_metadata.append(metadata)
        return scaled_score.flatten(), flattened_metadata

    def compute_data_score(self, model, resampled_input, device):
        if self.scoring_fn == 'grad_weight':
            return self.compute_grad_weight(model, resampled_input, device)
        elif self.scoring_fn == 'lose_win_ratio':
            return self.compute_lose_win_ratio(model, resampled_input, device)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_fn}")

    def repick_inputs(self, inputs, model):
        """reselect data to train based on some scoring function

        Args:
            inputs (_type_): _description_
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        # given a batch size, we consider batch_size * self.window_size samples to select from
        # algorithm:
        # 1. each gpu randomly sample self.window_size data from self.train_dataset
        # 2. each gpu computes the gradient scale terms
        # 3. gather all the gradient scale terms and sort by highest
        # 4. each gpu picks the top batch_size samples (offset by world_idx)
        # 5. replace the inputs with the new inputs computed above
        n = self.window_size
        bsz = inputs["prompt_input_ids"].shape[0]
        world_size = self.accelerator.num_processes
        world_idx = self.accelerator.process_index
        device = self.accelerator.device
        
        rand_val = np.random.rand()

        metadata = {
            "global_step": self.state.global_step,
            "world_idx": world_idx,
            "re-picked": False,
            "full_id": inputs['full_id'],
            "new_score": None,
            "ori_full_id": inputs['full_id'],
            "ori_score": None,
            'is_in_train': True if self.is_in_train else False,
            'metadata': {
                'new_chosen_logps': None,
                'new_rejected_logps': None,
                'new_chosen_ref_logps': None,
                'new_rejected_ref_logps': None,
                'ori_chosen_logps': None,
                'ori_rejected_logps': None,
                'ori_chosen_ref_logps': None,
                'ori_rejected_ref_logps': None,
            }
        }

        # noop before the first gradient step
        if self.is_in_train and self.state.global_step >= 1:
            # noop for:
            # - selective_freq amount of time
            # - or selective_warmup amount of time
            if self.state.global_step < self.selective_warmup \
                or rand_val > self.selective_freq:
                return inputs, metadata
            
            ######### 1. sample new data from the dataset
            random_indices__ = np.random.choice(
                range(len(self.selectable_data_idx) - world_size),
                replace=False,
                size=bsz * n
            ) + world_idx  # pray that after gathering there are not too many duplicates
            random_indices = []
            for idx in random_indices__:
                random_indices.append(self.selectable_data_idx[idx])
            random_indices = torch.tensor(random_indices, device=device)

            # add back in the current batch
            current_indices = []
            for full_idx in inputs['full_id']:
                current_indices.append(self._full_idx_to_index[full_idx])
            current_indices = torch.tensor(current_indices, device=device)
            random_indices = torch.cat([random_indices, current_indices])

            # dedup afterwards as we need to do gather first
            random_indices_list = random_indices.cpu().numpy().tolist()

            resampled_inputs = []
            for idx in random_indices_list:
                resampled_inputs.append(self.train_dataset[idx])

            data_loader = DataLoader(
                resampled_inputs,
                shuffle=False,
                collate_fn=self.data_collator,
                batch_size=bsz,
            )

            ######### 2. compute the gradient scale terms
            resampled_inputs_scores = []
            resampled_inputs_metadata = []
            for resampled in data_loader:
                resampled_input = {}
                for k, v in resampled.items():
                    if 'input_ids' in k or 'attention_mask' in k or 'labels' in k:
                        resampled_input[k] = v.to(device)
                    else:
                        resampled_input[k] = v
                
                with torch.no_grad():
                    score, metadata_ = self.compute_data_score(model, resampled_input, device)
                resampled_inputs_scores.append(score)
                resampled_inputs_metadata.extend(metadata_)
            resampled_inputs_scores = torch.concat(resampled_inputs_scores, axis=0)
            
            # update metadata
            current_score = []
            current_chosen_logps = []
            current_rejected_logps = []
            current_chosen_ref_logps = []
            current_rejected_ref_logps = []
            for idx, score, meta in zip(random_indices_list, resampled_inputs_scores, resampled_inputs_metadata):
                if idx in current_indices:
                    current_score.append(score.item())
                    current_chosen_logps.append(meta['chosen_logps'])
                    current_rejected_logps.append(meta['rejected_logps'])
                    current_chosen_ref_logps.append(meta['chosen_ref_logps'])
                    current_rejected_ref_logps.append(meta['rejected_ref_logps'])
            metadata['ori_score'] = current_score
            metadata['metadata']['ori_chosen_logps'] = current_chosen_logps
            metadata['metadata']['ori_rejected_logps'] = current_rejected_logps
            metadata['metadata']['ori_chosen_ref_logps'] = current_chosen_ref_logps
            metadata['metadata']['ori_rejected_ref_logps'] = current_rejected_ref_logps

            ######### 3. gather all the gradient scale terms and sort by highest
            # print(f"len(selectable_data_idx): {len(self.selectable_data_idx)}, threshold: {self.score_threshold}")
            # print(f"world_idx: {world_idx}, resampled_inputs_metadata: {len(resampled_inputs_metadata)}")
            gathered_scores = self.accelerator.gather(resampled_inputs_scores).cpu().numpy().tolist()
            gathered_indices = self.accelerator.gather(random_indices).cpu().numpy().tolist()
            gathered_metadata = accelerate.utils.gather_object(resampled_inputs_metadata)
            # print(f"world_idx: {world_idx}, gathered_scores: {len(gathered_scores)}, gathered_metadata: {len(gathered_metadata)}")

            # dedup after we gathered
            non_dup_scores = []
            non_dup_indices = []
            for scale, idx in zip(gathered_scores, gathered_indices):
                if idx not in set(non_dup_indices):
                    non_dup_scores.append(scale)
                    non_dup_indices.append(idx)
            non_dup_scores = np.array(non_dup_scores)
            non_dup_indices = np.array(non_dup_indices)

            ######### 4. each gpu picks the top batch_size samples (offset by world_idx)
            sorted_indices = np.argsort(non_dup_scores)[::-1]
            start_i = world_idx * bsz
            end_i = min((world_idx + 1) * bsz, len(sorted_indices))
            if start_i >= len(sorted_indices):
                print(f"[SKIPPING] world_idx: {world_idx}, start_i: {start_i}, end_i: {end_i}")
                return inputs, metadata
            
            top_n_idx = sorted_indices[start_i:end_i]
            top_n_data_idx = non_dup_indices[top_n_idx].tolist()
            top_n_scores = non_dup_scores[top_n_idx].tolist()
            if self.state.global_step % 10 == 0:
                print((
                    f"[[world_idx]]: {world_idx}, non_dup_scales: {non_dup_scores}, "
                    f"top_idx: {sorted_indices[start_i:end_i]}, " 
                    f"current_score: {current_score}, "
                    f"len(_selection_history): {len(self._selection_history)}, "
                    f"_selection_history[-10:]: {self._selection_history[-10:]}"
                ))

            ######### 5. replace the inputs with the new inputs computed above
            new_inputs_list = []
            new_input_full_ids = []
            # gather the metadata about the new inputs
            new_chosen_logps = []
            new_rejected_logps = []
            new_chosen_ref_logps = []
            new_rejected_ref_logps = []
            for data_idx, idx in zip(top_n_data_idx, top_n_idx):
                new_inputs = self.train_dataset[data_idx]
                new_inputs_list.append(new_inputs)
                new_input_full_ids.append(new_inputs['full_id'])
                new_chosen_logps.append(gathered_metadata[idx]['chosen_logps'])
                new_rejected_logps.append(gathered_metadata[idx]['rejected_logps'])
                new_chosen_ref_logps.append(gathered_metadata[idx]['chosen_ref_logps'])
                new_rejected_ref_logps.append(gathered_metadata[idx]['rejected_ref_logps'])
            
            metadata['re-picked'] = True
            metadata['full_id'] = new_input_full_ids
            metadata['new_score'] = top_n_scores
            metadata['metadata']['new_chosen_logps'] = new_chosen_logps
            metadata['metadata']['new_rejected_logps'] = new_rejected_logps
            metadata['metadata']['new_chosen_ref_logps'] = new_chosen_ref_logps
            metadata['metadata']['new_rejected_ref_logps'] = new_rejected_ref_logps
            
            # use a dataloader since we need to do padding as well
            batched_data = self.data_collator(new_inputs_list)
            new_inputs = {}
            for k, v in batched_data.items():
                if 'input_ids' in k or 'attention_mask' in k or 'labels' in k:
                    # add a batch dimension
                    new_inputs[k] = v.to(device)
                else:
                    new_inputs[k] = v
            return new_inputs, metadata
        else:
            return inputs, metadata

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
        
        # switch inputs depending on the computed relevance
        inputs, metadata = self.repick_inputs(inputs, model)

        gathered_metadata = [ metadata ]
        input_gathered = accelerate.utils.gather_object(gathered_metadata)
        if self.accelerator.is_main_process:
            self._selection_history.extend(input_gathered)
            if self.state.global_step % 2 == 0:
                self.save_selection_history()

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss