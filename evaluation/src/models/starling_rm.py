import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaPreTrainedModel, LlamaModel
from huggingface_hub import snapshot_download


class StarlingRMModel(nn.Module):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **kwargs)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False, dtype=model.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        
        directory = snapshot_download(model_path)  # e.g., "berkeley-nest/Starling-RM-7B-alpha"
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break
        real_state_dict = torch.load(checkpoint)
        self.load_state_dict(real_state_dict, strict=True)
        self.eval().requires_grad_(False)
        return

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        scores = torch.tensor(scores)
        return scores


class Starling34BRMModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.PAD_ID = 0
        # Initialize weights and apply final processing
        self.post_init()
        return
    
    def get_device(self):
        return self.transformer.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        bs = int(input_ids.shape[0])
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        scores = torch.stack(scores)
        return scores