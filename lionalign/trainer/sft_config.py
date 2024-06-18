# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from transformers import TrainingArguments


@dataclass
class SFTConfig(TrainingArguments):
    label_pad_token_id: int = -100
    max_seq_length: Optional[int] = 8192
    use_fast_model: Optional[bool] = True
    mask_user_labels: Optional[bool] = True
    mask_embed_grad: Optional[bool] = True
    shuffle_train_dataloader: Optional[bool] = True
    assistant_bos: Optional[str] = "<|assistant|>"
    assistant_eos: Optional[str] = "<eos>"
    dataset_cache_dir: Optional[str] = "dataset_cache"