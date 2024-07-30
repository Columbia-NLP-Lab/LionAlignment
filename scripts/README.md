# Training


## SFT

For Gemma-2b SFT training, please run:
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/accelerate_configs/deepspeed_zero3_4gpu.yaml scripts/run_sft.py configs/gemma-2b/sft/LION-Gemma-2b-sft-v1.0.yaml
```

For LLaMA-3-8b training, simply replace the config file with:
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/accelerate_configs/deepspeed_zero1_4gpu.yaml scripts/run_sft.py configs/llama-3-8b/sft/LION-LLaMA-3-8b-sft-v1.0.yaml
```

## DPO

DPO is split into two parts for better efficiency. 
The first part is to precompute the logprobs using the model to be fine-tuned as the reference model.
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes=4 configs/gemma-2b/dpo/LION-Gemma-2b-dpo-v1.0.yaml
```

The second part is 

```sh
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/accelerate_configs/deepspeed_zero1_1gpu.yaml scripts/run_precompute_dpo.py configs/gemma-2b/dpo/LION-Gemma-2b-dpo-v1.0.yaml
```