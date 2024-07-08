# LIONs Alignment

This repository contains the implementation for LIONs: An Empirically Optimized Approach to Align Language Models.

## Model Releases

### Gemma Models

- [LION-Gemma-2b-sft-v1.0](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-sft-v1.0)
- [LION-Gemma-2b-dpo-v1.0](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-dpo-v1.0)
- [LION-Gemma-2b-odpo-v1.0](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-odpo-v1.0)

### LLaMA Models

- [LION-LLaMA-3-8b-sft-v1.0](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-sft-v1.0)
- [LION-LLaMA-3-8b-dpo-v1.0](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-dpo-v1.0)
- [LION-LLaMA-3-8b-odpo-v1.0](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0)

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

You will need to install Flash Attention 2 by running:

```sh
python -m pip install flash-attn --no-build-isolation
```

# Training

## Stage 1: SFT

```
accelerate launch --config_file scripts/run_sft.py config/xxx
```

## Stage 2: DPO

```sh
accelerate launch --config_file scripts/run_dpo.py config/xxx
```

## Stage 3: Online DPO

Online DPO is split into two parts:

### Generate data
```sh
# Add the script to generate data
```

### Training
```sh
# Add the script for online DPO training
```

# Evaluation

For Evaluation, please check the folder `evaluation`.

# Citation

If you find our repo useful, please consider cite it as follows:
```bibtex
```



