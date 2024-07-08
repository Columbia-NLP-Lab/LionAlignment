# LIONs Alignment

This is the repo for LIONs: An Empirically Optimized Approach to Align Language Models.

# Model Releases

## Gemma Models

https://huggingface.co/Columbia-NLP/LION-Gemma-2b-sft-v1.0
https://huggingface.co/Columbia-NLP/LION-Gemma-2b-dpo-v1.0
[Columbia-NLP/LION-Gemma-2b-odpo-v1.0
](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-odpo-v1.0)


## LLaMA Models

[Columbia-NLP/LION-LLaMA-3-8b-sft-v1.0](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-sft-v1.0)
https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-dpo-v1.0
https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0

# Installation

```sh
pip install -r requirements.txt
```

# Training

## Stage 1: SFT

```
accelerate launch --config_file scripts/run_sft.py config/xxx
```

## Stage 2: DPO

```sh
accelerate launch --config_file scripts/run_sft.py config/xxx
```

## Stage 3: Online DPO

Online DPO is split into two parts:

### Generate data
```sh
```

### Training
```sh
```

# Citation

If you find our repo useful, please consider cite it as follows:
```bibtex
```



