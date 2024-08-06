<img align="left" width="80" height="80" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT0zHcP2vf_SQ8HgxLp9VaWBGdNaoViPcnyHQ&s">


# LIONs Alignment

The LION-series are trained using an **empirically optimized pipeline** that consists of three stages: SFT, DPO, and online preference learning (online DPO). We find simple techniques such as sequence packing, loss masking in SFT, increasing the preference dataset size in DPO, and online DPO training can significantly improve the performance of language models. Our best models (the LION-series) **exceed the performance of the official instruct models** tuned with closed-source data and algorithms.

## 🚀 Model Releases

We fine-tuned the base Gemma-2b and LLaMA-3-8b models. We released all our models and intermediate checkpoints for reproducibility.

### 💎 Gemma Models

<!---
- [LION-Gemma-2b-sft-v1.0](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-sft-v1.0)
- [LION-Gemma-2b-dpo-v1.0](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-dpo-v1.0)
- [LION-Gemma-2b-odpo-v1.0](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-odpo-v1.0)
--->


| Model | Method | Size | Arena-Hard | AlpacaEval-2 | MT-Bench | OpenLLM |
|-------------|--------|------|------:|------:|---------:|-------:|
|[Gemma-2b](https://huggingface.co/google/gemma-2b) | - | 2B | - | - | - | 46.69 |
|[Gemma-2b-it](https://huggingface.co/google/gemma-2b-it) | SFT+RLHF | 2B | 3.4 | 5.44 | 5.63 | 42.75 |
|[Gemma-2b-zephyr](https://huggingface.co/wandb/gemma-2b-zephyr-dpo) | SFT+DPO | 2B | 0.9 | 2.65 | 4.13 | 46.92 |
|[LLaMA-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | SFT | 7B | 4.6 | 5.35 | 6.22 | 53.16 |
|[Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | SFT | 7B | 2.5 | 7.62 | 6.57 | 52.06 |
|[LION-Gemma-2b-sft-v1.0 (ours)](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-sft-v1.0) | SFT | 2B | 2.4 | 7.79 | 6.37 | 54.78 |
|[LION-Gemma-2b-dpo-v1.0 (ours)](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-dpo-v1.0) | SFT+DPO | 2B | 4.6 | 8.75 | 6.58 | 55.35 |
|[LION-Gemma-2b-odpo-v1.0 (ours)](https://huggingface.co/Columbia-NLP/LION-Gemma-2b-odpo-v1.0) | SFT+DPO+ODPO | 2B | 5.0 | 9.57 | 6.75 | 55.98 |

### 🦙 LLaMA Models

<!---
- [LION-LLaMA-3-8b-sft-v1.0](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-sft-v1.0)
- [LION-LLaMA-3-8b-dpo-v1.0](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-dpo-v1.0)
- [LION-LLaMA-3-8b-odpo-v1.0](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0)
--->

| Model | Method | Size | Arena-Hard | AlpacaEval-2 | MT-Bench | OpenLLM |
|-------------|--------|------|------:|------:|---------:|-------:|
|[LLaMA-3-8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | - | 8B | - | - | - | 63.05 |
|[LLaMA-3-8b-it](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | SFT+RS+DPO+PPO | 8B | 20.6 | 22.9 | 8.00 | 68.28 |
|[LION-LLaMA-3-8b-sft-v1.0 (ours)](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-sft-v1.0) | SFT | 8B | 11.3 | 17.9 | 7.58 | 68.71 |
|[LION-LLaMA-3-8b-dpo-v1.0 (ours)](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-dpo-v1.0) | SFT+DPO | 8B | 19.1 | 21.8 | 8.12 | 71.28 |
|[LION-LLaMA-3-8b-odpo-v1.0 (ours)](https://huggingface.co/Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0) | SFT+DPO+ODPO | 8B | 22.0 | 26.8 | 8.19 | 71.41 |


## 📦 Installation

We recommend using Python 3.10 as the environment. To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

You will need to install Flash Attention 2 by running:

```sh
python -m pip install flash-attn --no-build-isolation
```

## 🏋️ Training

Training requires (at most) 4xA100 80GB GPUs. Please adjust the batch size and gradient accumulation steps if you have a different system.


### Stage 1: SFT


For SFT training, refer to the following script for an example of training LION-LLaMA-3-8b-sft-v1.0:

```sh
./examples/llama-3/sft.sh
```

### Stage 2: DPO

To speed up DPO training, we 1) pre-compute the reference logits and 2) run DPO trainer **without** initializing any reference model. We find this to reduce both the training time and memory usage!

As an example of training LION-LLaMA-3-8b-dpo-v1.0, you can use the following scripts:

```sh
./examples/llama-3/dpo.sh
```

### Stage 3: Online DPO

Online DPO is split into two parts: 1) generate online preference data and 2) train the model with DPO using the online data. As an example, you can train LION-LLaMA-3-8b-odpo-v1.0 with the following script:

```sh
./examples/llama-3/odpo.sh
```


## 📋 Evaluation

We provide easy **ALL-IN-ONE** evaluation pipeline in this directory for popular benchmarks including:

- MT-Bench
- Alpaca Eval 2.0
- Arena Hard Auto
- Huggingface OpenLLM 1.0
- Huggingface OpenLLM 2.0

For each benchmark, we additionally include the following features:

- automatically host your model on an `sglang` server for faster evaluation (except for OpenLLM evals)
- support multiple runs to compute the average performance (`--num_runs`)
- logging performance to `wandb` (`--to_wandb`)


For setup and run details, please refer to [test](test/README.md).


## 📚  Citation

If you find our repo useful, please consider cite it as follows:
```bibtex
@misc{yu2024lionsempiricallyoptimizedapproach,
      title={LIONs: An Empirically Optimized Approach to Align Language Models}, 
      author={Xiao Yu and Qingyang Wu and Yu Li and Zhou Yu},
      year={2024},
      eprint={2407.06542},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.06542}, 
}
```



