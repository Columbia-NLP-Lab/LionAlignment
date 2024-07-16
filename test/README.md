# Evaluation

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

## Dependencies

**First**, install the relevant packages for each benchmark:

- `alpaca_eval` for Alpaca Eval 2.0
    ```bash
    pip install git+https://github.com/tatsu-lab/alpaca_eval
    ```
    Note that this will also download a `results` folder, which contains all precomputed model results used for computing the leaderboard. To make the following scripts work, you need to *copy/move that folder to `data/alpaca_eval_results`*. See the next step for more details.
- `fastchat` for MT-Bench and Arena Hard
    ```bash
    pip install fschat[model_worker,llm_judge]
    pip install gradio==3.48.0
    ```
- `lm_eval` for OpenLLM 1.0 and 2.0
    ```bash
    git clone https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .
    ```

**Next**, configure your `data` folder to include the following directories:

```bash
data/
├── alpaca_eval_results  # copied from `results` folder in the official alpaca_eval repo
│   ├── Conifer-7B-DPO
│   ├── Contextual-KTO-Mistral-PairRM
│   ├── Ein-70B-v0.1
│   └── ... (other model answers)
├── arena-hard-v0.1      # copied from `data` folder in the official Arena Hard Auto repo
│   ├── model_answer
│   ├── model_judgment
│   └── question.jsonl
├── mt_bench             # copied from `data` folder in the official MT-Bench repo
│   ├── model_answer
│   ├── model_judgment
│   ├── question.jsonl
│   └── reference_answer
├── openllm_v2           # empty folder for storing openllm v2 results
└── openllm              # empty folder for storing openllm v1 results
```

**Finally**, ensure `sglang` is installed in your system, which is used by our scripts to *significantly speed up* the evaluation process by automatically hosting your model on a `sglang` server. You can install this by:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install -e "python[all]"

# Install FlashInfer CUDA kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```


## Running Evaluation

We provide examples to run each benchmark.

> Note: all the following utilizes `sglang` to generate responses, which is much faster but relies on manually passing in `chat_template`. Currently built-in ones are:
> ```bash
> # see https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/conversation.py for more details
> llama-2, chatml, vicuna_v1.1
> ```
> This means if you are using any of the above, you can do ``--chat_template=llama-2`` to use the corresponding template. Otherwise, you need to manully write a chat template file and pass it in. Sglang reads it in by doing:
> ```python
> Conversation(
>   name=template["name"],
>   system_template=template["system"] + "\n{system_message}",
>   system_message=template.get("system_message", ""),
>   roles=(template["user"], template["assistant"]),
>   sep_style=sep_style,
>   sep=template.get("sep", "\n"),
>   stop_str=template["stop_str"],
> )
> ```
> where `template` will be your chat template `.json` file.

### AlpacaEval 2.0

To evaluate `Columbia-NLP/LION-Gemma-2b-odpo-v1.0` on AlpacaEval 2.0, run:

```bash
MODEL_PATH=Columbia-NLP/LION-Gemma-2b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-Gemma-2b-odpo-v1.0
EVAL_GPU_IDX=0

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_alpaca_eval.py \
--model_path=${MODEL_PATH} \
--tokenizer_path=${MODEL_PATH} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=16 \
--chat_template=configs/chat_templates/lion-gemma-2b.json \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=false \
--result_save_path=data/alpaca_eval_results/${MODEL_NAME} \
--num_runs=1
```
note that:
- you can also run with a local model by simply modifying `--model_path`. 
- if you used `--to_wandb=true`, it will check for a `run_args.yaml` file under the model path, read the `wandb_id` field, and log results to that wandb run.
- if you are using a local model, yuo can also skip `--result_save_path` as it will default to saving results to your model path.


---

To evaluate `Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0` on AlpacaEval 2.0, run:


```bash
MODEL_PATH=Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0
EVAL_GPU_IDX=2

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_alpaca_eval.py \
--model_path=${MODEL_PATH} \
--tokenizer_path=${MODEL_PATH} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.7 \
--use_sglang \
--gen_parallel=8 \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=false \
--result_save_path=data/alpaca_eval_results/${MODEL_NAME} \
--num_runs=1
```

For LLaMA, we **do not** use `--chat_template` (which will inject a `system` message) and directly use the tokenizer to format chat inputs.


---


To display the results manually, run:
```bash
RUN_NAME=Columbia-NLP_LION-Gemma-2b-odpo-v1.0_run0

python lionalign/evaluation/show_alpaca_eval_result.py \
--output_path=data/alpaca_eval_results/${RUN_NAME}
```


### Arena Hard

To evaluate `Columbia-NLP/LION-Gemma-2b-odpo-v1.0` on Arena Hard, run:

```bash
MODEL_PATH=Columbia-NLP/LION-Gemma-2b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-Gemma-2b-odpo-v1.0
EVAL_GPU_IDX=0

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_arena_hard_auto.py \
--model_path=${MODEL_PATH} \
--tokenizer_path=${MODEL_PATH} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=16 \
--chat_template=configs/chat_templates/lion-gemma-2b.json \
--judge_parallel=8 \
--judge_model=gpt-4o-2024-05-13 \
--baseline_model=gpt-3.5-turbo-0125 \
--to_wandb=false \
--result_save_path=data/arena-hard-v0.1/model_performance/${MODEL_NAME} \
--num_runs=1
```

note that:
- this will use `gpt-4o-2024-05-13` as judge as its faster and cheaper. For full Arena Hard results, use `gpt-4-1106-preview` instead.
- all battles will be computed against the baseline model `gpt-3.5-turbo-0125`. For full Arena Hard results, use `gpt-4-0314` instead.

---

To evaluate `Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0` on Arena Hard, run:

```bash
MODEL_PATH=Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0
EVAL_GPU_IDX=0

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_arena_hard_auto.py \
--model_path=${MODEL_PATH} \
--tokenizer_path=${MODEL_PATH} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=16 \
--judge_only=false \
--judge_parallel=8 \
--judge_model=gpt-4-1106-preview \
--baseline_model=gpt-4-0314 \
--to_wandb=false \
--result_save_path=data/arena-hard-v0.1/model_performance/${MODEL_NAME} \
--num_runs=1
```

For LLaMA, we **do not** use `--chat_template` (which will inject a `system` message) and directly use the tokenizer to format chat inputs.

---


To display all the results under a judge/baseline combo:

```bash
python lionalign/evaluation/show_arena_hard_result.py \
--judge-name=gpt-4o-2024-05-13 \
--baseline=gpt-3.5-turbo-0125
```


### MT-Bench

To evaluate `Columbia-NLP/LION-Gemma-2b-odpo-v1.0` on MT-Bench, run:


```bash
MODEL_PATH=Columbia-NLP/LION-Gemma-2b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-Gemma-2b-odpo-v1.0
EVAL_GPU_IDX=0

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_mt_bench.py \
--model_path=${MODEL_PATH} \
--model_id=${MODEL_NAME} \
--use_sglang \
--gen_parallel=16 \
--chat_template=configs/chat_templates/lion-gemma-2b.json \
--judge_parallel=8 \
--judge_model=gpt-4 \
--to_wandb=false \
--num_runs=1 \
--y \
--result_save_path=data/mt_bench/model_performance/${MODEL_NAME}
```


---

To evaluate `Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0` on MT-Bench, run:


```bash
MODEL_PATH=Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0
EVAL_GPU_IDX=0

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_mt_bench.py \
--model_path=${MODEL_PATH} \
--model_id=${MODEL_NAME} \
--use_sglang \
--gen_parallel=16 \
--judge_parallel=8 \
--judge_model=gpt-4 \
--to_wandb=false \
--num_runs=1 \
--y \
--result_save_path=data/mt_bench/model_performance/${MODEL_NAME}
```

For LLaMA, we **do not** use `--chat_template` (which will inject a `system` message) and directly use the tokenizer to format chat inputs.


---


To display the results manually, run:
```bash
python lionalign/evaluation/show_mt_bench_result.py \
--model-list Columbia-NLP_LION-Gemma-2b-odpo-v1.0 \
--judge-model gpt-4
```


### OpenLLM 2.0

This assumes you have installed the `lm_eval` from the `main` branch, as OpenLLM 2.0 tasks are only recently released in `lm_eval`.

To evaluate `Columbia-NLP/LION-Gemma-2b-odpo-v1.0` on OpenLLM 2.0, run:

```bash
MODEL_PATH=Columbia-NLP/LION-Gemma-2b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-Gemma-2b-odpo-v1.0
EVAL_GPU_IDX=0

rm -rf data/openllm_v2/${MODEL_NAME}

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_lm_eval_v2.py \
--model_name_or_path=${MODEL_PATH} \
--openllm_only=true \
--torch_dtype=bfloat16 \
--attn_implementation="flash_attention_2" \
--output_path=data/openllm_v2/${MODEL_NAME}_bf16 \
--log_samples=false \
--to_wandb=false
```

---

To evaluate `Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0` on OpenLLM 2.0, run:

```bash
MODEL_PATH=Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0
EVAL_GPU_IDX=0

rm -rf data/openllm_v2/${MODEL_NAME}

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_lm_eval_v2.py \
--model_name_or_path=${MODEL_PATH} \
--torch_dtype=bfloat16 \
--attn_implementation="flash_attention_2" \
--batch_size=8 \
--output_path=data/openllm_v2/${MODEL_NAME}_bf16 \
--log_samples=false \
--to_wandb=false
```


### OpenLLM 1.0

To evaluate `Columbia-NLP/LION-Gemma-2b-odpo-v1.0` on OpenLLM 1.0, run:

```bash
MODEL_PATH=Columbia-NLP/LION-Gemma-2b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-Gemma-2b-odpo-v1.0
EVAL_GPU_IDX=0

rm -rf data/openllm/${MODEL_NAME}

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_lm_eval.py \
--model_name_or_path=${MODEL_PATH} \
--openllm_only=true \
--torch_dtype=bfloat16 \
--attn_implementation="flash_attention_2" \
--batch_size=8 \
--output_path=data/openllm/${MODEL_NAME}_bf16 \
--log_samples=false \
--to_wandb=false
```


---

To evaluate `Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0` on OpenLLM 1.0, run:

```bash
MODEL_PATH=Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0
MODEL_NAME=Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0
EVAL_GPU_IDX=0

rm -rf data/openllm/${MODEL_NAME}

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_lm_eval.py \
--model_name_or_path=${MODEL_PATH} \
--torch_dtype=bfloat16 \
--attn_implementation="flash_attention_2" \
--batch_size=8 \
--output_path=data/openllm/${MODEL_NAME}_bf16 \
--log_samples=false \
--to_wandb=false
```