# Evaluation

We provide easy evaluation pipeline for benchmarks such as MT-Bench, Alpaca Eval 2.0, and Arena Hard. It will automatically start a sglang server to speed up the inference.

## Dependencies

Use `requirements.txt` to install all dependencies.

Afterwards, run `export PYTHONPATH=$(pwd)` so all the relative imports would work.

## Evaluation

Before running any evaluation, make sure the following directories exist:
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
└── openllm              # empty folder for storing openllm results
```

For more details on what should and will go into each directory, see following sections.

### Alpaca Eval 2.0

Running evaluation on a model using `weighted_alpaca_eval_gpt4_turbo` as metric (i.e., runs `gpt-4-turbo` as judge and computes length controlled win rate). Similar to Arena Hard Auto, all respones are computed against a baseline answer (defaults to `tatsu-lab/alpaca_eval` in huggingface datasets).

To run this using a single command line, you need to **first install `alpaca_eval`**. To do this:
- if you have python >= 3.10, you can install from the official github:
    ```bash
    pip install git+https://github.com/tatsu-lab/alpaca_eval
    ```
- otherwise, we modified the `alpaca_eval` to work with lower python versions (tested 3.8). You can install it by:
    ```bash
    pip install git+https://github.com/When2RL/alpaca_eval.git
    ```

Note that this will also download a `results` folder, which contains all precomputed model results used for computing the leaderboard. To make the following scripts work, you need to **copy/move that folder to `data/alpaca_eval_results`**.


Then, to run the evaluation:
```bash
CKPT_FOLDER=model_checkpoints
MODEL_NAME=Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0
EVAL_GPU_IDX=3

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_alpaca_eval.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--tokenizer_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--gen_temperature=1.0 \
--use_sglang \
--gen_parallel=16 \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=true \
--num_runs=1
```

Note that `gen_temperature=1.0` is assumed for most other models in this benchmark.

To display the results manually, run:
```bash
RUN_NAME=Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0-evaluation

python src/evaluation/show_alpaca_eval_result.py \
--model_outputs=data/alpaca_eval_results/${RUN_NAME}/model_outputs.json \
--name=${RUN_NAME} \
--output_path=data/alpaca_eval_results/${RUN_NAME}
```


### Arena Hard Auto

Running evaluation on a model using `gpt-4o-2024-05-13` as judge.

```bash
CKPT_FOLDER=model_checkpoints/oil/gemma-2b-dpo-hpsweep
MODEL_NAME=gemma-2b-lion-v0.5-mix-10k-beta0.1-epoch26-from20
EVAL_GPU_IDX=7

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_arena_hard_auto.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--tokenizer_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=16 \
--judge_only=false \  # if you use judge_only=true, then all params before has no effect
--judge_parallel=8 \
--judge_model=gpt-4o-2024-05-13 \
--baseline_model=gpt-4-0314 \
--to_wandb=true \
--num_runs=1
```

note that this will compute battles *against the baseline model*, and all performance is computed based on if the model wins/loses/ties against the baseline model.

To display all the results under a judge (assuming baseline is `gpt-4-0314`):

```bash
python src/evaluation/show_arena_hard_result.py --judge-name=gpt-3.5-turbo-0125 --baseline=gpt-4-0314
```

> For measuring scaling performance of small models such as gemma-2b, you may want to use `gpt-3.5-turbo-0125` as baseline model instead


### MT-Bench

There are currently three ways to do this. The second and third should give the same results. The second methods is perferred if you are working with THIS repository mainly.


**1. Using the native scripts from `fastchat`**:

Say we are interested in the performance of `alignment-handbook/zephyr-7b-sft-full`

1. make sure you already symlinked the `data/mt_bench` to the correct location (TODO)
2. generate answers:
    ```bash
    python -m fastchat.llm_judge.gen_model_answer --model-path alignment-handbook/zephyr-7b-sft-full --model-id zephyr-7b-sft-full
    ```
    where after this we will refer to this model's performance using `zephyr-7b-sft-full`
3. score the results (assumes `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` set)
    ```bash
    python -m fastchat.llm_judge.gen_judgment --model-list zephyr-7b-sft-full --parallel 4 --judge-model gpt-4
    ```
    this by default appends all results to `data/mt_bench/model_judgment/gpt-4_single.jsonl`. To switch to using `gpt-3.5-turbo`, add the flag `"--judge-model gpt-3.5-turbo`.
4. show result (our own script, which shows more details):
    ```bash
    python src/evaluation/show_mt_bench_result.py --model-list zephyr-7b-sft-full --judge-model gpt-4
    ```

You can also browse through the results in a web-brower with:
```bash
python src/evaluation/qa_browser.py --judge-model=gpt-4-0125-preview
```
this browser is modified from the original browser so that you can 1) choose judge model, and 2) show model scores in the pairwise comparison panel.

**2. Using a wrapper script from this repo**:

The main difference is that here we support 1) running multiple iterations + average, 2) report the results to `wandb` automatically, and 3) using sglang to speed up inference.

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/test/run_mt_bench.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--use_sglang \ # whether to use sglang to speed up inference
--gen_parallel 16 \ # used only when you use sglang
--judge_parallel=16 \
--judge_model=gpt-4-0125-preview \
--to_wandb=true \  # log to wandb
--num_runs=2 \  # number of runs to average. TODO: num=3 often hangs
--y # no confirmation prompt
```
where:
- this will log the results to `wandb` by finding the `wandb_id` from the `{model_path}/run_args.yaml` file
- the final performance will also be logged to `{model_path}`. Otherwise, you can specify another directory by, e.g., adding the flag `--result_save_path=model_checkpoints/debug`
- the main implementations for generating model responses/judgments are basically copied from the `fastchat` scripts


If you do not want to use `sglang` or upload to `wandb`, then the equivalent of the above would be:
```bash
CUDA_VISIBLE_DEVICES=7 python scripts/test/run_mt_bench.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME}_templ_chatml \  # use chatml template
--judge_parallel=16 \
--judge_model=gpt-4-0125-preview \
--num_runs=2 \ 
--y # no confirmation prompt
```

**3. Using `run_llm_judge.py` from the `llm_judge_plus` repo**

Essentially the same functionality except for loop runs, but added support for using `sglang` to speed up inference. This requires you to have setup `sglang` correctly. If so, then you can do:

```bash
python run_llm_judge.py \
--model-path ${CKPT_FOLDER}/${MODEL_NAME} \
--model-id ${MODEL_NAME} \
--overwrite true
```

When reporting results from this method, make sure you report averages of at least 2 runs. See more details at: https://github.com/When2RL/llm_judge_plus.


### OpenLLM

This assumes you have installed `lm_eval`.

```bash
python scripts/test/run_lm_eval.py \
--model_name_or_path=Columbia-NLP/LION-LLaMA-3-8b-odpo-v1.0 \
--torch_dtype=bfloat16 \
--batch_size=16 \
--output_path=data/openllm/Columbia-NLP_LION-LLaMA-3-8b-odpo-v1.0
```
