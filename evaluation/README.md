# Evaluation

## Dependencies

Use `full_requirements.txt` to install all dependencies.

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
MODEL_NAME=gemma-2b-lion-v0.7-full-264k-beta0.05-epoch2-bsz64-zero1
EVAL_GPU_IDX=3

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_alpaca_eval.py \
--model_path=${CKPT_FOLDER}/${MODEL_NAME} \
--tokenizer_path=${CKPT_FOLDER}/${MODEL_NAME} \
--model_id=${MODEL_NAME} \
--gen_temperature=1.0 \
--use_sglang \
--gen_parallel=16 \
--chat_template=scripts/configs/chat_templates/hf_gemma_zephyr.json \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=true \
--num_runs=1
```

Note that `gen_temperature=1.0` is assumed for most other models in this benchmark.

To display the results manually, run:
```bash
RUN_NAME=gemma-2b-lion-v0.7-full-264k-beta0.05-epoch2-bsz64-zero1_run0

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
--chat_template=scripts/configs/chat_templates/hf_gemma_zephyr.json \
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
--chat_template scripts/configs/chat_templates/hf_gemma_zephyr.json \ # used only when using sglang
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
--chat-template chat_templates/HuggingFaceH4_zephyr-7b-beta.json \
--overwrite true
```

When reporting results from this method, make sure you report averages of at least 2 runs. See more details at: https://github.com/When2RL/llm_judge_plus.


### OpenLLM

This assumes you have installed the 0.4.1 version of `lm_eval`.

```bash
python scripts/test/run_lm_eval.py \
--model_name_or_path=HuggingFaceH4/zephyr-7b-beta \
--torch_dtype=bfloat16 \
--attn_implementation="flash_attention_2" \
--batch_size=16 \
--output_path=data/openllm/zephyr-7b-beta
```
With the latest version of `lm_eval`, the above can reproduce most results on the official OpenLLM leaderboard.


For stable lm, there seems to be a bug when loading a saved tokenizer, so you may need to do this:
```bash
python scripts/test/run_lm_eval.py \
--model_name_or_path=model_checkpoints_coffee/stablelm-sft-full_bsz32_lr2e-5/checkpoint-11472 \
--tokenizer_name_or_path=stabilityai/stablelm-2-1_6b \
--tokenizer_revision=39d1453f64ffe2a97df9b2f1e6d007eb28248245 \
--torch_dtype=bfloat16 \
--batch_size=16 \
--output_path=data/openllm/stablelm-sft-full_bsz32_lr2e-5
```

## Data Analysis

### Dataset Prediction

Is there a difference between datset x and dataset y? A simple and, as it turns out, quite effective way to do this is to consider the dataset prediction task:
- if the two datasets are easily distinguishable, then the model should be able to predict which dataset a given sample comes from
- if the two datasets are not easily distinguishable, we should then expect accuracy near 50%

Training and testing a `jinaai/jina-embeddings-v2-base-en` to **distinguish between UltraFeedback and UltraChat**. This gives around 86.7% accuracy!
```bash
python scripts/analysis/dset_prediction.py scripts/configs/dset_pred_ultra.yaml \
--output_dir=model_checkpoints/dset_analysis/dset_pred_ultrafeedback_v_ultrachat \
--seed=42
```

Training and testing to **distinguish between UltraChat and UltraChat (dummy test)**. This gives only around 46.7% accuracy.
```bash
python scripts/analysis/dset_prediction_dummy.py scripts/configs/dset_pred_dummy_ultra.yaml \
--output_dir=model_checkpoints/dset_analysis/dset_pred_ultrachat_v_ultrachat \
--seed=42
```

You can test subsets of the dataseta as well by modifying the config file. For example, training and testing to **distinguish between `evo_instruct` subset of UltraFeedback and the UltraChat dataset**:
```yaml
# scripts/configs/dset_pred_ultra.yaml
dataset_to_test:
  when2rl/UltraFeedback_binarized_cleaned_annotated: train_prefs
  HuggingFaceH4/ultrachat_200k: train_sft
# a dict for each dataset
filtering:
  when2rl/UltraFeedback_binarized_cleaned_annotated:
    source: evol_instruct
  HuggingFaceH4/ultrachat_200k:
per_dataset_size:
  train: 1000
  validation: 500
  test: 500
content_to_predict: prompt
```
Then run (which gives a `94.7` accuracy):
```bash
python scripts/analysis/dset_prediction.py scripts/configs/dset_pred_ultra.yaml \
--output_dir=model_checkpoints/dset_analysis/dset_pred_ultrachat_v_evoinstruct \
--seed=42
```


### Dataset Prediction V2

The idea is to check if there are any distinguishing data by comparing
- sub-dataset A directly against sub-dataset B
- data unique in A against data unique in B

This is essentially achieved by reading from `.csv` file that contains the data `full_id` that you want to predict.

```bash
python scripts/analysis/dset_prediction_idfile.py scripts/configs/analysis/dset_pred_idfile.yaml \
--output_dir=data/analysis/ultrafbk/rm-importance-ge0.9_v_score-diff-ge3_confidence \
--seed=42
```

To get those `.csv` file, an example would look like:

```python
## a properly formatted dataset where we added rm_weight and score_diff columns
real_datasets_w_weight_df[real_datasets_w_weight_df['rm_weight'] >= 0.9].join(
    real_datasets_w_weight_df[real_datasets_w_weight_df['score_diff'] < 3.0],
    how='inner',
    lsuffix='_caller',
)['prompt_id'].to_csv(
    "../../data/analysis/ultrafbk/rm-importance-ge0.9_v_score-diff-ge3_confidence/rm-importance-only_train_data_ids.csv",
    index=False
)
```

### Compute LM Reward


(multigpu not yet supported)

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/analysis/compute_lm_reward.py scripts/configs/analysis/lm_reward.yaml \
--model_name_or_path=alignment-handbook/zephyr-7b-sft-full \
--ref_model_name_or_path=model_checkpoints/oil/reprod/zephyr-7b-dpo-full-orca_2epoch \
--output_dir=data/analysis/orca_pairs/dpo_v_sft_importance
```

to change what dataset its evaluated on, modify the `lm_reward.yaml` file.


### Compute (Partial) Importance Weight


(multigpu not yet supported)

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/analysis/compute_importance_weight.py \
scripts/configs/analysis/importance_weight.yaml \
--model_name_or_path=WizardLM/WizardLM-7B-V1.0 \
--output_dir=data/analysis/orca_pairs/WizardLM-7B-V1.0_weights \
--torch_dtype=bfloat16 \
--per_device_train_batch_size=4 \
--use_flash_attention_2=true
```

to change what dataset its evaluated on, modify the `lm_reward.yaml` file.

To load models upto 70B size, the following config works:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/analysis/compute_importance_weight.py \
scripts/configs/analysis/importance_weight.yaml \
--model_name_or_path=WizardLM/WizardLM-70B-V1.0 \
--output_dir=data/analysis/orca_pairs/WizardLM-70B-V1.0_8bit_weights \
--per_device_train_batch_size=1 \
--torch_dtype=bfloat16 \
--use_flash_attention_2=true \
--llm_int8_enable_fp32_cpu_offload=True \
--load_in_8bit=True
```


### Reward Augmentation

First we can recover using gpt-4 as judge to annotate datasets:
```bash
python scripts/analysis/judge_dset.py \
--eval_mode=single \  # {single, all}
--dset_name=when2rl/UltraFeedback_binarized_cleaned_annotated \
--dset_split=train_prefs \
--num_to_judge=150 \  # number of samples to use for annotation
--judge_model=gpt-4-0125-preview \
--judge_parallel=8 \
--output_path=data/analysis/ultrafbk/gpt-4-0125-preview_150-single.csv
```

NOTE: It seems that using `all` gives a different result than `single` mode! With single this is consistent with the original annotation for 92.61% of the time, but with `all` it is only 80.54% of the time.


<!-- TODO: Can we use a RM to filter out and only keep the "high-quality" samples? -->

prelimnary analysis: measure the performance of existing RM models:
```bash
python scripts/analysis/predict_preference.py \
--output_dir model_checkpoints_coffee/dset_analysis/reward_preds/ultrafbk_500_starlingrm \
--model_name_or_path berkeley-nest/Starling-RM-7B-alpha
```

## Generating More Data

Methods to get more data

### Generate and Judge

To generate a model's response given the prompts of a given dataset and then judge it:

1. launch with sglang to speed up inference:
    ```bash
    python -m sglang.launch_server \
    --model-path Columbia-NLP/gemma-2b-lion-sft-v0.1 \
    --port 41911 \
    --enable-flashinfer \
    --attention-reduce-in-fp32 \
    --chat-template scripts/configs/chat_templates/hf_gemma_zephyr.json
    ```

2. genereate model response (TODO: hardcoded sglang url and dset)
    ```bash
    python scripts/gen_data/gen_response.py
    ```

3. generate judgement score
    ```bash
    python scripts/gen_data/judge_gen_data.py \
    --dset_name=when2rl/dpo-mix-7k-rescaled_reformatted \
    --dset_split=train \
    --gen_data_path=data/dpo-mix-7k/gemma_2b_sft.csv \
    --output_path=data/dpo-mix-7k/gemma_2b_sft_gpt4-turbo-scored.csv \
    --num_to_judge=500 # test before going on the full dset (specify -1)
    ```