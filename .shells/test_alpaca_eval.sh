export PYTHONPATH=$(pwd)
source ../when2rl/.env


MODEL_NAME_OR_PATH=model_checkpoints_coffee/reprod/LION-Gemma-2b-odpo-v0.9
MODEL_NAME=LION-Gemma-2b-odpo-v0.9
MT_JUDGE_MODEL=gpt-4
AHA_JUDGE_MODEL=gpt-4-1106-preview
AHA_BASELINE_MODEL=gpt-4-0314
EVAL_GPU_IDX=7
CHAT_TEMPLATE=configs/chat_templates/lion-gemma-2b.json


# CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_mt_bench.py \
# --model_path=${MODEL_NAME_OR_PATH} \
# --model_id=${MODEL_NAME} \
# --use_sglang \
# --gen_parallel=16 \
# --chat_template=${CHAT_TEMPLATE} \
# --judge_parallel=8 \
# --judge_model=${MT_JUDGE_MODEL} \
# --to_wandb=false \
# --num_runs=1 \
# --y \
# --result_save_path=data/mt_bench/model_performance/${MODEL_NAME}


# CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_arena_hard_auto.py \
# --model_path=${MODEL_NAME_OR_PATH} \
# --tokenizer_path=${MODEL_NAME_OR_PATH} \
# --chat_template=${CHAT_TEMPLATE} \
# --model_id=${MODEL_NAME} \
# --gen_temperature=0.0 \
# --use_sglang \
# --gen_parallel=16 \
# --judge_only=false \
# --judge_parallel=8 \
# --judge_model=${AHA_JUDGE_MODEL} \
# --baseline_model=${AHA_BASELINE_MODEL} \
# --to_wandb=false \
# --result_save_path=data/arena-hard-v0.1/model_performance/${MODEL_NAME} \
# --num_runs=1


CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_alpaca_eval.py \
--model_path=${MODEL_NAME_OR_PATH} \
--tokenizer_path=${MODEL_NAME_OR_PATH} \
--chat_template=${CHAT_TEMPLATE} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=8 \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=false \
--result_save_path=data/alpaca_eval_results/${MODEL_NAME} \
--num_runs=1


# rm -rf data/openllm/${MODEL_NAME}

# CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python scripts/test/run_lm_eval.py \
# --model_name_or_path=${MODEL_NAME_OR_PATH} \
# --torch_dtype=bfloat16 \
# --attn_implementation="flash_attention_2" \
# --batch_size=8 \
# --log_samples=false \
# --output_path=data/openllm/${MODEL_NAME}_bf16 \
# --to_wandb=false