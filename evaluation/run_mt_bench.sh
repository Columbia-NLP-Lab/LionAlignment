#!/bin/bash

display_help() {
    echo "Usage: $0 [MODEL_NAME]"
    echo
    echo "   MODEL_NAME       Path to the model directory"
    echo
    echo "Example:"
    echo "   bash $0 /path/to/model"
}

# Check if help is requested
if [[ $1 == "--help" || $1 == "-h" ]]; then
    display_help
    exit 0
fi

MODEL_NAME=$1

RESULT_SAVE_PATH="mt_bench_outputs/$(basename ${MODEL_NAME})"
MODEL_ID="$(basename ${MODEL_NAME})"

export PYTHONPATH=$(pwd)

python scripts/test/run_mt_bench.py \
--model_path=${MODEL_NAME}  \
--model_id=${MODEL_ID} \
--use_sglang \
--gen_parallel=16 \
--judge_parallel=16 \
--judge_model=gpt-4 \
--to_wandb=false \
--num_runs=1 \
--result_save_path=${RESULT_SAVE_PATH} \
--y=True
