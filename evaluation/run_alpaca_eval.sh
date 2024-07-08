#!/bin/bash

display_help() {
    echo "Usage: $0 [MODEL_NAME] [GEN_TEMPERATURE] [CHAT_TEMPLATE]"
    echo
    echo "   MODEL_NAME       Path to the model directory"
    echo "   GEN_TEMPERATURE  Generation temperature value"
    echo "   CHAT_TEMPLATE    Path to the chat template JSON file"
    echo
    echo "Example:"
    echo "   bash $0 /path/to/model 1.0 /path/to/chat_template.json"
}

# Check if help is requested
if [[ $1 == "--help" || $1 == "-h" ]]; then
    display_help
    exit 0
fi

# Check if the required arguments are provided
MODEL_NAME=$1
GEN_TEMPERATURE=$2
CHAT_TEMPLATE=${3:-None}  # Set CHAT_TEMPLATE to None if not provided


RESULT_SAVE_PATH="alpaca2_outputs/$(basename ${MODEL_NAME})_temp${GEN_TEMPERATURE}"
MODEL_ID="$(basename ${MODEL_NAME})_temp${GEN_TEMPERATURE}"

export PYTHONPATH=$(pwd)
python scripts/test/run_alpaca_eval.py \
--model_path=${MODEL_NAME} \
--tokenizer_path=${MODEL_NAME} \
--model_id=${MODEL_ID} \
--gen_temperature=${GEN_TEMPERATURE} \
--use_sglang \
--gen_parallel=16 \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=false \
--num_runs=1 \
--result_save_path=${RESULT_SAVE_PATH}