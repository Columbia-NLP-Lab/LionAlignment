#!/bin/bash

display_help() {
    echo "Usage: $0 [MODEL_NAME] [CHAT_TEMPLATE]"
    echo
    echo "   MODEL_NAME       Path to the model directory"
    echo "   CHAT_TEMPLATE    Path to the chat template JSON file"
    echo
    echo "Example:"
    echo "   bash $0 /path/to/model /path/to/chat_template.json"
}

# Check if help is requested
if [[ $1 == "--help" || $1 == "-h" ]]; then
    display_help
    exit 0
fi


MODEL_NAME=$1
CHAT_TEMPLATE=${2:-None}  # Set CHAT_TEMPLATE to None if not provided

RESULT_SAVE_PATH="arena_hard_outputs/$(basename ${MODEL_NAME})"
MODEL_ID="$(basename ${MODEL_NAME})"

export PYTHONPATH=$(pwd)

python scripts/test/run_arena_hard_auto.py \
--model_path=${MODEL_NAME}  \
--tokenizer_path=${MODEL_NAME}  \
--model_id=${MODEL_ID} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=16 \
--chat_template=${CHAT_TEMPLATE} \
--judge_only=false \
--judge_parallel=16 \
--judge_model=gpt-4-1106-preview \
--baseline_model=gpt-4-0314 \
--to_wandb=false \
--num_runs=1 \
--result_save_path=${RESULT_SAVE_PATH}


export PYTHONPATH=$(pwd)

MODEL_NAME=princeton-nlp/Llama-3-Base-8B-SFT-SimPO
TOKENIZER_NAME=princeton-nlp/Llama-3-Base-8B-SFT-DPO
RESULT_SAVE_PATH="arena_hard_outputs/$(basename ${MODEL_NAME})"
MODEL_ID="$(basename ${MODEL_NAME})"
python scripts/test/run_arena_hard_auto.py \
--model_path=${MODEL_NAME}  \
--tokenizer_path=${TOKENIZER_NAME}  \
--model_id=${MODEL_ID} \
--gen_temperature=0.0 \
--use_sglang \
--gen_parallel=16 \
--judge_only=false \
--judge_parallel=16 \
--judge_model=gpt-4-1106-preview \
--baseline_model=gpt-4-0314 \
--to_wandb=false \
--num_runs=1 \
--result_save_path=${RESULT_SAVE_PATH}
