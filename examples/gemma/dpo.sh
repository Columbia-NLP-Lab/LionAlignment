export PYTHONPATH=$(pwd)

SAVE_DIR=model_checkpoints/reprod/LION-Gemma-2b-dpo-v1.0
CONFIG_FILE=configs/gemma-2b/dpo/LION-Gemma-2b-dpo-v1.0.yaml

TRAIN_GPU_IDX=0,1,2,3
NUM_GPUS=4
PORT=29556

# 1. Get Log Probs
# remove --multi_gpu if error occurs
CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--multi_gpu \
--num_processes=${NUM_GPUS} \
--main_process_port=${PORT} \
scripts/precompute_dpo_logprobs.py ${CONFIG_FILE}


# 2. Train with DPO
CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file configs/accelerate_configs/deepspeed_zero3_4gpu.yaml \
--num_processes=${NUM_GPUS} \
--main_process_port=${PORT} \
scripts/run_precompute_dpo ${CONFIG_FILE} \
--output_dir=${SAVE_DIR}
