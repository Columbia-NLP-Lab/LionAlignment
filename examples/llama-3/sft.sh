export PYTHONPATH=$(pwd)

SAVE_DIR=model_checkpoints/reprod/LION-LLaMA-3-8b-sft-v1.0
CONFIG_FILE=configs/llama-3-8b/sft/LION-LLaMA-3-8b-sft-v1.0.yaml

TRAIN_GPU_IDX=0,1,2,3
NUM_GPUS=4
PORT=29597

CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file configs/accelerate_configs/deepspeed_zero3_4gpu.yaml \
--main_process_port=${PORT} \
--num_processes=${NUM_GPUS} \
scripts/run_sft.py ${CONFIG_FILE} \
--output_dir=${SAVE_DIR}