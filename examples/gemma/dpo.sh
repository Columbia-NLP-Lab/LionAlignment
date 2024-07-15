export PYTHONPATH=$(pwd)

SAVE_DIR=model_checkpoints_coffee/reprod/LION-Gemma-2b-dpo-v1.0
LOGP_TRAIN_FILE=data/precompute/lion-dpo-v0.3/LION-Gemma-2b-sft-v1.0-train-full-165k.csv
LOGP_TEST_FILE=data/precompute/lion-dpo-v0.3/LION-Gemma-2b-sft-v1.0-test.csv
CONFIG_FILE=configs/gemma-2b/dpo/LION-Gemma-2b-dpo-v1.0.yaml

TRAIN_GPU_IDX=4,5,6,7
NUM_GPUS=4
MAIN_PORT=29597

CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file configs/accelerate_configs/deepspeed_zero3_4gpu.yaml \
--main_process_port=${MAIN_PORT} \
--num_processes=${NUM_GPUS} \
scripts/run_dpo_xy.py ${CONFIG_FILE} \
--model_name_or_path=Columbia-NLP/LION-Gemma-2b-sft-v1.0 \
--ref_model_name_or_path=Columbia-NLP/LION-Gemma-2b-sft-v1.0 \
--precompute_train_ref_file_path=${LOGP_TRAIN_FILE} \
--precompute_test_ref_file_path=${LOGP_TEST_FILE} \
--dataset_splits="{\"Columbia-NLP/lion-dpo-mix-v0.3\": [\"train_165k\", \"test\"]}" \
--dataset_mixer="{\"Columbia-NLP/lion-dpo-mix-v0.3\": 1.0}" \
--output_dir=${SAVE_DIR} \
--max_data_size=-1 \
--ref_update_steps=-1 \
--do_eval=true \
--evaluation_strategy=epoch \
--max_length=2048 \
--beta=0.05 \
--num_train_epochs=4 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=32 \
--wandb_group=lion-gemma-v1.0 \
--wandb_project=when2rl \
--save_strategy=no \
--save_total_limit=-1

## save_strategy no and save_total_limit -1 to save the last checkpoint regardless of eval perf.