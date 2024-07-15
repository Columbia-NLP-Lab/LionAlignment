#### this script requires sglang servers for inference
#### as an example, run separately:
#### python -m sglang.launch_server \
#### --model-path Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
#### --enable-flashinfer --attention-reduce-in-fp32 \
#### --chat-template configs/chat_templates/lion-gemma-2b.json \
#### --port {61304, 61305, 61306, 61307}
#### note that these sglang processes WILL be terminated by the scripts/gen_preference_pairs.py at the end
export PYTHONPATH=$(pwd)

### gen data
EVAL_GPU_IDX=0,1,2,3
NUM_GPUS=4
PORT=29537


CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--num_processes=${NUM_GPUS} \
--main_process_port=${PORT} \
scripts/gen_preference_pairs.py \
--model_path=Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
--model_id=Columbia-NLP_LION-Gemma-2b-dpo-v1.0 \
--sglang_ports='61304,61305,61306,61307' \
--prompt_dataset=Columbia-NLP/DPO-UltraFeedback_binarized \
--prompt_dataset_split=train_prefs \
--max_samples=-1 \
--n_to_rank=5 \
--gen_temperature=0.8 \
--gen_parallel=16 \
--judge_only=false \
--judge_batch_size=8 \
--dset_save_name=UltraFeedback-LION-Gemma-2b-dpo-v1.0


### train after generation

SAVE_DIR=model_checkpoints_coffee/reprod/LION-Gemma-2b-odpo-v0.9
LOGP_TRAIN_FILE=data/precompute/lion-dpo-online/LION-Gemma-2b-dpo-v1.0-UltraFeedback-odpo-train-full.csv
LOGP_TEST_FILE=data/precompute/lion-dpo-online/LION-Gemma-2b-dpo-v1.0-UltraFeedback-test.csv
CONFIG_FILE=configs/gemma-2b/dpo/LION-Gemma-2b-dpo-v1.0.yaml

TRAIN_GPU_IDX=0,1,2,3
NUM_GPUS=4
MAIN_PORT=29597

CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file configs/accelerate_configs/deepspeed_zero3_4gpu.yaml \
--main_process_port=${MAIN_PORT} \
--num_processes=${NUM_GPUS} \
scripts/run_dpo_xy.py ${CONFIG_FILE} \
--model_name_or_path=Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
--ref_model_name_or_path=Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
--precompute_train_ref_file_path=${LOGP_TRAIN_FILE} \
--precompute_test_ref_file_path=${LOGP_TEST_FILE} \
--dataset_splits='{"Columbia-NLP/lion-dpo-mix-v0.3": ["test"], "data/lion-dpo-online/UltraFeedback-LION-Gemma-2b-dpo-v1.0": ["train"]}' \
--dataset_mixer='{"Columbia-NLP/lion-dpo-mix-v0.3": 1.0, "data/lion-dpo-online/UltraFeedback-LION-Gemma-2b-dpo-v1.0": 1.0}' \
--output_dir=${SAVE_DIR} \
--max_data_size=-1 \
--ref_update_steps=-1 \
--do_eval=true \
--evaluation_strategy=epoch \
--max_length=2048 \
--beta=0.05 \
--num_train_epochs=2 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=32 \
--wandb_group=lion-gemma-v1.0 \
--wandb_project=when2rl \
--save_strategy=no \
--save_total_limit=-1