#### this script requires sglang or vllm servers for inference
#### as an example, run separately:
#### CUDA_VISIBLE_DEVICES={x} python -m sglang.launch_server \
#### --model-path Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
#### --enable-flashinfer --attention-reduce-in-fp32 \
#### --chat-template configs/chat_templates/lion-gemma-2b.json \
#### --port {61304, 61305, 61306, 61307}
#### note that these sglang processes WILL be terminated by the scripts/gen_preference_pairs.py at the end
export PYTHONPATH=$(pwd)

### gen data
EVAL_GPU_IDX=4,5,6,7
NUM_GPUS=4
PORT=29537


CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--multi_gpu \
--num_processes=${NUM_GPUS} \
--main_process_port=${PORT} \
scripts/gen_preference_pairs.py \
--model_path=Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
--model_id=Columbia-NLP_LION-Gemma-2b-dpo-v1.0 \
--sglang_ports='61304,61305,61306,61307' \
--prompt_dataset=Columbia-NLP/DPO-UltraFeedback_binarized \
--prompt_dataset_split=train_prefs \
--max_samples=-1 \
--p_ori_pair=0.15 \
--p_ori_chosen=0.35 \
--n_to_rank=5 \
--gen_temperature=0.8 \
--gen_parallel=16 \
--judge_only=false \
--judge_batch_size=8 \
--dset_save_name=UltraFeedback-LION-Gemma-2b-dpo-v1.0-to-odpo


### train after generation

SAVE_DIR=model_checkpoints_coffee/reprod/LION-Gemma-2b-odpo-v1.0
LOGP_TRAIN_FILE=data/precompute/lion-dpo-online/LION-Gemma-2b-dpo-v1.0-UltraFeedback-odpo-train-mixed-full.csv
LOGP_TEST_FILE=data/precompute/lion-dpo-online/LION-Gemma-2b-dpo-v1.0-UltraFeedback-test.csv
CONFIG_FILE=configs/gemma-2b/dpo/LION-Gemma-2b-dpo-v1.0.yaml

TRAIN_GPU_IDX=4,5,6,7
NUM_GPUS=4
MAIN_PORT=29597

CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--multi_gpu \
--config_file configs/accelerate_configs/deepspeed_zero3_4gpu.yaml \
--main_process_port=${MAIN_PORT} \
--num_processes=${NUM_GPUS} \
scripts/run_dpo_xy.py ${CONFIG_FILE} \
--model_name_or_path=Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
--ref_model_name_or_path=Columbia-NLP/LION-Gemma-2b-dpo-v1.0 \
--precompute_train_ref_file_path=${LOGP_TRAIN_FILE} \
--precompute_test_ref_file_path=${LOGP_TEST_FILE} \
--dataset_splits='{"Columbia-NLP/lion-dpo-mix-v0.3": ["train_165k", "test"], "data/lion-dpo-online/UltraFeedback-LION-Gemma-2b-dpo-v1.0-to-odpo": ["train_mix"]}' \
--dataset_mixer='{"Columbia-NLP/lion-dpo-mix-v0.3": 0.3, "data/lion-dpo-online/UltraFeedback-LION-Gemma-2b-dpo-v1.0-to-odpo": 0.1}' \
--output_dir=${SAVE_DIR} \
--max_data_size=-1 \
--ref_update_steps=-1 \
--do_eval=true \
--evaluation_strategy=epoch \
--max_length=2048 \
--beta=0.05 \
--num_train_epochs=1 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--gradient_accumulation_steps=8 \
--wandb_group=lion-gemma-v1.0 \
--wandb_project=when2rl \
--save_strategy=no \
--save_total_limit=-1


### eval
source ../when2rl/.env

MODEL_NAME_OR_PATH=${SAVE_DIR}
MODEL_NAME=LION-Gemma-2b-odpo-v1.0
MT_JUDGE_MODEL=gpt-4
AHA_JUDGE_MODEL=gpt-4-1106-preview
AHA_BASELINE_MODEL=gpt-4-0314
EVAL_GPU_IDX=7
CHAT_TEMPLATE=configs/chat_templates/lion-gemma-2b.json

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDX} python test/run_alpaca_eval.py \
--model_path=${MODEL_NAME_OR_PATH} \
--tokenizer_path=${MODEL_NAME_OR_PATH} \
--chat_template=${CHAT_TEMPLATE} \
--model_id=${MODEL_NAME} \
--gen_temperature=0.7 \
--use_sglang \
--gen_parallel=8 \
--judge_only=false \
--judge_parallel=8 \
--to_wandb=true \
--result_save_path=data/alpaca_eval_results/${MODEL_NAME} \
--num_runs=1