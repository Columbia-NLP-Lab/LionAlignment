#### this script requires sglang or vllm servers for inference
#### as an example, run separately:
#### CUDA_VISIBLE_DEVICES={x} python -m sglang.launch_server \
#### --model-path Columbia-NLP/LION-LLaMA-3-8b-dpo-v1.0 \
#### --enable-flashinfer --attention-reduce-in-fp32 \
#### --tokenizer-path Columbia-NLP/LION-LLaMA-3-8b-dpo-v1.0 \
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
--model_path=Columbia-NLP/LION-LLaMA-3-8b-dpo-v1.0 \
--model_id=Columbia-NLP_LION-LLaMA-3-8b-dpo-v1.0 \
--sglang_ports='61304,61305,61306,61307' \
--prompt_dataset=Columbia-NLP/DPO-UltraFeedback_binarized \
--prompt_dataset_split=train_prefs \
--max_samples=-1 \
--n_to_rank=5 \
--gen_temperature=0.8 \
--gen_parallel=16 \
--judge_only=false \
--judge_batch_size=8 \
--dset_save_name=UltraFeedback-LION-LLaMA-3-8b-dpo-v1.0-to-odpo


### train after generation

SAVE_DIR=model_checkpoints/reprod/LION-LLaMA-3-8b-odpo-v1.0
CONFIG_FILE=configs/llama-3-8b/odpo/LION-LLaMA-3-8b-odpo-v1.0.yaml

TRAIN_GPU_IDX=4,5,6,7
NUM_GPUS=4
PORT=29597


# 1. Get Log Probs
# remove --multi_gpu if error occurs
CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--multi_gpu \
--num_processes=${NUM_GPUS} \
--main_process_port=${PORT} \
scripts/precompute_dpo_logprobs.py ${CONFIG_FILE} \
--train_dataset_splits='{"data/lion-dpo-online/UltraFeedback-LION-LLaMA-3-8b-dpo-v1.0-to-odpo": "train"}' \
--train_dataset_mixer='{"data/lion-dpo-online/UltraFeedback-LION-LLaMA-3-8b-dpo-v1.0-to-odpo": 1.0}' \
--eval_dataset_splits='{"Columbia-NLP/DPO-UltraFeedback_binarized": "test_prefs"}' \
--eval_dataset_mixer='{"Columbia-NLP/DPO-UltraFeedback_binarized": 1.0}'


# 2. Train with DPO
CUDA_VISIBLE_DEVICES=${TRAIN_GPU_IDX} ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file configs/accelerate_configs/deepspeed_zero3_4gpu.yaml \
--main_process_port=${PORT} \
--num_processes=${NUM_GPUS} \
scripts/run_precompute_dpo.py ${CONFIG_FILE} \
--train_dataset_splits='{"data/lion-dpo-online/UltraFeedback-LION-LLaMA-3-8b-dpo-v1.0-to-odpo": "train"}' \
--train_dataset_mixer='{"data/lion-dpo-online/UltraFeedback-LION-LLaMA-3-8b-dpo-v1.0-to-odpo": 1.0}' \
--eval_dataset_splits='{"Columbia-NLP/DPO-UltraFeedback_binarized": "test_prefs"}' \
--eval_dataset_mixer='{"Columbia-NLP/DPO-UltraFeedback_binarized": 1.0}' \
--output_dir=${SAVE_DIR} \
--do_eval=true \
--evaluation_strategy=epoch \
--save_strategy=no \
--wandb_group=lion-llama-v1.0