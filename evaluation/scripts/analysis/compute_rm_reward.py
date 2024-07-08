from tqdm.auto import tqdm
from src.models.ultra_rm import UltraRMModel
from src.models.starling_rm import StarlingRMModel, Starling34BRMModel
from src.utils.utils import create_dir_if_not_exists
from src.utils.data_utils import add_full_id
from transformers import LlamaTokenizer, AutoTokenizer
from datasets import load_dataset
import math
import torch
import argparse
import jsonlines
import os
import numpy as np
import json


def format_chat(chat_history: list, model_name):
    if model_name == "openbmb/UltraRM-13b":
        ### Human: {instruction}\n Assistant: {completion}
        assert chat_history[0]['role'] == 'user'
        out = ''
        for turn in chat_history:
            role = turn['role']
            content = turn['content']
            if role == 'assistant':
                out += f"Assistant: {content}\n"
            else:
                out += f"Human: {content}\n"
        out = out.strip()
        return out
    elif model_name == "berkeley-nest/Starling-RM-7B-alpha":
        ### <s>[INST] Hello? </s> [/INST] Hi, how can I help you?</s>
        assert chat_history[0]['role'] == 'user'
        out = ''
        for turn in chat_history:
            role = turn['role']
            content = turn['content']
            if role == 'assistant':
                out += f" {content}</s>"
            else:
                out += f"<s>[INST] {content} </s> [/INST]"
        out = out.strip()
        return out
    elif model_name == "Nexusflow/Starling-RM-34B":
        ### <|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\nHi, how can I help you?<|im_end|>
        assert chat_history[0]['role'] == 'user'
        out = ''
        for turn in chat_history:
            role = turn['role']
            content = turn['content']
            if role == 'assistant':
                out += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                out += f"<|im_start|>user\n{content}<|im_end|>\n"
        out = out.strip()
        return out
    else:
        raise ValueError("apply_template: Model not supported")
    return


def prepare_pref_dataset(data_dict, model_name: str):
    # chosen_text = format_two_turn(data_dict['chosen'])
    chosen_text = format_chat(data_dict['chosen'], model_name)
    rejected_text = format_chat(data_dict['rejected'], model_name)

    data_dict['chosen_text'] = chosen_text
    data_dict['rejected_text'] = rejected_text
    return data_dict


def dict_data_iterator(data, batch_size=4):
    batch_ = []
    for sample in data:
        batch_.append(sample)
        if len(batch_) == batch_size:
            # a single dict with list
            keys = batch_[0].keys()
            batch = {k: [d[k] for d in batch_] for k in keys}
            yield batch
            batch_ = []
    if len(batch_) > 0:
        keys = batch_[0].keys()
        batch = {k: [d[k] for d in batch_] for k in keys}
        yield batch


def compute_accuracy(predictions: list):
    num_total = 0
    num_correct = 0
    for pred in predictions:
        chosen_reward_np = np.array(pred['chosen_reward'])
        rejected_reward_np = np.array(pred['rejected_reward'])
        num_correct += np.sum(chosen_reward_np > rejected_reward_np)
        num_total += len(chosen_reward_np)
    return num_correct / num_total


def prepare_model_n_tokenizer(args):
    model_name_or_path = args.model_name_or_path
    if model_name_or_path == "berkeley-nest/Starling-RM-7B-alpha":
        model = StarlingRMModel(
            model_name_or_path,
            # torch_dtype=torch.bfloat16,  # bf16 or fp16 will SIGNIFICANTLY screw up the results
            # use_flash_attention_2=True
        )
        model = model.cuda()
        tokenizer = model.tokenizer
        tokenizer.truncation_side = "left"
    elif model_name_or_path == "Nexusflow/Starling-RM-34B":
        model = Starling34BRMModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16
        )
        model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat")
        tokenizer.truncation_side = "left"
    elif model_name_or_path == "openbmb/UltraRM-13b":
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            truncation=True,
            padding=True,
            padding_side="right",
            truncation_side="left",
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = UltraRMModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True
        )
        model = model.cuda()
        model = model.eval()
    else:
        raise ValueError("Model not supported")
    return model, tokenizer


def prepare_dataset(args):
    raw_dataset = load_dataset(
        # "when2rl/UltraFeedback_binarized_cleaned_annotated",
        # "Columbia-NLP/lion-dpo-mix-v0.1",
        args.eval_dset_name,
        split=args.eval_dset_split,
    )
    raw_dataset = raw_dataset.map(
        add_full_id,
        num_proc=64,
        keep_in_memory=True,
        desc=f"Adding full_id to {args.eval_dset_name}/{args.eval_dset_split}",
    )
    dataset_for_reward = raw_dataset.map(
        prepare_pref_dataset,
        fn_kwargs={"model_name": args.model_name_or_path},
        num_proc=8,
        desc="Preparing data for reward model",
    )
    dataset_for_reward = dataset_for_reward.select_columns(
        ["chosen_text", "rejected_text", "score_chosen", "score_rejected", "full_id"]
    )
    dataset_for_reward = dataset_for_reward.shuffle(seed=42)
    return dataset_for_reward


def save_results(predictions):
    # first flatten the predictions
    flat_predictions = []
    for pred in predictions:
        for i in range(len(pred['full_id'])):
            curr_pred = {}
            curr_pred['full_id'] = pred['full_id'][i]
            curr_pred['chosen_reward'] = pred['chosen_reward'][i]
            curr_pred['rejected_reward'] = pred['rejected_reward'][i]
            flat_predictions.append(curr_pred)
    
    save_path = os.path.join(args.output_dir, "reward_predictions.jsonl")
    with jsonlines.open(save_path, "w") as writer:
        writer.write_all(flat_predictions)
    return


def main(args):
    create_dir_if_not_exists(args.output_dir)
    ## save config
    save_path = os.path.join(args.output_dir, "config.json")
    with open(save_path, "w") as writer:
        json.dump(vars(args), writer, indent=4)

    ### prepare data
    preference_dataset = prepare_dataset(args)

    ### prepare model
    model, tokenizer = prepare_model_n_tokenizer(args)

    NUM_TO_SCORE = len(preference_dataset)
    BATCH_SIZE = 2
    num_batchs = math.ceil(NUM_TO_SCORE / BATCH_SIZE)
    pbar = tqdm(total=num_batchs, desc="Scoring Preference Dataset")
    predictions = []
    steps = 0
    for batch in dict_data_iterator(preference_dataset.select(range(NUM_TO_SCORE)), batch_size=BATCH_SIZE):
        chosen_texts = batch['chosen_text']
        rejected_texts = batch['rejected_text']

        # predict chosen
        with torch.no_grad():
            inputs = tokenizer(chosen_texts, return_tensors="pt", padding='longest', truncation=True, max_length=2048)
            inputs = inputs.to("cuda")
            chosen_reward = model(**inputs)

            # predict rejected
            inputs = tokenizer(rejected_texts, return_tensors="pt", padding='longest', truncation=True, max_length=2048)
            inputs = inputs.to("cuda")
            rejected_reward = model(**inputs)

        curr_predictions = {}
        curr_predictions['full_id'] = batch['full_id']
        curr_predictions['chosen_reward'] = chosen_reward.cpu().float().numpy().tolist()
        curr_predictions['rejected_reward'] = rejected_reward.cpu().float().numpy().tolist()
        predictions.append(curr_predictions)
        pbar.update(1)
        steps += 1

        if steps % 500 == 0:
            save_results(predictions)
    pbar.close()
    save_results(predictions)

    ### compute acc
    acc = compute_accuracy(predictions)
    performance = {
        "accuracy": acc,
        "num_samples": NUM_TO_SCORE
    }
    print("Performance")
    print(json.dumps(performance, indent=4))
    return


if __name__ == "__main__":
    # example:
    # python scripts/analysis/compute_rm_reward.py \
    # --output_dir data/lion-dpo-mix-v0.1/starling34B-rm-label \
    # --model_name_or_path Nexusflow/Starling-RM-34B \
    # --eval_dset_name Columbia-NLP/lion-dpo-mix-v0.1 \
    # --eval_dset_split train_10k
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save the predictions and performance",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Model name or path",
    )
    parser.add_argument(
        "--eval_dset_name",
        type=str,
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--eval_dset_split",
        type=str,
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()

    main(args)