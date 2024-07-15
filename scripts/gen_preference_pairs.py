import concurrent.futures
import openai
import llm_blender
import os
import time
import signal
import math
import subprocess
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import List
from copy import deepcopy
from tqdm.auto import tqdm
from lionalign.arguments import H4ArgumentParser
from lionalign.data.utils import create_dir_if_not_exists
from pathlib import Path
from dataclasses import dataclass, field
from accelerate import Accelerator
from accelerate.utils import gather_object


accelerator = Accelerator()


@dataclass
class DPODataGenArgs:
    """args for answer generation
    """
    model_path: str = field(
        default='', metadata={"help": "The model name or path to generate answers."}
    )
    model_id: str = field(
        default="", metadata={"help": "The model name to add metadata."}
    )
    n_to_rank: int = field(
        default=8, metadata={"help": "The number of answers to rank."}
    )
    gen_temperature: float = field(
        default=0.8, metadata={"help": "The temperature for generation."}
    )
    gen_max_tokens: int = field(
        default=2048, metadata={"help": "The max tokens for generation."}
    )
    gen_parallel: int = field(
        default=8, metadata={"help": "The number of concurrent API calls."}
    )
    chat_template: str = field(
        default=None, metadata={"help": "The chat template to use. It can be a json."}
    )
    openai_api_base: str = field(
        default=None, metadata={"help": "The openai api base url."}
    )
    logging_save_path: str = field(
        default="", metadata={"help": "The path to save the sglang log file."}
    )
    ## sglang
    sglang_ports: List[str] = field(
        default = None, metadata={"help": "The ports to run sglang servers."}
    )


    def __post_init__(self):
        if self.model_path == '':
            raise ValueError("model_path must be specified.")
        if self.model_id == '':
            self.model_id = self.model_path
        
        if self.logging_save_path == "":
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError((
                    f"{self.model_path} not found. "
                    "You are likely using a pretrained model path as save folder. "
                    "Please specify a dedicated save folder using --logging_save_path."
                ))
        else:
            # make sure the dir exists
            create_dir_if_not_exists(self.logging_save_path)
        return


@dataclass
class DPODataJudgeArgs:
    """args for judgment generation
    """
    judge_model: str = field(default="pair-rm")
    judge_batch_size: int = field(
        default=2, metadata={"help": "The number of concurrent API calls."}
    )


@dataclass
class GenericArgs:
    """args for logging + other generic args
    """
    prompt_dataset: str = field(
        default="when2rl/UltraFeedback_binarized_cleaned_annotated", metadata={"help": "The dataset to use for prompts."}
    )
    prompt_dataset_split: str = field(
        default="train_prefs", metadata={"help": "The split to use for prompts."}
    )
    max_samples: int = field(
        default=-1, metadata={"help": "The maximum number of samples to use."}
    )
    judge_only: bool = field(
        default=False, metadata={"help": "Whether to only run the judgment loop. Suitable if you already have gen files and want to try another judge."}
    )
    dset_save_name: str = field(
        default="tmp", metadata={"help": "The name of the dataset to save."}
    )


def get_model_answers(
    client: openai.Client,
    model_name: str,
    result_list: list,
    sample: dict,
    temperature: float = 0.8,
    max_tokens: int = 512,
    n: int = 8
):
    answers = []

    prompt_messages = sample['chosen'][:-1]
    for _ in range(n):
        try:
            response = client.chat.completions.create(
                model="default",
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error in getting model response: {e}")
            continue
        
        model_answer = response.choices[0].message.content
        answers.append(model_answer)

    ## update
    cloned_sample = deepcopy(sample)
    cloned_sample.pop('chosen')
    cloned_sample.pop('rejected')
    cloned_sample.pop('messages')
    cloned_sample['prompt_messages'] = prompt_messages
    cloned_sample['other_info'] = {
        'model_name': model_name,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'n': n,
    }
    cloned_sample['gen_answers'] = answers
    result_list.append(cloned_sample)
    return


def gen_api_answer(gen_args: DPODataGenArgs, dataset: Dataset, gen_save_path: str):
    client = openai.Client(
        base_url=gen_args.openai_api_base, api_key="EMPTY"
    )

    RESULT_LIST = []

    n_answers = gen_args.n_to_rank
    n_parallel = gen_args.gen_parallel
    temperature = gen_args.gen_temperature
    max_tokens = gen_args.gen_max_tokens
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = []
        for _, sample in enumerate(dataset):
            future = executor.submit(
                get_model_answers,
                client,
                gen_args.model_id,
                RESULT_LIST,
                sample,
                temperature,
                max_tokens,
                n_answers,
            )
            futures.append(future)
        
        num_completed = 0
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Getting model answers"
        ):
            future.result()

            num_completed += 1
            if num_completed % 1000 == 0:
                print(f"Completed {num_completed} samples.")
                # flush
                new_dataset_ = [Dataset.from_list(RESULT_LIST)]
                gathered_dataset = gather_object(new_dataset_)
                new_dataset = concatenate_datasets(gathered_dataset)
                if accelerator.is_main_process:
                    new_dataset.save_to_disk(gen_save_path)
                
                accelerator.wait_for_everyone()
    new_dataset = Dataset.from_list(RESULT_LIST)
    return new_dataset


def get_pids_from_sglang(sglang_ports: list):
    data = subprocess.check_output("ps -fe | grep sglang", shell=True).decode().split("\n")

    pid_dicts = {}
    for d in data:
        for port in sglang_ports:
            if f'--port {port}' in d:
                pid = d.split()[1]
                pid_dicts[port] = pid
    return pid_dicts


def _format_single_chat(conv_list: list):
    text = ''
    for turn in conv_list:
        role = turn['role']
        content = turn['content']
        text += f"{role}: {content}\n"
    return text.strip()


def format_ranking_dset(dataset):
    all_inputs = []
    all_candidates_texts = []

    for sample in dataset:
        prompt_text = _format_single_chat(sample['prompt_messages'])
        gen_answers = sample['gen_answers']

        all_inputs.append(prompt_text)
        all_candidates_texts.append(gen_answers)
    return all_inputs, all_candidates_texts


def list_generator(lst, batch_size=64):
    accumulate = []
    for item in lst:
        accumulate.append(item)
        if len(accumulate) == batch_size:
            yield accumulate
            accumulate = []
    if len(accumulate) > 0:
        yield accumulate


def gen_judgment(judge_args: DPODataJudgeArgs, dataset: Dataset, judge_save_path: str):
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM", device='cuda') # load PairRM

    all_inputs, all_candidates_texts = format_ranking_dset(dataset)

    flush_size = 512
    inputs_batched = list_generator(all_inputs, batch_size=flush_size)
    candidates_batched = list_generator(all_candidates_texts, batch_size=flush_size)
    dataset_batched = list_generator(dataset, batch_size=flush_size)

    num_iterations = math.ceil(len(all_inputs) / flush_size)
    updated_dataset = []
    for input_b, candidate_b, dataset_b in tqdm(
        zip(inputs_batched, candidates_batched, dataset_batched),
        total=num_iterations,
        desc="Ranking",
    ):
        ranks = blender.rank(
            input_b, candidate_b,
            return_scores=False,
            batch_size=judge_args.judge_batch_size,
        )

        for idx, sample in enumerate(dataset_b):
            rank = ranks[idx].tolist()
            model_answers = sample['gen_answers']

            min_rank = min(rank)  # should always be 1
            best_answer_idx = rank.index(min_rank)
            max_rank = max(rank)
            worst_answer_idx = rank.index(max_rank)  # there may be duplicates = equal ranks

            best_answer = model_answers[best_answer_idx]
            worst_answer = model_answers[worst_answer_idx]

            cloned_sample = deepcopy(sample)
            prompt_messages = cloned_sample.pop('prompt_messages')
            cloned_sample['chosen'] = prompt_messages + [{"role": "assistant", "content": best_answer}]
            cloned_sample['rejected'] = prompt_messages + [{"role": "assistant", "content": worst_answer}]
            cloned_sample['messages'] = prompt_messages + [{"role": "assistant", "content": best_answer}]
            cloned_sample['other_info']['best_rank'] = min_rank
            cloned_sample['other_info']['worst_rank'] = max_rank
            updated_dataset.append(cloned_sample)

        # flush
        new_dataset_ = [Dataset.from_list(updated_dataset)]
        new_dataset = gather_object(new_dataset_)
        new_dataset = concatenate_datasets(new_dataset)
        if accelerator.is_main_process:
            # save this temporary dataset
            print(f"[gen_judgment] Saving judge dataset to {judge_save_path}")
            new_dataset.save_to_disk(judge_save_path)
        accelerator.wait_for_everyone()
    return Dataset.from_list(updated_dataset)


def filter_dup_gens(data_dict):
    gen_answers = data_dict['gen_answers']
    unique_answers = set(gen_answers)
    if len(unique_answers) == 1:
        return False
    return True


def main(gen_args: DPODataGenArgs, judge_args: DPODataJudgeArgs, args: GenericArgs):
    print(f"Generating answers using {gen_args.model_id} on {args.prompt_dataset=}; {args.prompt_dataset_split=}.")

    dataset = load_dataset(args.prompt_dataset, split=args.prompt_dataset_split)
    if args.max_samples > 0:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(args.max_samples))

    ## generate everything first, then evaluate
    ## loop of gen + evaluate sometimes hangs for unknown reasons at the THIRD run
    gen_save_path = f"data/lion-dpo-online/{args.dset_save_name}/tmpgen"
    if args.judge_only:
        print(f'Judging only... Loading gen dataset from {gen_save_path}')
        dataset = Dataset.load_from_disk(gen_save_path)
    else:
        sglang_ports = gen_args.sglang_ports[0].split(',')
        sglang_pids = get_pids_from_sglang(sglang_ports)

        if len(sglang_ports) != accelerator.num_processes:
            raise ValueError("Number of ports and processes do not match.")

        with accelerator.split_between_processes(list(range(len(dataset)))) as sub_dataset_idx:
            sub_dataset = dataset.select(sub_dataset_idx)

            port = sglang_ports[accelerator.process_index]
            pid = sglang_pids[port]
            api_base = f"http://127.0.0.1:{port}/v1"

            gen_args_cloned = deepcopy(gen_args)
            gen_args_cloned.openai_api_base = api_base

            sub_dataset = gen_api_answer(gen_args_cloned, sub_dataset, gen_save_path)
            # kill sglang
            time.sleep(5)
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)

            sub_dataset = [sub_dataset]
        
        # concatenate
        gathered_dataset = gather_object(sub_dataset)
        dataset = concatenate_datasets(gathered_dataset)
        if accelerator.is_main_process:
            # save this temporary dataset
            print(f"Saving gen dataset to {gen_save_path}")
            dataset.save_to_disk(gen_save_path)
        accelerator.wait_for_everyone()

    ori_len = len(dataset)
    dataset = dataset.filter(
        filter_dup_gens,
        keep_in_memory=True,
        num_proc=32,
        desc="Filtering duplicate generations"
    )
    print(f"Filtered {ori_len - len(dataset)} duplicate generations. Now at {len(dataset)} samples.")
    
    ## judge everything
    judge_save_path = f"data/lion-dpo-online/{args.dset_save_name}/train"
    with accelerator.split_between_processes(list(range(len(dataset)))) as sub_dataset_idx:
        sub_dataset = dataset.select(sub_dataset_idx)

        print(f"process {accelerator.process_index} judging {len(sub_dataset)} samples.")
        sub_dataset = gen_judgment(judge_args, sub_dataset, judge_save_path)
        sub_dataset = [sub_dataset]
    
    gathered_dataset = gather_object(sub_dataset)
    dataset = concatenate_datasets(gathered_dataset)
    if accelerator.is_main_process:
        print(f"Saving dataset to {judge_save_path}")
        dataset.save_to_disk(judge_save_path)
    accelerator.wait_for_everyone()
    return


if __name__ == "__main__":
    parser = H4ArgumentParser((DPODataGenArgs, DPODataJudgeArgs, GenericArgs))
    gen_args, judge_args, args = parser.parse()

    main(gen_args, judge_args, args)