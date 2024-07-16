# this assumes lm_eval==0.4.2
import lm_eval
import json
import numpy as np
import time
import torch
import wandb
import yaml
import os
import ray
from lionalign.arguments import H4ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError
from dataclasses import dataclass, field
from pathlib import Path
from lm_eval import evaluator
from lm_eval.tasks import TaskManager


os.environ["TOKENIZERS_PARALLELISM"] = "true"


TASKS_WE_USE = [
    {'name': 'hellaswag', 'num_shots': 10, 'is_gen': False, 'in_openllm': True, 'use_in_fast_mode': True, 'metric': 'acc_norm'},
    {'name': 'arc_challenge', 'num_shots': 25, 'is_gen': False, 'in_openllm': True, 'use_in_fast_mode': True, 'metric': 'acc_norm'},
    {'name': 'truthfulqa_mc2', 'num_shots': 0, 'is_gen': False, 'in_openllm': True, 'use_in_fast_mode': False, 'metric': 'acc'},
    {'name': 'winogrande', 'num_shots': 5, 'is_gen': False, 'in_openllm': True, 'use_in_fast_mode': False, 'metric': 'acc'},
    {'name': 'gsm8k', 'num_shots': 5, 'is_gen': True, 'in_openllm': True, 'use_in_fast_mode': True, 'metric': 'exact_match,strict-match'},
    {'name': 'mmlu', 'num_shots': 5, 'is_gen': False, 'in_openllm': True, 'use_in_fast_mode': False, 'metric': 'acc'},
    {'name': 'bbh_cot_fewshot_date_understanding', 'num_shots': None, 'is_gen': True, 'in_openllm': False, 'use_in_fast_mode': True, 'metric': 'exact_match,get-answer'},
    {'name': 'bbh_cot_fewshot_movie_recommendation', 'num_shots': None, 'is_gen': True, 'in_openllm': False, 'use_in_fast_mode': True, 'metric': 'exact_match,get-answer'},
    {'name': 'bbh_cot_fewshot_reasoning_about_colored_objects', 'num_shots': None, 'is_gen': True, 'in_openllm': False, 'use_in_fast_mode': False, 'metric': 'exact_match,get-answer'}
]

TASK_TO_METRIC = {v['name']: v['metric'] for v in TASKS_WE_USE}
TASK_TO_NUM_SHOT = {v['name']: v['num_shots'] for v in TASKS_WE_USE}
ALL_TASKS = [v['name'] for v in TASKS_WE_USE]
GEN_TASKS = set([v['name'] for v in TASKS_WE_USE if v['is_gen']])
OPENLLM_TASKS = set([v['name'] for v in TASKS_WE_USE if v['in_openllm']])
FAST_EVAL_TASKS = set([v['name'] for v in TASKS_WE_USE if v['use_in_fast_mode']])


@dataclass
class LMEvalArguments:
    ## model args
    model: str = field(
        default="hf",
        metadata={"help": "The model TYPE"},
    )
    model_name_or_path: str = field(
        default="HuggingFaceH4/zephyr-7b-beta",
        metadata={"help": "The model name or path."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The model revision."},
    )
    tokenizer_name_or_path: str = field(
        default="",
        metadata={"help": "In some rare occasion, you may want to manually specify the tokenizer name or path. If empty, set to model_name_or_path."},
    )
    tokenizer_revision: str = field(
        default="main",
        metadata={"help": "The tokenizer revision."},
    )
    attn_implementation: str = field(
        default=None,
        metadata={"help": "The attention implementation."},
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "The torch dtype."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code."},
    )
    ## eval args
    batch_size: int = field(
        default=16,
        metadata={"help": "The batch size."},
    )
    fast_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use fast mode = eval less tasks."},
    )
    openllm_only: bool = field(
        default=False,
        metadata={"help": "Whether to eval only openllm tasks."},
    )
    ## save args
    output_path: str = field(
        default="data/openllm/zephyr-7b-beta",
        metadata={"help": "The output path for the results."},
    )
    log_samples: bool = field(
        default=True,
        metadata={"help": "Whether to log samples."},
    )
    verbosity: str = field(
        default="DEBUG",
        metadata={"help": "The verbosity level."},
    )
    ## logging args
    to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to log to wandb."},
    )
    wandb_project: str = field(
        default="",
        metadata={"help": "The wandb project."},
    )
    wandb_id: str = field(
        default="",
        metadata={"help": "The wandb run id to upload results to. If empty, we will check the model_name_or_path/run_args.yaml"},
    )

    def __post_init__(self):
        if self.output_path:
            path = Path(self.output_path)
            # check if file or 'dir/results.json' exists
            if path.is_file() or Path(self.output_path).joinpath("results.json").is_file():
                print(
                    f"File already exists at {path}. Results will be overwritten."
                )
                assert not path.is_file(), "File already exists"
            # if path json then get parent dir
            elif path.suffix in (".json", ".jsonl"):
                raise NotImplementedError("Not implemented")
            else:
                path.mkdir(parents=True, exist_ok=True)
        elif self.log_samples and not self.output_path:
            assert self.output_path, "Specify --output_path"

        if self.tokenizer_name_or_path == "":
            self.tokenizer_name_or_path = self.model_name_or_path

        if self.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif self.torch_dtype == "auto":
            self.torch_dtype = "auto"
        else:
            raise NotImplementedError("Torch dtype not implemented")

        if self.to_wandb and self.wandb_id == "":
            path = Path(self.model_name_or_path)
            if 'checkpoint' in path.name:
                # check parent dir for run_args.yaml
                path = path.parent
                print(f"Checking parent dir for run_args.yaml: {path}")
            
            assert path.joinpath("run_args.yaml").is_file(), f"File not found at {path.joinpath('run_args.yaml')}"
            with open(path.joinpath("run_args.yaml"), "r", encoding="utf-8") as fread:
                all_args = yaml.load(fread, Loader=yaml.Loader)
            self.wandb_id = all_args['wandb_id']
            self.wandb_project = all_args['wandb_project']
            print(f"Read wandb info from {path.joinpath('run_args.yaml')}: {self.wandb_project}/{self.wandb_id}")
        return


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def save_results(args: LMEvalArguments, new_results: dict, new_tasks: list, prev_results=None):
    # since we are looping over lm_eval.simple_evaluate, we need to update the results manually for this to work
    # new_tasks is the tasks that is ONLY in new_results
    # prev_results is the results from the previous run of simple_evaluate. The goal is to keep the `results` in output_path_file updated
    path = Path(args.output_path)
    output_path_file = path.joinpath("results.json")

    if prev_results is not None:
        # combine the dicts
        for k, old_results in prev_results.items():
            if k not in new_results:
                new_results[k] = old_results
            elif isinstance(old_results, dict):
                new_results[k] = {**old_results, **new_results[k]}
            elif isinstance(old_results, (str, float, int)):
                new_results[k] = old_results
            else:
                print("skipping", k, old_results)
    
    samples = new_results.pop("samples", None)
    
    dumped = json.dumps(
        new_results, indent=2, default=_handle_non_serializable, ensure_ascii=False
    )

    output_path_file.open("w", encoding="utf-8").write(dumped)
    if args.log_samples:
        for task_name, config in new_results["configs"].items():
            if task_name not in new_tasks:
                print(f"Task {task_name} not in new_tasks: {new_tasks}")
                continue
            filename = path.joinpath(f"{task_name}.jsonl")
            samples_dumped = json.dumps(
                samples[task_name],
                indent=2,
                default=_handle_non_serializable,
                ensure_ascii=False,
            )
            filename.write_text(samples_dumped, encoding="utf-8")
    return new_results


def get_performance(args: LMEvalArguments, all_results, all_tasks):
    metrics = {}
    all_averages = []
    openllm_averages = []
    classification_average = []
    generation_average = []
    for task, task_result in all_results['results'].items():
        if task in all_tasks:
            # clean the "acc,none" to "acc"
            task_result_cleaned = {}
            for k, v in task_result.items():
                if k == 'alias':
                    continue
                k = k.replace(",none", "")
                task_result_cleaned[k] = v

                # get the average
                if k != TASK_TO_METRIC[task]:
                    continue
                ### now v is the metric of interest
                all_averages.append(v)
                # openllm
                if task in OPENLLM_TASKS:
                    openllm_averages.append(v)
                # gen or classification
                if task in GEN_TASKS:
                    generation_average.append(v)
                else:
                    classification_average.append(v)
            metrics[task] = task_result_cleaned
    metrics['openllm_average'] = np.mean(openllm_averages).item()
    metrics['classification_average'] = np.mean(classification_average).item()
    metrics['generation_average'] = np.mean(generation_average).item()
    metrics['all_average'] = np.mean(all_averages).item()
    
    ### save this thing
    path = Path(args.output_path)
    output_path_file = path.joinpath("performance.json")
    dumped = json.dumps(
        metrics, indent=2, default=_handle_non_serializable, ensure_ascii=False
    )
    output_path_file.open("w", encoding="utf-8").write(dumped)
    return metrics


def _has_model_weights(dir_path: str):
    for files in Path(dir_path).iterdir():
        if files.is_file() and files.suffix == ".safetensors":
            return True
    return False


def find_checkpoint_paths(base_path: str):
    found_dirs = []
    # first check if it is a HF model
    hf_api = HfApi()
    try:
        _ = hf_api.model_info(base_path)
        found_dirs.append((base_path, ""))
    except (RepositoryNotFoundError, HFValidationError):
        # it is a local model
        if _has_model_weights(base_path):
            found_dirs.append((base_path, ""))

        for folder in Path(base_path).iterdir():
            if folder.is_dir():
                if _has_model_weights(folder):
                    found_dirs.append((folder, f"_{folder.name}"))
    return found_dirs


@ray.remote(num_gpus=1)
def evaluate_single_model(args: LMEvalArguments, wandb_suffix: str = ""):
    # lm_eval.tasks.initialize_tasks(verbosity=args.verbosity)
    task_manager = TaskManager(args.verbosity, include_path=None)

    ## manual init
    if args.attn_implementation is not None:
        loaded_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=args.torch_dtype,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code
        )
    else:
        loaded_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code
        )
    loaded_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    if 'stablelm' in loaded_tokenizer.name_or_path:
        print("Setting pad token id to 100288 assuming you are using StableLM tokenizer")
        loaded_tokenizer.pad_token_id = 100288

    ## move to cuda
    loaded_model = loaded_model.to('cuda')
    loaded_model = loaded_model.eval()

    lm = lm_eval.api.registry.get_model(args.model).create_from_arg_string(
        '',
        {
            "pretrained": loaded_model,
            "tokenizer": loaded_tokenizer,
            "trust_remote_code": True,
            "batch_size": args.batch_size,
            "max_batch_size": None,
            "device": None,
        },
    )

    start_time = time.time()

    #### run lm_eval for all tasks
    prev_results = None
    for task in ALL_TASKS:
        print(f"Running task {task} with {TASK_TO_NUM_SHOT[task]} shots")
        if args.fast_eval and task not in FAST_EVAL_TASKS:
            print(f"Skipping task {task} as we are doing fast evaluation")
            continue
        if args.openllm_only and task not in OPENLLM_TASKS:
            print(f"Skipping task {task} as we are doing openllm only evaluation")
            continue
        new_results = evaluator.simple_evaluate( # call simple_evaluate
            model=lm,
            tasks=[task],
            batch_size=args.batch_size,
            num_fewshot=TASK_TO_NUM_SHOT[task],
            log_samples=True,
            gen_kwargs=None,
            task_manager=task_manager,
            # limit=16
        )
        prev_results = save_results(
            args,
            new_results=new_results,
            new_tasks=[task],
            prev_results=prev_results
        )
    time_taken = time.time() - start_time

    # print the results in prev_results
    all_tasks = ALL_TASKS
    performance = get_performance(args, prev_results, all_tasks)
    print(json.dumps(performance, indent=2, ensure_ascii=False))
    print(f"Time taken: {time_taken} seconds")
    
    ## upload results
    if args.wandb_id != '':
        wandb.init(
            project=args.wandb_project,
            id=args.wandb_id,
            resume=True
        )
        wandb_perf = {f"lm_eval{wandb_suffix}/{k}": v for k, v in performance.items()}
        wandb_perf[f"lm_eval{wandb_suffix}/time_taken"] = time_taken
        wandb.log(wandb_perf)
        wandb.finish()
    return


def main(args: LMEvalArguments):
    print(f"Eval parameters {args}")

    base_model_path = args.model_name_or_path
    # scans the base directory to find all model weights.
    # this can either be the base_dir itself, or be base_dir/checkpoint-xxx
    for model_path_found, suffix in find_checkpoint_paths(base_model_path):
        print(f"Model weights found at {model_path_found}. Using {suffix=}.")
        args.model_name_or_path = model_path_found
        args.tokenizer_name_or_path = model_path_found
        ray.get(evaluate_single_model.remote(args, suffix))
    return


if __name__ == '__main__':
    parser = H4ArgumentParser(LMEvalArguments)
    args = parser.parse()
    
    main(args)