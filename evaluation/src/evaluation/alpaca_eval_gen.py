import concurrent.futures
import openai
import json
import datasets
import time
import os
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from alpaca_eval import evaluate
from src.utils.utils import create_dir_if_not_exists



@dataclass
class AlpacaEvalGenArgs:
    """args for answer generation
    """
    model_path: str = field(
        metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."}
    )
    model_id: str = field(metadata={"help": "A custom name for the model."})
    tokenizer_path: str = field(
        default='',
        metadata={"help": "The path to the tokenizer. This can be a local folder or a Hugging Face repo ID."}
    )
    max_new_tokens: int = field(
        default=2048, metadata={"help": "The maximum number of new generated tokens."}
    )
    gen_temperature: float = field(
        default=0.7, metadata={"help": "The temperature for generation."}
    )
    gen_top_p: float = field(
        default=0.7, metadata={"help": "The top_p for generation."}
    )
    ### api_gen related args
    use_sglang: bool = field(
        default=True, metadata={"help": "Whether to use SGlang for geneation."}
    )
    openai_api_base: str = field(
        default=None, metadata={"help": "Used internally with SGlang. Will be automatically set by the caller program."}
    )
    gen_parallel: int = field(
        default=16, metadata={"help": "The number of concurrent API calls."}
    )
    chat_template: str = field(
        default=None, metadata={"help": "The chat template to use. It can be a json."}
    )


    def __post_init__(self):
        # if using fschat, template is specified in model_id
        if not self.use_sglang:
            raise ValueError("Only sglang is supported for now.")
        if self.tokenizer_path == '':
            self.tokenizer_path = self.model_path
        return


@dataclass
class AlpacaEvalJudgeArgs:
    """args for judgment generation
    """
    model_outputs: str = field(
        default='',
        metadata={"help": "The path to the model outputs. This will be set automatically when used with generation."}
    )
    annotators_config: str = field(
        default='weighted_alpaca_eval_gpt4_turbo',
        metadata={"help": "The path to the annotators config."}
    )
    name: str = field(
        default='',
        metadata={"help": "The name of the evaluation. Defaults to model_id when used with generation."}
    )
    output_path: str = field(
        default='auto',
        metadata={"help": "The path to save the evaluation results."}
    )
    leaderboard_mode_to_print: str = field(
        default='minimal',
        metadata={"help": "The mode to print the leaderboard."}
    )
    judge_parallel: int = field(
        default=8, metadata={"help": "The number of concurrent API calls."}
    )
    def __post_init__(self):
        if self.leaderboard_mode_to_print == 'all':
            self.leaderboard_mode_to_print = None
        return


@dataclass
class GenericArgs:
    """args for logging + other generic args
    """
    judge_only: bool = field(
        default=False, metadata={"help": "Whether to only run the judgment loop. Suitable if you already have gen files and want to try another judge."}
    )
    num_runs: int = field(
        default=1,
        metadata={"help": "Number of iterations to run this evaluation."},
    )
    result_save_path: str = field(
        default="",
        metadata={"help": "The folder to save the result. Will default to model_path/mt_bench_(judge_model).json"},
    )
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
        if self.num_runs > 1:
            raise ValueError("Only one run is supported for now.")
        return


def get_model_answers(
    client: openai.Client,
    idx: int,
    result_dict: dict,
    sample: dict,
    temperature: float = 0.7,
    top_p: float = 0.7,
    max_tokens: int = 512,
):
    num_retries = 3
    has_response = False
    for _ in range(num_retries):
        try:
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {"role": "user", "content": sample["instruction"]},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            has_response = True
        except Exception as e:
            print(f"Error in getting model response: {e}")
            time.sleep(5)
    
    if not has_response:
        result_dict[idx] = {
            'instruction': sample["instruction"],
            'output': "ERROR",
            'dataset': sample["dataset"],
            'datasplit': 'eval'
        }
        return
    
    model_answer = response.choices[0].message.content

    result_dict[idx] = {
        'instruction': sample["instruction"],
        'output': model_answer,
        'dataset': sample["dataset"],
        'datasplit': 'eval'
    }
    return


def gather_results(client, gen_temp=0.7, gen_top_p=0.7, gen_max_tokens=2048, n_parallel=8):
    eval_dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    
    result_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = []
        for idx, sample in enumerate(eval_dataset):
            future = executor.submit(
                get_model_answers,
                client,
                idx,
                result_dict,
                sample,
                gen_temp,
                gen_top_p,
                gen_max_tokens,
            )
            futures.append(future)
        
        num_completed = 0
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Getting model answers for alpaca-eval"
        ):
            future.result()
            num_completed += 1

            if num_completed % 50 == 0:
                print(f"Completed {num_completed} samples")
    return result_dict


def save_results(model_name: str, result_dict: dict, output_path: str):
    output_list = []
    for i in range(len(result_dict)):
        data = result_dict[i]
        data['generator'] = model_name
        output_list.append(data)

    with open(output_path, "w", encoding="utf-8") as fwrite:
        json.dump(output_list, fwrite, indent=2)
    return


def gen_api_answer(args: AlpacaEvalGenArgs):
    client = openai.Client(
        base_url=args.openai_api_base, api_key="EMPTY"
    )

    ### get model answers
    result_dict = gather_results(
        client,
        gen_temp=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        gen_max_tokens=args.max_new_tokens,
        n_parallel=args.gen_parallel,
    )

    ### save data
    save_dir = os.path.join("data", "alpaca_eval_results", args.model_id)
    create_dir_if_not_exists(save_dir)
    output_file = os.path.join(save_dir, "model_outputs.json")
    save_results(args.model_id, result_dict, output_file)
    return


def gen_judgment(args: AlpacaEvalJudgeArgs):
    evaluate(
        model_outputs=args.model_outputs,
        annotators_config=args.annotators_config,
        name=args.name,
        output_path=args.output_path,
        annotation_kwargs={
            'num_procs': args.judge_parallel
        },
        leaderboard_mode_to_print=args.leaderboard_mode_to_print
    )
    return