from src.evaluation.arena_hard_common import (
    load_questions,
    load_model_answers,
    make_config,
    chat_completion_openai,
    chat_completion_anthropic,
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
)
from dataclasses import dataclass, field
from typing import List
import concurrent
import json
import tqdm
import openai
import os
import time
import shortuuid
import tiktoken
import re


@dataclass
class ArenaHardGenArgs:
    """args for answer generation
    """
    model_path: str = field(
        metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."}
    )
    tokenizer_path : str = field(
        metadata={"help": "The path to the tokenizer. This can be a local folder or a Hugging Face repo ID."}
    )
    model_id: str = field(metadata={"help": "A custom name for the model."})
    bench_name: str = field(
        default="arena-hard-v0.1", metadata={"help": "The name of the benchmark question set."}
    )
    max_new_token: int = field(
        default=4096, metadata={"help": "The maximum number of new generated tokens."}
    )
    num_choices: int = field(
        default=1, metadata={"help": "How many completion choices to generate."}
    )
    gen_temperature: float = field(
        default=0.0, metadata={"help": "The temperature for generation."}
    )
    ### api_gen related args
    use_sglang: bool = field(
        default=True, metadata={"help": "Whether to use SGlang for geneation."}
    )
    api_type: str = field(
        default="openai", metadata={"help": "The type of API to use for generation."}
    )
    openai_api_base: str = field(
        default=None, metadata={"help": "Used internally with SGlang. Will be automatically set by the caller program."}
    )
    gen_parallel: int = field(
        default=8, metadata={"help": "The number of concurrent API calls."}
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
class ArenaHardJudgeArgs:
    """args for judgment generation
    """
    bench_name__: str = field(
        default="arena-hard-v0.1", metadata={"help": "The name of the benchmark question set."}
    )
    judge_model: str = field(default="gpt-4-0125-preview")
    baseline_model: str = field(default="gpt-4-0314")
    model_list: List[str] = field(
        default_factory=lambda: [], metadata={"help": "A list of models to be evaluated"}
    )
    judge_parallel: int = field(
        default=8, metadata={"help": "The number of concurrent API calls."}
    )


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


def get_gen_answer(
    question: dict, model: str, endpoint_info: dict, num_choices: int, max_tokens: int, temperature: float, answer_file: str
):
    api_dict = endpoint_info["endpoint"]
    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]

    api_type = endpoint_info["api_type"]

    conv = []

    if "system_prompt" in endpoint_info.keys():
        conv.append({"role": "system", "content": endpoint_info["system_prompt"]})
    elif model in OPENAI_MODEL_LIST:
        conv.append({"role": "system", "content": "You are a helpful assistant."})

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    choices = []
    for i in range(num_choices):
        turns = []
        for j in range(len(question["turns"])):
            conv.append({"role": "user", "content": question["turns"][j]["content"]})
            if api_type == "anthropic":
                output = chat_completion_anthropic(model=endpoint_info["model_name"],
                                                   messages=conv,
                                                   temperature=temperature,
                                                   max_tokens=max_tokens)
            else:
                output = chat_completion_openai(model=endpoint_info["model_name"], 
                                                messages=conv, 
                                                temperature=temperature, 
                                                max_tokens=max_tokens, 
                                                api_dict=api_dict)
            conv.append({"role": "assistant", "content": output})

            turns.append({"content": output, "token_len": len(encoding.encode(output))})
        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans) + "\n")
    return


def gen_api_answer(args: ArenaHardGenArgs):
    assert args.openai_api_base is not None
    openai.api_base = args.openai_api_base  # set it to the custom api base
    
    model = args.model_id
    settings = {
        "bench_name": args.bench_name,
        "max_tokens": args.max_new_token,
        "num_choices": args.num_choices,
        "temperature": args.gen_temperature,
    }
    endpoint_info = {
        "api_type": args.api_type,
        "model_name": args.model_id,
        "tokenizer": args.tokenizer_path,
        "endpoint": {
            "api_base": args.openai_api_base,
            "api_key": None,
        },
        # "system_prompt": args.system_prompt,  # no system prompt for now
        "parallel": args.gen_parallel,
    }

    existing_answer = load_model_answers(os.path.join("data", settings["bench_name"], "model_answer"))
    

    question_file = os.path.join("data", settings["bench_name"], "question.jsonl")
    questions = load_questions(question_file)

    answer_file = os.path.join("data", settings["bench_name"], "model_answer", f"{model}.jsonl")
    print(f"Output to {answer_file}")

    parallel = endpoint_info["parallel"]

    # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
    if  endpoint_info["tokenizer"] != '':
        question_list = [question["turns"][0]["content"] for question in questions]
        if model in OPENAI_MODEL_LIST:
            tokenizer = tiktoken.encoding_for_model(endpoint_info["model_name"])
            tokens = [tokenizer.encode(prompt) for prompt in question_list]
            max_tokens = [(settings["max_tokens"] - len(token) - 100) for token in tokens]
        else:
            from transformers import AutoTokenizer
            
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"])

            tokens = tokenizer(question_list)
            max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
    else:
        max_tokens = [settings["max_tokens"]] * len(questions)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = []
        count = 0
        for index, question in enumerate(questions):
            if model in existing_answer and question["question_id"] in existing_answer[model]:
                count += 1
                continue
            future = executor.submit(
                get_gen_answer,
                question,
                model,
                endpoint_info,
                settings["num_choices"],
                max_tokens[index],
                settings["temperature"],
                answer_file,
            )
            futures.append(future)
        if count > 0:
            print(f"{count} number of existing answers")
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
    return


def get_judge_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


# get answer from model
def get_judge_answer(model, conv, temperature, max_tokens, endpoint_dict=None, token_stats=None):
    api_dict = endpoint_dict["endpoint"]

    if endpoint_dict["api_type"] == "anthropic":
        output = chat_completion_anthropic(model, conv, temperature, max_tokens)
    else:
        output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict, token_stats=token_stats)

        if (token_stats["num_requests"] + 1) % 25 == 0:
            compl_token = token_stats['completion_tokens']
            prompt_token = token_stats['prompt_tokens']
            req = token_stats['num_requests']
            print(f"Avg. completion tokens: {compl_token/req:.2f}, Avg. prompt tokens: {prompt_token/req:.2f}")

            # reset
            token_stats['completion_tokens'] = 0
            token_stats['prompt_tokens'] = 0
            token_stats['num_requests'] = 0
    return output


def get_judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id":question["question_id"],
        "model":answer["model_id"],
        "judge": model,
        "games":[]
    }
    token_stats = args["token_stats"]
    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i+1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1: # swap position
                    temp = baseline
                    baseline = answer
                    answer = temp

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]
            
            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(2):
            new_judgment = get_judge_answer(
                model,
                conv,
                configs["temperature"],
                configs["max_tokens"],
                args["endpoint_dict"],
                token_stats=token_stats
            )

            judgment += ("\n" + new_judgment)

            score, try_again = get_judge_score(judgment, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        result = {
            "user_prompt": conv[1]["content"],
            "judgment": judgment,
            "score":score
        }
        output["games"].append(result)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")
    return


def gen_judgment(args: ArenaHardJudgeArgs):
    if hasattr(openai, "api_base"):
        openai.api_base = None  # reset openai api base to default

    # directly copied from gen_judgment.py, with some minor changes
    configs = make_config("src/configs/judge_config.yaml")
    configs['judge_model'] = args.judge_model
    configs['baseline_model'] = args.baseline_model
    configs['model_list'] = args.model_list
    endpoint_list = make_config("src/configs/judge_api_config.yaml")

    print(f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
          + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}')

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]
    
    output_files = {}
    # output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    judge_model = configs["judge_model"]
    baseline_model = configs["baseline_model"]
    output_dir = f"data/{configs['bench_name']}/model_judgment/{judge_model}/vs_{baseline_model}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    endpoint_info = endpoint_list[configs["judge_model"]]
    endpoint_info["parallel"] = args.judge_parallel

    token_stats = {
        'completion_tokens': 0,
        'prompt_tokens': 0,
        'num_requests': 0
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_info["parallel"]) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                question_id = question["question_id"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not question_id in model_answers[model]:
                    print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                    continue

                if model in existing_judgments and question_id in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][question_id]
                if ref_answers:
                    kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                    assert len(kwargs["reference"]) == len(configs["ref_model"])
                else:
                    kwargs["reference"] = None
                if configs["baseline"]:
                    kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
                else:
                    kwargs["baseline_answer"] = None
                kwargs["configs"] = configs
                kwargs["endpoint_dict"] = endpoint_info
                kwargs["output_file"] = output_files[model]
                kwargs["regex_pattern"] = pattern
                kwargs["token_stats"] = token_stats
                future = executor.submit(get_judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
    return