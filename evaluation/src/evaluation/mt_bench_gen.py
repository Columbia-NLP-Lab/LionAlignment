from fastchat.llm_judge.gen_model_answer import (
    run_eval,
    str_to_torch_dtype,
    reorg_answer_file
)
from fastchat.llm_judge.gen_judgment import (
    make_judge_single,
    make_judge_pairwise,
    make_match_single,
    make_match,
    make_match_all_pairs,
)
from src.evaluation.mt_bench_init import register_all
from src.evaluation.mt_bench_common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    NEED_REF_CATS,
    temperature_config,
    ANTHROPIC_MODEL_LIST,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_palm,
    get_conversation_template,
)
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List
import concurrent
import json
import numpy as np
import openai
import os
import time
import shortuuid


available_templates = register_all()  # register all relevant chat templates


@dataclass
class MTBenchGenArgs:
    """args for answer generation
    """
    model_path: str = field(
        metadata={"help": "The path to the weights. This can be a local folder or a Hugging Face repo ID."}
    )
    model_id: str = field(metadata={"help": "A custom name for the model."})
    bench_name: str = field(
        default="mt_bench", metadata={"help": "The name of the benchmark question set."}
    )
    question_begin: int = field(
        default=None, metadata={"help": "A debug option. The begin index of questions."}
    )
    question_end: int = field(
        default=None, metadata={"help": "A debug option. The end index of questions."}
    )
    max_new_token: int = field(
        default=1024, metadata={"help": "The maximum number of new generated tokens."}
    )
    num_choices: int = field(
        default=1, metadata={"help": "How many completion choices to generate."}
    )
    num_gpus_per_model: int = field(
        default=1, metadata={"help": "The number of GPUs per model."}
    )
    num_gpus_total: int = field(
        default=1, metadata={"help": "The total number of GPUs."}
    )
    max_gpu_memory: str = field(
        default=None, metadata={"help": "Maxmum GPU memory used for model weights per GPU."}
    )
    dtype: str = field(
        default=None,
        metadata={
            "help": "Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
            "choices": ["float32", "float16", "bfloat16"],
        },
    )
    revision: str = field(
        default="main", metadata={"help": "The model revision to load."}
    )
    ### api_gen related args
    use_sglang: bool = field(
        default=False, metadata={"help": "Whether to use SGlang for geneation."}
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
            if all([template not in self.model_id for template in available_templates]):
                raise ValueError(f"Template name not found in model-id. Available ones: {available_templates}.")
        return


@dataclass
class MTBenchJudgeArgs:
    """args for judgment generation
    """
    bench_name__: str = field(
        default="mt_bench", metadata={"help": "The name of the benchmark question set."}
    )
    judge_file: str = field(
        default="data/judge_prompts.jsonl", metadata={"help": "The file of judge prompts."}
    )
    judge_model: str = field(default="gpt-4-0125-preview")
    baseline_model: str = field(default="gpt-3.5-turbo")
    mode: str = field(
        default="single",
        metadata={
            "help": (
                "Evaluation mode. "
                "`pairwise-baseline` runs pairwise comparision against a baseline. "
                "`pairwise-all` runs pairwise comparision between all pairs. "
                "`single` runs single answer grading."
            ),
            "choices": ["single", "pairwise-baseline", "pairwise-all"],
        },
    )
    model_list: List[str] = field(
        default_factory=lambda: [], metadata={"help": "A list of models to be evaluated"}
    )
    judge_parallel: int = field(
        default=1, metadata={"help": "The number of concurrent API calls."}
    )
    first_n: int = field(
        default=None, metadata={"help": "A debug option. Only run the first `n` judgments."}
    )
    y: bool = field(default=False, metadata={"help": "Skip the confirmation prompt."})


@dataclass
class GenericArgs:
    """args for logging + other generic args
    """
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


def gen_model_answer(mt_bench_args: MTBenchGenArgs):
    # directly copied from gen_model_answer.py, with some minor changes
    if mt_bench_args.num_gpus_total // mt_bench_args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{mt_bench_args.bench_name}/question.jsonl"
    answer_file = f"data/{mt_bench_args.bench_name}/model_answer/{mt_bench_args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=mt_bench_args.model_path,
        model_id=mt_bench_args.model_id,
        question_file=question_file,
        question_begin=mt_bench_args.question_begin,
        question_end=mt_bench_args.question_end,
        answer_file=answer_file,
        max_new_token=mt_bench_args.max_new_token,
        num_choices=mt_bench_args.num_choices,
        num_gpus_per_model=mt_bench_args.num_gpus_per_model,
        num_gpus_total=mt_bench_args.num_gpus_total,
        max_gpu_memory=mt_bench_args.max_gpu_memory,
        dtype=str_to_torch_dtype(mt_bench_args.dtype),
        revision=mt_bench_args.revision,
    )

    reorg_answer_file(answer_file)
    return


def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str
):
    if "required_temperature" in question.keys():
        temperature = question["required_temperature"]
    elif question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2 model
    for i in range(num_choices):
        conv = get_conversation_template(model)
        conv.system_message = ""  # set system message in --chat_template flag with sglang instead.

        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            if model in ANTHROPIC_MODEL_LIST:
                output = chat_completion_anthropic(model, conv, temperature, max_tokens)
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            else:
                output = chat_completion_openai(model, conv, temperature, max_tokens)

            conv.update_last_message(output)
            turns.append(output)

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
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")
    return


def gen_api_answer(args: MTBenchGenArgs):
    if args.openai_api_base is not None:
        openai.api_base = args.openai_api_base  # set it to the custom api base

    question_file = f"data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"
    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.gen_parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model_id,
                args.num_choices,
                args.max_new_token,
                answer_file,
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
    return


def gen_judgment(mt_bench_args: MTBenchJudgeArgs):
    if hasattr(openai, "api_base"):
        openai.api_base = None  # reset openai api base to default

    # directly copied from gen_judgment.py, with some minor changes
    question_file = f"data/{mt_bench_args.bench_name__}/question.jsonl"
    answer_dir = f"data/{mt_bench_args.bench_name__}/model_answer"
    ref_answer_dir = f"data/{mt_bench_args.bench_name__}/reference_answer"

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(mt_bench_args.judge_file)

    if mt_bench_args.first_n:
        questions = questions[: mt_bench_args.first_n]

    if mt_bench_args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = mt_bench_args.model_list

    if mt_bench_args.mode == "single":
        judges = make_judge_single(mt_bench_args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"data/{mt_bench_args.bench_name__}/model_judgment/{mt_bench_args.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(mt_bench_args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"data/{mt_bench_args.bench_name__}/model_judgment/{mt_bench_args.judge_model}_pair.jsonl"
        )
        if mt_bench_args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = mt_bench_args.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = mt_bench_args.bench_name__
    match_stat["mode"] = mt_bench_args.mode
    match_stat["judge"] = mt_bench_args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    if not mt_bench_args.y:
        input("Press Enter to confirm...")

    # Play matches
    if mt_bench_args.judge_parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(mt_bench_args.judge_parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass
    return