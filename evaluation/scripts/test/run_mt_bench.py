from fastchat.llm_judge.gen_model_answer import (
    get_conversation_template
)
from src.evaluation.mt_bench_gen import (
    gen_model_answer, gen_api_answer, gen_judgment,
    MTBenchGenArgs, MTBenchJudgeArgs, GenericArgs
)
from src.evaluation.mt_bench_init import register_all
from src.evaluation.utils import (
    find_free_port, run_sglang_server,
    find_checkpoint_paths
)
from src.trainers.configs import H4ArgumentParser
from src.utils.utils import create_dir_if_not_exists
from pathlib import Path
import pandas as pd
import json
import numpy as np
import wandb
import yaml
import signal
import time
import os


available_templates = register_all()  # register all relevant chat templates


def maybe_init_sglang(args: MTBenchGenArgs, result_save_path=''):
    if args.use_sglang:
        free_port = find_free_port()
        print(f"Running sglang at port {free_port}.")

        if result_save_path == "":
            path = Path(args.model_path)
            result_save_path = path.joinpath("sglang_log.txt")
        else:
            path = Path(result_save_path)
            result_save_path = path.joinpath("sglang_log.txt")

        sglang_process = run_sglang_server(
            args.model_path,
            free_port,
            args.chat_template,
            result_save_path=result_save_path
        )
        args.openai_api_base = f"http://127.0.0.1:{free_port}/v1"
        return sglang_process
    else:
        return None


def gather_result_single(judge_args: MTBenchJudgeArgs):
    questions_file = "data/mt_bench/question.jsonl"

    df_questions = pd.read_json(questions_file, lines=True)
    df_qinfo = df_questions[["question_id", "category"]]

    input_file = (
        f"data/{judge_args.bench_name__}/model_judgment/{judge_args.judge_model}_single.jsonl"
    )

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["question_id", "model", "score", "turn"]]
    df = df[df["score"] != -1]

    if judge_args.model_list is not None:
        df = df[df["model"].isin(judge_args.model_list)]

    if judge_args.bench_name__ == "mt_bench":
        print("########## Details ##########")
        easier_copy_paste = {
            "turn 1": [],
            "turn 2": [],
            "turn avg": [],
            "categories": [],
        }
        df_details = df.merge(df_qinfo, on="question_id")
        for idx, sample in df_details.groupby(["model", "turn", "category"]).mean().iterrows():
            model_name, turn, category = idx
            score = sample["score"]
            print(f"{model_name} (turn {turn}, {category}): {score:.2f}")
            easier_copy_paste[f"turn {turn}"].append(score)

        for idx, sample in df_details.groupby(["model", "category"]).mean().iterrows():
            model_name, category = idx
            score = sample["score"]
            print(f"{model_name} ({category} avg): {score:.5f}")
            easier_copy_paste["turn avg"].append(score)
            easier_copy_paste["categories"].append(category)
        

        for k, scores in easier_copy_paste.items():
            print(f"{k}:")
            if k == "categories":
                print(scores)
            else:
                scores = [f"{s:.2f}" for s in scores]
                print(f"{','.join(scores)}")
    return easier_copy_paste


def compute_avg_performance(run_performances):
    categories = run_performances[0]["categories"]
    avg_performance = {}
    for p in run_performances:
        turn_avg = p["turn avg"]
        for c, score in zip(categories, turn_avg):
            if c not in avg_performance:
                avg_performance[c] = []
            avg_performance[c].append(score)
    
    for c in avg_performance:
        avg_performance[c] = np.mean(avg_performance[c])
    avg_performance['overall'] = np.mean([v for v in avg_performance.values()])
    return avg_performance


def evaluate_single_model(gen_args: MTBenchGenArgs, judge_args: MTBenchJudgeArgs, args: GenericArgs, wandb_suffix: str = ''):
    run_performances = []
    base_model_id = gen_args.model_id

    ## generate everything first, then evaluate
    ## loop of gen + evaluate sometimes hangs for unknown reasons at the THIRD run
    sglang_process = maybe_init_sglang(gen_args, args.result_save_path)
    for i in range(args.num_runs):
        print(f"Run {i+1}/{args.num_runs} generation in progress...")

        model_id = f"{base_model_id}_run{i}"
        gen_args.model_id = model_id
        
        if gen_args.openai_api_base is not None:
            gen_api_answer(gen_args)
        else:
            gen_model_answer(gen_args)
    if sglang_process is not None:
        time.sleep(5)
        os.killpg(os.getpgid(sglang_process.pid), signal.SIGTERM)
    
    ## judge everything
    for i in range(args.num_runs):
        print(f"Run {i+1}/{args.num_runs} judgment in progress...")

        model_id = f"{base_model_id}_run{i}"
        judge_args.model_list = [model_id]

        gen_judgment(judge_args)
        performance = gather_result_single(judge_args)
        run_performances.append(performance)

    ### average performance
    avg_performance = compute_avg_performance(run_performances)
    print("Average performance:")
    print(json.dumps(avg_performance, indent=4))

    ### log to wandb
    if args.wandb_id != '':
        wandb.init(
            project=args.wandb_project,
            id=args.wandb_id,
            resume=True
        )
        wandb_perf = {
            f"mt-bench_{judge_args.judge_model}{wandb_suffix}/{k}": v 
            for k, v in avg_performance.items()
        }
        wandb.log(wandb_perf)
        wandb.finish()
    
    ### save result to file
    to_save_performance = avg_performance.copy()
    if args.result_save_path == "":
        path = Path(gen_args.model_path)
        save_file_path = path.joinpath(f"mt_bench_{judge_args.judge_model}.json")
    else:
        path = Path(args.result_save_path)
        save_file_path = path.joinpath(f"mt_bench_{judge_args.judge_model}.json")
    
    with open(save_file_path, "w", encoding="utf-8") as fwrite:
        # add some info in to_save_performance
        categories = [k for k in run_performances[0]['categories']]
        easier_copy_paste = [
            to_save_performance[c] for c in categories
        ]
        to_save_performance['easier_copy_paste'] = ','.join([f"{v:.2f}" for v in easier_copy_paste])
        to_save_performance['num_runs'] = args.num_runs
        to_save_performance['all_run_performances'] = run_performances
        json.dump(to_save_performance, fwrite, indent=4)
    return


def main(gen_args: MTBenchGenArgs, judge_args: MTBenchJudgeArgs, args: GenericArgs):
    ### set wandb if found
    if args.to_wandb and args.wandb_id == "":
        path = Path(gen_args.model_path)
        if 'checkpoint' in path.name:
            # check parent dir for run_args.yaml
            path = path.parent
            print(f"Checking parent dir for run_args.yaml: {path}")
        
        assert path.joinpath("run_args.yaml").is_file(), f"File not found at {path.joinpath('run_args.yaml')}"
        with open(path.joinpath("run_args.yaml"), "r", encoding="utf-8") as fread:
            all_args = yaml.load(fread, Loader=yaml.Loader)
        args.wandb_id = all_args['wandb_id']
        args.wandb_project = all_args['wandb_project']
        print(f"Read wandb info from {path.joinpath('run_args.yaml')}: {args.wandb_project}/{args.wandb_id}")
    ### check save dir
    if args.result_save_path == "":
        path = Path(gen_args.model_path)
        if not path.exists():
            raise FileNotFoundError((
                f"{gen_args.model_path} not found. "
                "You are likely using a pretrained model path as save folder. "
                "Please specify a dedicated save folder using --result_save_path."
            ))
    else:
        # make sure the dir exists
        create_dir_if_not_exists(args.result_save_path)

    ### gen answer and judge
    judge_args.bench_name__ = gen_args.bench_name
    conv = get_conversation_template(gen_args.model_id)
    sglang_template = gen_args.chat_template
    print("[[If you are not using sglang]], you will be using conversation template:", conv)
    print("[[If you are using sglang]], you will be using chat template:", sglang_template)

    base_model_path = gen_args.model_path
    base_model_id = gen_args.model_id
    # scans the base directory to find all model weights.
    # this can either be the base_dir itself, or be base_dir/checkpoint-xxx
    for model_path_found, suffix in find_checkpoint_paths(base_model_path):
        print(f"Model weights found at {model_path_found}. Using model_id={base_model_id}{suffix}.")
        gen_args.model_path = model_path_found
        gen_args.model_id = f'{base_model_id}{suffix}'
        evaluate_single_model(gen_args, judge_args, args, wandb_suffix=suffix)
        break
    return


if __name__ == '__main__':
    parser = H4ArgumentParser((MTBenchGenArgs, MTBenchJudgeArgs, GenericArgs))
    gen_args, judge_args, args = parser.parse()
    
    main(gen_args, judge_args, args)