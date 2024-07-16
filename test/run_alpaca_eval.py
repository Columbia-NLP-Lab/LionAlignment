from lionalign.evaluation.alpaca_eval_gen import (
    gen_api_answer, gen_judgment,
    AlpacaEvalGenArgs, AlpacaEvalJudgeArgs, GenericArgs
)
from lionalign.evaluation.utils import (
    find_free_port, run_sglang_server,
    find_checkpoint_paths
)
from lionalign.arguments import H4ArgumentParser
from lionalign.data.utils import create_dir_if_not_exists
from pathlib import Path
import pandas as pd
import json
import numpy as np
import wandb
import yaml
import os
import time
import signal


def init_sglang(args: AlpacaEvalGenArgs, result_save_path=''):
    assert args.use_sglang, "Alpaca Eval 2.0 assumes model api with sglang."

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
        result_save_path=result_save_path,
        tokenizer_path=args.tokenizer_path
    )
    args.openai_api_base = f"http://127.0.0.1:{free_port}/v1"
    return sglang_process


def gather_result_single(args: AlpacaEvalJudgeArgs):
    eval_leaderboard_path = os.path.join(args.output_path, "weighted_alpaca_eval_gpt4_turbo", "leaderboard.csv")
    df = pd.read_csv(eval_leaderboard_path)
    df = df.rename(columns={'Unnamed: 0': 'model_name'})
    df.index = df['model_name'].values

    # curr model
    curr_perf = df.loc[args.name].to_dict()
    curr_perf.pop('model_name')
    curr_perf.pop('mode')
    return curr_perf


def compute_avg_performance(run_performances):
    keys = run_performances[0].keys()
    avg_performance_ = {}
    for p in run_performances:
        for k in keys:
            if k not in avg_performance_:
                avg_performance_[k] = []
            avg_performance_[k].append(p[k])

    avg_performance = {}
    for k, v in avg_performance_.items():
        avg_performance[k] = np.mean(v)
    return avg_performance


def evaluate_single_model(gen_args: AlpacaEvalGenArgs, judge_args: AlpacaEvalJudgeArgs, args: GenericArgs, wandb_suffix: str = ''):
    run_performances = []
    base_model_id = gen_args.model_id

    ## generate everything first, then evaluate
    ## loop of gen + evaluate sometimes hangs for unknown reasons at the THIRD run
    if args.judge_only:
        print('Judging only...')
    else:
        sglang_process = init_sglang(gen_args, args.result_save_path)
        for i in range(args.num_runs):
            print(f"Run {i+1}/{args.num_runs} generation in progress...")

            model_id = f"{base_model_id}_run{i}"
            gen_args.model_id = model_id
            
            gen_api_answer(gen_args)
        # kill sglang
        time.sleep(5)
        os.killpg(os.getpgid(sglang_process.pid), signal.SIGTERM)
    
    ## judge everything
    for i in range(args.num_runs):
        print(f"Run {i+1}/{args.num_runs} judgment in progress...")

        model_id = f"{base_model_id}_run{i}"
        model_answer_path = os.path.join("data", "alpaca_eval_results", model_id)
        model_answer_file = os.path.join(model_answer_path, "model_outputs.json")
        judge_args.name = model_id
        judge_args.model_outputs = model_answer_file
        judge_args.output_path = model_answer_path

        gen_judgment(judge_args)

        performance = gather_result_single(judge_args)
        run_performances.append(performance)

    print(run_performances)
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
            f"alpaca-eval-2.0{wandb_suffix}/{k}": v 
            for k, v in avg_performance.items()
        }
        wandb.log(wandb_perf)
        wandb.finish()
    
    ### save result to file
    to_save_performance = avg_performance.copy()
    if args.result_save_path == "":
        path = Path(gen_args.model_path)
    else:
        path = Path(args.result_save_path)
    save_file_path = path.joinpath("alpaca-eval-2.0.json")
    
    with open(save_file_path, "w", encoding="utf-8") as fwrite:
        # add some info in to_save_performance
        to_save_performance['num_runs'] = args.num_runs
        to_save_performance['all_run_performances'] = run_performances
        json.dump(to_save_performance, fwrite, indent=4)
    return


def main(gen_args: AlpacaEvalGenArgs, judge_args: AlpacaEvalJudgeArgs, args: GenericArgs):
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
    sglang_template = gen_args.chat_template
    print("[[If you are using sglang]], you will be using chat template:", sglang_template)

    base_model_path = gen_args.model_path
    base_model_id = gen_args.model_id
    base_judged_model_name = judge_args.name
    # scans the base directory to find all model weights.
    # this can either be the base_dir itself, or be base_dir/checkpoint-xxx
    for model_path_found, suffix in find_checkpoint_paths(base_model_path):
        print(f"Model weights found at {model_path_found}. Using model_id={base_model_id}{suffix}.")
        gen_args.model_path = model_path_found
        gen_args.model_id = f'{base_model_id}{suffix}'
        if base_judged_model_name == '':
            judge_args.name = f'{base_model_id}{suffix}'
        evaluate_single_model(gen_args, judge_args, args, wandb_suffix=suffix)
    return


if __name__ == '__main__':
    parser = H4ArgumentParser((AlpacaEvalGenArgs, AlpacaEvalJudgeArgs, GenericArgs))
    gen_args, judge_args, args = parser.parse()
    
    main(gen_args, judge_args, args)