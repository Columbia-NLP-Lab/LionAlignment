from src.evaluation.arena_hard_gen import (
    gen_api_answer, gen_judgment,
    ArenaHardGenArgs, ArenaHardJudgeArgs, GenericArgs
)
from src.evaluation.show_arena_hard_result import (
    get_battles_from_judgment, load_model_answers,
    compute_mle_elo, get_bootstrap_result,
    get_win_rate_column
)
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
import os
import time
import signal


def init_sglang(args: ArenaHardGenArgs, result_save_path=''):
    assert args.use_sglang, "Arena hard assumes model api with sglang."

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


def gather_result_single(args: ArenaHardJudgeArgs):
    assert len(args.model_list) == 1, "Only single model evaluation is supported."

    answer_dir = os.path.join("data", args.bench_name__, "model_answer")
    model_answers = load_model_answers(answer_dir)

    battles = get_battles_from_judgment(args.judge_model, args.baseline_model, False, 3)
    bootstrap_online_elo = compute_mle_elo(battles)

    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, num_round=100)

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)

        length = 0
        if model in model_answers:
            for _, row in model_answers[model].items():
                turn = row["choices"][0]["turns"][0]
                length += turn["token_len"]
            length /= len(model_answers[model])

        stats.at[i, "avg_tokens"] = int(length)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()
    
    stats.sort_values(by="model", inplace=True)
    stats["score"] = get_win_rate_column(stats, "score", args.baseline_model).tolist()
    stats["lower"] = get_win_rate_column(stats, "lower", args.baseline_model).tolist()
    stats["upper"] = get_win_rate_column(stats, "upper", args.baseline_model).tolist()
    decimal = 1
    
    stats.sort_values(by="score", ascending=False, inplace=True)
    curr_model_perf = {}
    models_joined = []
    for _, row in stats.iterrows():
        interval = str((round(row['lower'] - row['score'], decimal), round(row['upper'] - row['score'], decimal)))
        curr_model = row['model']
        print(f"{curr_model : <65} | score: {round(row['score'], decimal) : ^5} | 95% CI: {interval : ^12} | average #tokens: {int(row['avg_tokens'])}")

        models_joined.append(curr_model)
        if curr_model in args.model_list:
            curr_model_perf = {
                'score': round(row['score'], decimal),
                '95% CI': interval,
                'avg_tokens': int(row['avg_tokens'])
            }
    curr_model_perf['models_in_arena'] = models_joined
    return curr_model_perf


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
        if k in ['95% CI', 'models_in_arena']:
            # skip confidence interval as its a tuple
            continue
        avg_performance[k] = np.mean(v)
    return avg_performance


def evaluate_single_model(gen_args: ArenaHardGenArgs, judge_args: ArenaHardJudgeArgs, args: GenericArgs, wandb_suffix: str = ''):
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
            
            # arena hard only uses model api to generate answers
            gen_api_answer(gen_args)
        # kill sglang
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
            f"arena-hard_{judge_args.judge_model}{wandb_suffix}/baseline_{judge_args.baseline_model}/{k}": v 
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
    save_file_path = path.joinpath(f"arena-hard_{judge_args.judge_model}-baseline_{judge_args.baseline_model}.json")
    
    with open(save_file_path, "w", encoding="utf-8") as fwrite:
        # add some info in to_save_performance
        to_save_performance['num_runs'] = args.num_runs
        to_save_performance['all_run_performances'] = run_performances
        json.dump(to_save_performance, fwrite, indent=4)
    return


def main(gen_args: ArenaHardGenArgs, judge_args: ArenaHardJudgeArgs, args: GenericArgs):
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
    sglang_template = gen_args.chat_template
    print("[[If you are using sglang]], you will be using chat template:", sglang_template)

    if gen_args.chat_template in [None, 'None', 'none', '']:
        gen_args.chat_template = None

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
    parser = H4ArgumentParser((ArenaHardGenArgs, ArenaHardJudgeArgs, GenericArgs))
    gen_args, judge_args, args = parser.parse()
    
    main(gen_args, judge_args, args)