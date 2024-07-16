from lionalign.evaluation.alpaca_eval_gen import (
    gen_judgment, AlpacaEvalJudgeArgs
)
from lionalign.arguments import H4ArgumentParser
import os


def main(args: AlpacaEvalJudgeArgs):
    if args.model_outputs == '':
        # use output_path
        gen_result_file = os.path.join(args.output_path, 'model_outputs.json')
        if not os.path.exists(gen_result_file):
            raise FileNotFoundError(f"Model outputs not found at {gen_result_file}")
        args.model_outputs = gen_result_file
    
    judge_result_file = os.path.join(args.output_path, args.annotators_config, 'leaderboard.csv')
    if not os.path.exists(judge_result_file):
        raise FileNotFoundError(f"Judgment results not found at {judge_result_file}")

    args.leaderboard_mode_to_print = None  # filter none, display all
    gen_judgment(args)
    return


if __name__ == '__main__':
    parser = H4ArgumentParser((AlpacaEvalJudgeArgs,))
    judge_args = parser.parse()
    
    main(judge_args)