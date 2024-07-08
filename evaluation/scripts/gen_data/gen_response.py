import concurrent.futures
import openai
import hashlib
import pandas as pd
import argparse
from datasets import load_dataset
from copy import deepcopy
from tqdm.auto import tqdm
from src.utils.data_utils import add_full_id


def get_model_answers(
    client: openai.Client,
    result_list: list,
    sample: dict,
    prompt_messages: list,
    temperature: float = 0.0,
    max_tokens: int = 512,
    n: int = 1
):
    answers = set()

    messages = sample['chosen'][:-1]
    for _ in range(n):
        try:
            response = client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error in getting model response: {e}")
            continue
        
        model_answer = response.choices[0].message.content
        answers.add(model_answer)

    full_id = sample["full_id"]
    result_list.append({
        "full_id": full_id,
        "gen_answers": list(answers),
    })
    return


def main(args: argparse.Namespace):
    # save_file = "data/dpo-mix-7k/gemma_2b_sft.csv"
    save_file = args.save_file

    client = openai.Client(
        base_url="http://127.0.0.1:41911/v1", api_key="EMPTY"
    )

    dpo_dataset = load_dataset("when2rl/dpo-mix-7k-rescaled_reformatted", split="train")
    dpo_dataset = dpo_dataset.map(
        add_full_id,
        num_proc=8,
        keep_in_memory=True,
        desc="Adding full_id to the dataset",
    )

    prompt_messages = []
    
    RESULT_LIST = []

    n_answers = 1
    n_parallel = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = []
        for idx, sample in enumerate(dpo_dataset):
            future = executor.submit(
                get_model_answers,
                client,
                RESULT_LIST,
                sample,
                prompt_messages,
                0.0,
                1024,
                n_answers,
            )
            futures.append(future)
        
        num_completed = 0
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Getting model answers"
        ):
            future.result()
            num_completed += 1

            if num_completed % 200 == 0:
                print(f"Completed {num_completed} samples")
                tmp_df = pd.DataFrame(RESULT_LIST)
                tmp_df.to_csv(save_file, index=False)

    tmp_df = pd.DataFrame(RESULT_LIST)
    tmp_df.to_csv(save_file, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_file",
        type=str,
        default="data/dpo-mix-7k/gemma_2b_sft.csv"
    )
    args = parser.parse_args()

    main(args)