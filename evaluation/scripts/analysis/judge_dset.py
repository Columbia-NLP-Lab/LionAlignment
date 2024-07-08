from distilabel.tasks import UltraFeedbackTask, Prompt
from datasets import Dataset, load_dataset
from src.utils.data_utils import add_full_id
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from src.trainers.configs import H4ArgumentParser
import random
import openai
import tiktoken
import concurrent.futures
import pandas as pd


@dataclass
class JudgmentArguments:
    dset_name: str = field(
        default="when2rl/UltraFeedback_binarized_cleaned_annotated",
        metadata={"help": "Dataset name"}
    )
    dset_split: str = field(
        default="train_prefs",
        metadata={"help": "Dataset split"}
    )
    eval_mode: str = field(
        default="all",
        metadata={
            "choices": ["all", "single"],
            "help": "Whether to judge all samples or just a single one at a time"
        }
    )
    output_path: str = field(
        default="data/openhermes/gpt-4-turbo-preference.csv",
        metadata={"help": "Path to save the output CSV file"}
    )
    num_to_judge: int = field(
        default=10,
        metadata={"help": "Number of samples to judge"}
    )
    judge_model: str = field(
        default="gpt-4-0125-preview",
        metadata={"help": "Model to use for judging"}
    )
    judge_parallel: int = field(
        default=8,
        metadata={"help": "Number of parallel judgments"}
    )
    

    def __post_init__(self):
        if self.eval_mode not in ["all", "single"]:
            raise ValueError("eval_mode must be either 'all' or 'single'")
        return



def format_chat(chat_list):
    mapping_role = {"user": "User: ", "assistant":"Assistant: "}
    input_text = "\n"
    for e in chat_list:
        role = e["role"]
        content = e["content"]

        input_text += f"{mapping_role[role]}{content}\n"
    return input_text


def format_conversation(conversation_list):
    all_but_last = conversation_list[:-1]
    input_text = format_chat(all_but_last)
    output = conversation_list[-1]["content"]
    return input_text, output


def get_model_judgment(
    judge_model, client: openai.Client, task: UltraFeedbackTask,
    result_list: list, full_id: str, side: list, prompt: Prompt
):
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.formatted_prompt},
            ],
            temperature=0,
            max_tokens=256,
        )
        ratings = task.parse_output(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in getting model response: {e}")
        ratings = [{}, {}]

    result_list.append({
        "full_id": full_id,
        "side": side,
        "ratings": ratings,
    })
    return


def maybe_flatten_judgments(judgment_df):
    flattened = []
    for _, row in judgment_df.iterrows():
        full_id = row["full_id"]
        sides = row["side"]
        ratings = row["ratings"]

        for side, rating in zip(sides, ratings):
            flattened.append({
                "full_id": full_id,
                "side": side,
                "rating": rating
            })
    return pd.DataFrame(flattened)


def prepare_dataset(args: JudgmentArguments, raw_dataset: Dataset):
    split_name = args.dset_split

    random.seed(42)

    reformatted_data = []
    for sample in tqdm(raw_dataset[split_name]):
        full_id = sample["full_id"]
        chosen_conv = sample["chosen"]
        rejected_conv = sample["rejected"]
        chosen_text, chosen_output = format_conversation(chosen_conv)
        rejected_text, rejected_output = format_conversation(rejected_conv)

        assert chosen_text == rejected_text
        if chosen_output == rejected_output:
            print("Warning: chosen and rejected output are the same for sample", full_id)
            continue

        # swap the order of the generations
        generations = [chosen_output, rejected_output]
        sides = ["chosen", "rejected"]

        # randomize the order
        if random.random() > 0.5:
            generations = generations[::-1]
            sides = sides[::-1]

        reformatted_data.append({
            "full_id": full_id,
            "input": chosen_text,
            "generations": generations,
            "sides": sides
        })

    if args.num_to_judge == -1:
        prepared_dataset = Dataset.from_list(reformatted_data)
        return prepared_dataset
    
    reformatted_data = random.sample(reformatted_data, args.num_to_judge)
    prepared_dataset = Dataset.from_list(reformatted_data)
    return prepared_dataset


def generate_judgments(args: JudgmentArguments, prepared_dataset: Dataset):
    task = UltraFeedbackTask.for_overall_quality()

    client = openai.Client()
    tokenizer = tiktoken.encoding_for_model(args.judge_model)

    RESULT_LIST = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.judge_parallel) as executor:
        futures = []
        for sample in prepared_dataset:
            full_id = sample['full_id']
            sides = sample['sides']
            input_text = sample['input']
            generations = sample['generations']

            est_length = len(tokenizer.encode(input_text + generations[0]))
            if est_length > 4096:
                print(f"Skipping sample {full_id} due to length")
                continue

            ## either judge both at the same time (comparison based)
            if args.eval_mode == "all":
                prompt = task.generate_prompt(
                    input=input_text,
                    generations=generations
                )

                future = executor.submit(
                    get_model_judgment,
                    
                    args.judge_model,
                    client,
                    task,
                    RESULT_LIST,
                    full_id,
                    sides,
                    prompt,
                )
                futures.append(future)
            else:
                ## or judge them one by one
                for side, gen in zip(sides, generations):
                    prompt = task.generate_prompt(
                        input=input_text,
                        generations=[gen]
                    )

                    future = executor.submit(
                        get_model_judgment,
                        args.judge_model,
                        client,
                        task,
                        RESULT_LIST,
                        full_id,
                        [side],
                        prompt,
                    )
                    futures.append(future)
        
        num_completed = 0
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Getting model judgments"
        ):
            future.result()
            num_completed += 1

            if num_completed % 10 == 0:
                tmp_df = pd.DataFrame(RESULT_LIST)
                tmp_df = maybe_flatten_judgments(tmp_df)
                tmp_df.to_csv(args.output_path, index=False)

    tmp_df = pd.DataFrame(RESULT_LIST)
    tmp_df = maybe_flatten_judgments(tmp_df)
    tmp_df.to_csv(args.output_path, index=False)
    return


def main(args: JudgmentArguments):
    raw_dataset = load_dataset(args.dset_name)
    raw_dataset = raw_dataset.map(
        add_full_id,
        num_proc=32,
        desc="Adding full_id"
    )

    prepared_dataset = prepare_dataset(args, raw_dataset)
    
    generate_judgments(args, prepared_dataset)
    return


if __name__ == "__main__":
    parser = H4ArgumentParser((JudgmentArguments))
    judge_args = parser.parse()
    
    print('received args', judge_args)
    main(judge_args)