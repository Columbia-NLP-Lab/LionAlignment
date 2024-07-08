from distilabel.tasks import UltraFeedbackTask, Prompt
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from src.utils.data_utils import add_full_id
from src.trainers.configs import H4ArgumentParser
import random
import openai
import tiktoken
import concurrent.futures
import pandas as pd
import pickle


@dataclass
class JudgmentArguments:
    """judge the generations specified in the gen_data_path"""

    dset_name: str = field(
        default="when2rl/dpo-mix-7k-rescaled_reformatted",
        metadata={"help": "Dataset name"}
    )
    dset_split: str = field(
        default="train",
        metadata={"help": "Dataset split"}
    )
    gen_data_path: str = field(
        default="data/dpo-mix-7k/gemma_2b_sft.csv",
        metadata={"help": "Path to save the generated data CSV file"}
    )
    output_path: str = field(
        default="data/dpo-mix-7k/gemma_2b_sft_gpt4-turbo-scored.csv",
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
        if self.gen_data_path == '':
            raise ValueError("Please provide a valid path for --gen_data_path")
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


def get_model_single_judgment(
    judge_model, client: openai.Client, task: UltraFeedbackTask,
    result_list: list, full_id: str,
    generation: str, sides: list, prompt: Prompt
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
        print(f"Token usage: {response.usage}")

        gen_idx = sides.index("gen")
        chosen_idx = sides.index("chosen")

        ratings = task.parse_output(response.choices[0].message.content)
        chosen_rating = ratings[chosen_idx]
        gen_rating = ratings[gen_idx]
    except Exception as e:
        print(f"Error in getting model response: {e}")
        chosen_rating = {'rating': -100}
        gen_rating = {'rating': -100}

    result_list.append({
        "full_id": full_id,
        "generation": generation,
        "chosen_rating": chosen_rating,
        "gen_rating": gen_rating,
    })
    return


def prepare_dataset(args: JudgmentArguments, raw_dataset: DatasetDict):
    split_name = args.dset_split
    raw_dset = raw_dataset[split_name]

    ## 1. join the generated data and the raw data
    raw_dset_df = raw_dset.to_pandas()
    gen_data_df = pd.read_csv(args.gen_data_path)
    raw_dset_df.index = raw_dset_df["full_id"].values
    gen_data_df.index = gen_data_df["full_id"].values
    raw_dset_df = raw_dset_df.drop(columns=["full_id"])
    joined_df = gen_data_df.join(
        raw_dset_df, on="full_id", how="inner"
    )

    random.seed(42)

    reformatted_data = []
    for _, sample in tqdm(joined_df.iterrows(), desc="Formatting scoring data", total=len(joined_df)):
        full_id = sample["full_id"]
        chosen_conv = sample["chosen"]
        gen_texts = sample["gen_answers"]

        chosen_prompt, chosen_answer = format_conversation(chosen_conv)
        reformatted_data.append({
            "full_id": full_id,
            "input": chosen_prompt,
            "chosen_answer": chosen_answer,
            "generations": eval(gen_texts),  # this was a string version of list
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
            input_text = sample['input']
            chosen_response = sample['chosen_answer']
            generations = sample['generations']

            est_length = len(tokenizer.encode(input_text + generations[0]))
            if est_length > 4096:
                print(f"Skipping sample {full_id} due to length")
                continue
            
            ## or judge them one by one
            for _, gen in enumerate(generations):
                resps = [chosen_response, gen]
                sides = ["chosen", "gen"]
                if random.random() > 0.5:
                    resps = resps[::-1]
                    sides = sides[::-1]  # flip the order of the responses

                prompt = task.generate_prompt(
                    input=input_text,
                    generations=resps
                )

                future = executor.submit(
                    get_model_single_judgment,
                    args.judge_model,
                    client,
                    task,
                    RESULT_LIST,
                    full_id,
                    gen,
                    sides,
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
                tmp_save_path = args.output_path.replace(".csv", "_tmp.pkl")
                with open(tmp_save_path, "wb") as f:
                    pickle.dump(RESULT_LIST, f)
                
                tmp_df = pd.DataFrame(RESULT_LIST)
                tmp_df.to_csv(args.output_path, index=False)

    tmp_df = pd.DataFrame(RESULT_LIST)
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