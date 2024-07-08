import concurrent.futures
import openai
import hashlib
import pandas as pd
from datasets import load_dataset
from copy import deepcopy
from tqdm.auto import tqdm


def add_info(data_dict: dict):
    response = data_dict["response"]
    the_answer_is_step = response.find("The answer is:")
    if the_answer_is_step == -1:
        data_dict["answer"] = None
        print(f"Could not find the answer in the following response: {response}")
    else:
        the_answer_is_step += len("The answer is:")
        solution = response[the_answer_is_step:]
        data_dict["answer"] = solution.strip()

    ## add id
    query = data_dict["query"]
    full_encoded = f"{query} {response}"
    full_encoded_id = hashlib.sha256(full_encoded.encode("utf-8")).hexdigest()
    data_dict['full_id'] = full_encoded_id
    return data_dict


def get_model_answers(
    client: openai.Client,
    result_list: list,
    sample: dict,
    prompt_messages: list,
    temperature: float = 0.7,
    max_tokens: int = 512,
    n: int = 3
):
    answers = set()

    messages = deepcopy(prompt_messages)
    messages.append({"role": "user", "content": sample["query"]})
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


def main():
    save_file = "data/metamath/gemma_2b_sft.csv"

    client = openai.Client(
        base_url="http://127.0.0.1:6624/v1", api_key="EMPTY"
    )

    math_dset = load_dataset("meta-math/MetaMathQA", split="train")
    math_dset = math_dset.map(
        add_info,
        num_proc=8,
        keep_in_memory=True,
        desc="Adding info to the dataset",
    )

    prompt_messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Solve the following math questions and end your response with 'The answer is: (your answer)'"},
    ]
    random_examples = [
        math_dset[0], math_dset[1], math_dset[2]
    ]
    for sample in random_examples:
        prompt_messages.append({"role": "user", "content": sample["query"]})
        prompt_messages.append({"role": "assistant", "content": sample["response"]})


    RESULT_LIST = []

    n_answers = 1
    n_parallel = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = []
        for idx, sample in enumerate(math_dset):
            if idx in [0, 1, 2]:
                continue
            future = executor.submit(
                get_model_answers,
                client,
                RESULT_LIST,
                sample,
                prompt_messages,
                0.0,
                512,
                n_answers,
            )
            futures.append(future)
        
        num_completed = 0
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Getting model answers"
        ):
            future.result()
            num_completed += 1

            if num_completed % 1000 == 0:
                print(f"Completed {num_completed} samples")
                tmp_df = pd.DataFrame(RESULT_LIST)
                tmp_df.to_csv(save_file, index=False)

    tmp_df = pd.DataFrame(RESULT_LIST)
    tmp_df.to_csv(save_file, index=False)
    return


if __name__ == "__main__":
    main()