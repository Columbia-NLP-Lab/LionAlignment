from typing import Dict, Optional
from dataclasses import dataclass, field, asdict
from datasets import DatasetDict, concatenate_datasets, load_dataset
from src.trainers.configs import H4ArgumentParser
from src.utils.utils import create_dir_if_not_exists
from src.dataloads.data import apply_chat_template
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, TrainingArguments, Trainer,
    set_seed
)
import evaluate
import random
import numpy as np
import json
import yaml
import os
import pandas as pd
import hashlib


@dataclass
class DsetPredictionArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_to_test: Dict[str, str] = field(
        metadata={"help": ("dictionary of dset: [splits] to use for training and testing")},
    )
    fullid_files: Dict[str, Optional[Dict]] = field(
        metadata={"help": ("path of full id file: label for that file")},
    )
    per_dataset_size: Dict[str, float] = field(
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    chat_template: str = field(
        metadata={"help": ("The chat template to use for the dataset.")},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "The maximum length of the input features."},
    )
    content_to_predict: str = field(
        default="prompt",
        metadata={"help": "The field to predict in the dataset. Choices: ['prompt', 'response']"},
    )

    ## model related
    model_name_or_path: str = field(
        default="jinaai/jina-embeddings-v2-base-en",
        metadata={"help": "The HF model path or model checkpoint for weights initialization."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for Adam."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "The batch size per GPU/TPU core/CPU for evaluation."},
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "The number of epochs for training."},
    )
    output_dir: str = field(
        default="model_checkpoints_coffee/debug",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    save_strategy: str = field(
        default="no",
        metadata={"help": "Defaults to no since experiment is fast."},
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "The number of steps to save the model."},
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "The total number of checkpoints to save."},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "The number of steps to evaluate the model."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"},
    )

    def __post_init__(self):
        assert len(self.dataset_to_test) == 1, "Only one datasets can be passed in here"
        assert self.content_to_predict in ["prompt", "response"], "content_to_predict must be either 'prompt' or 'response'"
        return


def _format_content(data_dict: dict, content_to_predict: str):
    if content_to_predict == "prompt":
        return data_dict["prompt"]
    elif content_to_predict == "response":
        return data_dict["messages"][1]['content']
    else:
        raise ValueError("content_to_predict must be either 'prompt' or 'response'")
    return


def _default_reformatting_fn(data_dict: dict, dset_label: str, content_to_predict: str):
    text = _format_content(data_dict, content_to_predict)
    return {
        "text": text,
        "label": dset_label,
    }


def _keep_if_found(data_dict: dict, full_id_df: pd.DataFrame):
    prompt_id = data_dict['prompt_id']
    if prompt_id not in full_id_df.index:
        return False
    return True


## assumes there is a 'prompt' field
def get_datasets_for_analysis(dset_prediction_args: DsetPredictionArguments, label2id: dict):
    datasets_config = dset_prediction_args.dataset_to_test
    splits = dset_prediction_args.per_dataset_size
    seed = dset_prediction_args.seed

    raw_datasets = DatasetDict()

    raw_train_datasets = []
    raw_val_datasets = []
    raw_test_datasets = []

    for ds, split in datasets_config.items():
        break

    raw_dset = load_dataset(ds, split=split)
    print(f"Loaded {ds} dataset: {len(raw_dset)}")

    for full_id_file, label_name in dset_prediction_args.fullid_files.items():
        full_id_df = pd.read_csv(full_id_file)
        if 'idx' in full_id_df.columns:
            print("Setting index to idx")
            full_id_df = full_id_df.set_index('idx')
        else:
            full_id_df = full_id_df.set_index('prompt_id')
        
        dataset = raw_dset.filter(
            _keep_if_found,
            fn_kwargs={
                "full_id_df": full_id_df,
            },
            num_proc=8,
            keep_in_memory=True,  # otherwise it creates a lot of files in cache
            desc=f"Filtering {ds} dataset",
        )

        print(f"Using {len(dataset)} after filtering")

        dataset = dataset.map(
            _default_reformatting_fn,
            fn_kwargs={
                "dset_label": label2id[label_name],
                "content_to_predict": dset_prediction_args.content_to_predict,
            },
            num_proc=8,
            keep_in_memory=True,  # otherwise it creates a lot of files in cache
            desc=f"Reformatting {ds} dataset",
        )

        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select_columns(['text', 'label'])

        # select train, val, test
        train_end_idx = splits["train"]
        val_end_idx = train_end_idx + splits["validation"]
        test_end_idx = val_end_idx + splits["test"]
        train_subset = dataset.select(range(train_end_idx))
        val_subset = dataset.select(range(train_end_idx, val_end_idx))
        test_subset = dataset.select(range(val_end_idx, test_end_idx))

        raw_train_datasets.append(train_subset)
        raw_val_datasets.append(val_subset)
        raw_test_datasets.append(test_subset)

    raw_datasets["train"] = concatenate_datasets(raw_train_datasets).shuffle(seed=seed)
    raw_datasets["val"] = concatenate_datasets(raw_val_datasets).shuffle(seed=seed)
    raw_datasets["test"] = concatenate_datasets(raw_test_datasets)
    return raw_datasets


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def tokenize_text(samples, tokenizer, **tokenizer_args):
    return tokenizer(samples["text"], **tokenizer_args)


def main():
    parser = H4ArgumentParser(DsetPredictionArguments)
    dset_prediction_args: DsetPredictionArguments = parser.parse()

    # Set seed for reproducibility
    set_seed(dset_prediction_args.seed)
    
    print(f"DsetPredictionArguments {dset_prediction_args}")

    ### save config
    create_dir_if_not_exists(dset_prediction_args.output_dir)
    config_path = os.path.join(dset_prediction_args.output_dir, "config.yaml")
    with open(config_path, "w") as fwrite:
        yaml.dump(asdict(dset_prediction_args), fwrite, default_flow_style=False)
    
    ### prepare datasets
    model_name = dset_prediction_args.model_name_or_path
    label_names = dset_prediction_args.fullid_files.values()

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # here we switch back to using prompt_id to avoid tokenizer formatting issues
    datasets = get_datasets_for_analysis(dset_prediction_args, label2id)
    tokenized_datasets = datasets.map(
        tokenize_text,
        fn_kwargs={"tokenizer": tokenizer, "truncation": True, "max_length": dset_prediction_args.max_length},
        num_proc=8,
        desc="Applying tokenization",
    )

    #### print a few examples
    print("Log a few random samples from the processed training set\n")
    for index in random.sample(range(len(datasets["train"])), 3):
        print(f"Sample {index} of the processed training set:")
        print(datasets['train'][index]['text'])
        print("Label:", id2label[datasets['train'][index]['label']])
        print()
    print("Log a few random samples from the processed test set\n")
    for index in random.sample(range(len(datasets["test"])), 3):
        print(f"Sample {index} of the processed test set:")
        print(datasets['test'][index]['text'])
        print("Label:", id2label[datasets['test'][index]['label']])
        print()
    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
        max_length=dset_prediction_args.max_length,
    )

    ### train model
    seq_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label), id2label=id2label, label2id=label2id,
        trust_remote_code=True,
    )

    training_args = TrainingArguments(
        output_dir=dset_prediction_args.output_dir,
        learning_rate=dset_prediction_args.learning_rate,
        per_device_train_batch_size=dset_prediction_args.per_device_train_batch_size,
        per_device_eval_batch_size=dset_prediction_args.per_device_eval_batch_size,
        num_train_epochs=dset_prediction_args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=dset_prediction_args.eval_steps,
        save_strategy=dset_prediction_args.save_strategy,
        save_steps=dset_prediction_args.save_steps,
        save_total_limit=dset_prediction_args.save_total_limit,
        report_to="none",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=seq_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    ### test
    pred_output = trainer.predict(tokenized_datasets["test"])
    print('Performance')
    print(json.dumps(pred_output.metrics, indent=2))

    results_path = os.path.join(dset_prediction_args.output_dir, "results.json")
    with open(results_path, "w") as fwrite:
        input_texts = datasets["test"]['text']
        json.dump(
            {
                "predictions": pred_output.predictions.tolist(),
                "input_texts": input_texts,
                "labels": pred_output.label_ids.tolist(),
                "id2label": id2label,
                "metrics": pred_output.metrics,
            }, fwrite,
            indent=2,
        )
    return

if __name__ == "__main__":
    main()