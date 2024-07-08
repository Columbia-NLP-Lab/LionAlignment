from typing import Dict
from dataclasses import dataclass, field
from datasets import DatasetDict, concatenate_datasets, load_dataset
from src.trainers.configs import H4ArgumentParser
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, TrainingArguments, Trainer,
    set_seed
)
import evaluate
import numpy as np
import json
import os
import random


@dataclass
class DsetPredictionArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_to_test: Dict[str, str] = field(
        metadata={"help": ("dummy test, use subset one dataset and see if there is a difference")},
    )
    per_dataset_size: Dict[str, float] = field(
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
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
    eval_steps: int = field(
        default=100,
        metadata={"help": "The number of steps to evaluate the model."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "The seed for reproducibility."},
    )

    def __post_init__(self):
        assert len(self.dataset_to_test) == 1, "Only one dataset is allowed for dummy testing"
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


def _random_label_reformatting_fn(data_dict: dict, ids: list, content_to_predict: str):
    text = _format_content(data_dict, content_to_predict)
    return {
        "text": text,
        "label": random.choice(ids),
    }


## assumes there is a 'prompt' field
def get_datasets(datasets_config, splits, content_to_predict: str, id2label: dict, seed: int):
    raw_datasets = DatasetDict()

    raw_train_datasets = []
    raw_val_datasets = []
    raw_test_datasets = []

    for ds, split in datasets_config.items():
        raw_dset = load_dataset(ds, split=split)
        dataset = raw_dset.map(
            _random_label_reformatting_fn,
            fn_kwargs={
                "ids": list(id2label.keys()),
                "content_to_predict": content_to_predict,
            },
            num_proc=8,
            desc=f"Reformatting {ds} dataset",
        )
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select_columns(['text', 'label'])

        # select train, val, test
        train_end_idx = splits["train"] * 2
        val_end_idx = train_end_idx + splits["validation"] * 2
        test_end_idx = val_end_idx + splits["test"] * 2
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
    
    model_name = dset_prediction_args.model_name_or_path
    datasets_config = dset_prediction_args.dataset_to_test
    splits = dset_prediction_args.per_dataset_size
    content_to_predict = dset_prediction_args.content_to_predict

    dset_name = list(datasets_config.keys())[0]
    id2label = {0: f'{dset_name}_0', 1: f'{dset_name}_1'}
    label2id = {v: k for k, v in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    datasets = get_datasets(datasets_config, splits, content_to_predict, id2label, seed=dset_prediction_args.seed)
    tokenized_datasets = datasets.map(
        tokenize_text,
        fn_kwargs={"tokenizer": tokenizer, "truncation": True, "max_length": dset_prediction_args.max_length},
        num_proc=8,
        desc="Applying tokenization",
    )

    #### print a few examples    
    print("Log a few random samples from the processed training set")
    for index in random.sample(range(len(datasets["train"])), 4):
        print(f"Sample {index} of the processed training set:")
        print(datasets['train'][index]['text'])
        print("Label:", id2label[datasets['train'][index]['label']])

    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
        max_length=dset_prediction_args.max_length,
    )

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

    pred_output = trainer.predict(tokenized_datasets["test"])
    print('Performance')
    print(json.dumps(pred_output.metrics, indent=2))

    results_path = os.path.join(dset_prediction_args.output_dir, "results.json")
    with open(results_path, "w") as f:
        input_texts = datasets["test"]['text']
        json.dump({
            "predictions": pred_output.predictions.tolist(),
            "input_texts": input_texts,
            "labels": pred_output.label_ids.tolist(),
            "id2label": id2label,
            "metrics": pred_output.metrics,
        }, f)

if __name__ == "__main__":
    main()