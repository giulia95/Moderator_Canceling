import argparse

from functools import partial
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, AutoConfig

from moderation.classification.utils import format_prompts, load_qa_dataset


def compute_predictions(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # load the dataset
    #dataset, labels, _, id2label = load_qa_dataset(args.dataset_name, args.split, args.config_name, args.problem_type,
    
    print(args)

    dataset, labels, label2id, id2label = load_qa_dataset(
        args.dataset_name,
        config_name=args.config_name,
        problem_type=args.problem_type,
        label_processing = args.label_processing,
        label_column = args.label_column,
        cluster_filter_mode = args.cluster_filter_mode,
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        finetuning_task="text-classification",
        problem_type=args.problem_type,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # load the tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=args.dtype,
        device_map=args.device,
        trust_remote_code=True,
        config=config,
    )
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # check if the dataset labels are the same as the model's labels
    if set(labels) != set(model.config.id2label.values()):
        msg = "Labels do not match between dataset and model"
        raise ValueError(msg)

    if id2label != model.config.id2label:
        msg = "Label2id does not match between dataset and model"
        raise ValueError(msg)

    train_test = dataset['train'].train_test_split(
            test_size=0.2,
            seed=42,
            shuffle=True,
        )
    dataset = train_test["test"] # only the test part

    # format prompts
    dataset = dataset.map(
        partial(format_prompts, tokenizer=tokenizer, template=None),
        desc="Formatting prompts using template",
    )

    # compute predictions
    predictions = torch.zeros(len(dataset), len(labels), dtype=torch.float64)

    with torch.inference_mode():
        for i, example in enumerate(tqdm(dataset.to_iterable_dataset(), total=len(dataset))):
            inputs = tokenizer(
                example["text"],
                truncation=True,
                padding=False,
                max_length=args.max_length,
                return_tensors="pt",
            )
            outputs = model(**inputs.to(model.device))
            logits = outputs.logits[0].detach()

            predictions[i, :] = (
                torch.sigmoid(logits)
                if args.problem_type == "multi_label_classification"
                else torch.softmax(logits, dim=-1)
            )

    predictions = predictions.detach().cpu().numpy()

    # save predictions
    filepath = Path(args.out_filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, predictions)

    # ---- Save CSV with predictions ----
    # Safety check
    assert predictions.shape[0] == len(dataset), \
        "Predictions and test dataset size mismatch"
    assert predictions.shape[1] == 2, \
        "Expected logits for exactly 2 classes"

    # Dataset -> DataFrame
    df_test = dataset.to_pandas()

    # Add one column per class
    model_name = args.model_name
    df_test[f"{model_name}_logit_class_0"] = predictions[:, 0]
    df_test[f"{model_name}_logit_class_1"] = predictions[:, 1]

    # Save CSV
    csv_path = filepath.with_suffix(".csv")
    df_test.to_csv(csv_path, index=False)