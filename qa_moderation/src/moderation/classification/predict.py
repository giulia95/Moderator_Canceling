import argparse

from functools import partial
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from moderation.classification.utils import format_prompts, load_qa_dataset


def compute_predictions(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # load the dataset
    dataset, labels, _, id2label = load_qa_dataset(args.dataset_name, args.split, args.config_name, args.problem_type)

    # load the tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=args.dtype,
        device_map=args.device,
        trust_remote_code=True,
        problem_type=args.problem_type,
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
