from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
import torch

from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import BatchEncoding, EvalPrediction, PreTrainedTokenizer
    from transformers.tokenization_utils import PaddingStrategy


@overload
def load_qa_dataset(  # type: ignore
    dataset_name: str,
    split: None = None,
    config_name: str | None = None,
    template: str = "Question: {question} Answer: {answer}",
) -> tuple["DatasetDict", list[str], dict[str, int], dict[int, str]]: ...


@overload
def load_qa_dataset(  # type: ignore
    dataset_name: str,
    split: str,
    config_name: str | None = None,
    template: str = "Question: {question} Answer: {answer}",
) -> tuple["Dataset", list[str], dict[str, int], dict[int, str]]: ...


def load_qa_dataset(
    dataset_name: str,
    split: str | None = None,
    config_name: str | None = None,
    template: str = "Question: {question} Answer: {answer}",
) -> tuple["Dataset | DatasetDict", list[str], dict[str, int], dict[int, str]]:
    dataset = load_dataset(dataset_name, name=config_name, split=split)
    categories: list[dict[str, bool]]

    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        categories = dataset[splits[0]]["category"]
    else:
        dataset = cast("Dataset", dataset)
        categories = dataset["category"]

    labels = list(categories[0].keys())
    id2label = dict(enumerate(labels))
    label2id = {label: i for i, label in id2label.items()}

    def format_prompts(example: dict[str, Any]) -> dict[str, str]:
        return {"text": template.format(question=example["prompt"], answer=example["response"])}

    def format_labels(example: dict[str, Any]) -> dict[str, torch.Tensor]:
        multi_hot_vec = torch.zeros(len(label2id), dtype=torch.float32)

        for name, is_present in example["category"].items():
            if is_present is True:
                multi_hot_vec[label2id[name]] = 1.0

        return {"label": multi_hot_vec}

    dataset = dataset.map(format_prompts, desc="Formatting text using template")
    dataset = dataset.map(format_labels, desc="Formatting labels to multi-hot")

    return dataset, labels, label2id, id2label


def tokenize(
    examples: dict,
    tokenizer: "PreTrainedTokenizer",
    padding: "bool | str | PaddingStrategy" = "max_length",
    max_length: int | None = None,
) -> "BatchEncoding":
    if "text" not in examples:
        msg = "The dataset must contain a 'text' column."
        raise ValueError(msg)

    return tokenizer(examples["text"], padding=padding, max_length=max_length, truncation=True)


def compute_metrics(pred: "EvalPrediction") -> dict:
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids[0] if isinstance(pred.label_ids, tuple) else pred.label_ids

    return compute_all_metrics(preds, labels)


def compute_all_metrics(preds: np.ndarray, labels: np.ndarray, id2labels: dict[int, str] | None = None) -> dict:
    # convert logits to multi-hot vectors (same as using sigmoid with 0.5 threshold)
    preds = np.asarray([np.where(p > 0, 1, 0) for p in preds])

    flagged_labels = labels.any(axis=-1)
    flagged_predictions = preds.any(axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_precision": precision_score(labels, preds, average="macro"),
        "macro_recall": recall_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "micro_precision": precision_score(labels, preds, average="micro"),
        "micro_recall": recall_score(labels, preds, average="micro"),
        "flagged/accuracy": accuracy_score(flagged_labels, flagged_predictions),
        "flagged/precision": precision_score(flagged_labels, flagged_predictions),
        "flagged/recall": recall_score(flagged_labels, flagged_predictions),
        "flagged/f1": f1_score(flagged_labels, flagged_predictions),
    }

    if id2labels is not None:
        for idx, label in id2labels.items():
            metrics.update(
                {
                    f"{label}/accuracy": accuracy_score(labels[:, idx], preds[:, idx]),
                    f"{label}/precision": precision_score(labels[:, idx], preds[:, idx]),
                    f"{label}/recall": recall_score(labels[:, idx], preds[:, idx]),
                    f"{label}/f1": f1_score(labels[:, idx], preds[:, idx]),
                }
            )

    return metrics
