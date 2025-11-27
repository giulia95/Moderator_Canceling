from typing import TYPE_CHECKING, Any, Callable, cast, overload

import torch

from datasets import DatasetDict, load_dataset

from moderation.utils import compute_all_metrics, tune_thresholds


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import BatchEncoding, EvalPrediction, PreTrainedTokenizer
    from transformers.tokenization_utils import PaddingStrategy


@overload
def load_qa_dataset(  # type: ignore
    dataset_name: str,
    split: None = None,
    config_name: str | None = None,
    problem_type: str = "multi_label_classification",
) -> tuple["DatasetDict", list[str], dict[str, int], dict[int, str]]: ...


@overload
def load_qa_dataset(  # type: ignore
    dataset_name: str, split: str, config_name: str | None = None, problem_type: str = "multi_label_classification"
) -> tuple["Dataset", list[str], dict[str, int], dict[int, str]]: ...


def load_qa_dataset(
    dataset_name: str,
    split: str | None = None,
    config_name: str | None = None,
    problem_type: str = "multi_label_classification",
) -> tuple["Dataset | DatasetDict", list[str], dict[str, int], dict[int, str]]:
    dataset = load_dataset(dataset_name, name=config_name, split=split)

    if problem_type == "single_label_classification":

        def _format_bin_labels(example: dict[str, Any]) -> dict[str, torch.Tensor | int]:
            return {"label": 0 if example["is_safe"] else 1}

        dataset = cast("Dataset", dataset)
        dataset = dataset.map(_format_bin_labels, desc="Formatting labels to safe/unsafe (0/1)")

        labels = ["safe", "unsafe"]
        id2label = {0: "safe", 1: "unsafe"}
        label2id = {"safe": 0, "unsafe": 1}
    else:
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

        def _format_labels(example: dict[str, Any]) -> dict[str, torch.Tensor]:
            multi_hot_vec = torch.zeros(len(label2id), dtype=torch.float32)

            for name, is_present in example["category"].items():
                if is_present is True:
                    multi_hot_vec[label2id[name]] = 1.0

            return {"label": multi_hot_vec}

        dataset = dataset.map(_format_labels, desc="Formatting labels to multi-hot")

    return dataset, labels, label2id, id2label


def format_prompts(
    example: dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    template: str | None = None,
) -> dict[str, str]:
    if "prompt" not in example or "response" not in example:
        msg = "The dataset must contain a 'prompt' and 'response' column."
        raise ValueError(msg)

    if template is not None:
        # apply the given template
        text = template.format(question=example["prompt"], answer=example["response"])
    elif tokenizer.chat_template is not None:
        # chat template for conversational models
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        text = cast("str", text)
    else:
        # default to concatenation for non-conversational models
        text = example["prompt"] + " " + example["response"]

    return {"text": text}


def tokenize(
    examples: dict,
    tokenizer: "PreTrainedTokenizer",
    padding: "bool | str | PaddingStrategy" = "max_length",
    max_length: int | None = None,
    add_eos_token: bool = False,
) -> "BatchEncoding":
    if "text" not in examples:
        msg = "The dataset must contain a 'text' column."
        raise ValueError(msg)

    if tokenizer.eos_token is None or not add_eos_token:
        texts = examples["text"]
    else:
        eos_token = tokenizer.eos_token if isinstance(tokenizer.eos_token, str) else tokenizer.eos_token[0]
        texts = [text + eos_token if not text.endswith(eos_token) else text for text in examples["text"]]

    return tokenizer(texts, padding=padding, max_length=max_length, truncation=True)


def compute_metrics(pred: "EvalPrediction") -> dict:
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids[0] if isinstance(pred.label_ids, tuple) else pred.label_ids

    preds = torch.sigmoid(torch.tensor(preds)).numpy()

    return compute_all_metrics(preds, labels)


def compute_metrics_with_threshold_tuning(
    id2labels: dict[int, str], problem_type: str = "multi_label_classification"
) -> Callable[["EvalPrediction"], dict]:
    def _compute_metrics(pred: "EvalPrediction") -> dict:
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        labels = pred.label_ids[0] if isinstance(pred.label_ids, tuple) else pred.label_ids

        if problem_type == "multi_label_classification":
            preds = torch.sigmoid(torch.tensor(preds)).numpy()
            thresholds = tune_thresholds(preds, labels, id2labels, problem_type)

            return compute_all_metrics(preds, labels, id2labels=id2labels, thresholds=thresholds)

        if problem_type == "single_label_classification":
            preds = torch.softmax(torch.tensor(preds), dim=-1).numpy()

            return compute_all_metrics(preds, labels, id2labels=None, thresholds=None)

        msg = f"Unknown problem type: {problem_type}"
        raise ValueError(msg)

    return _compute_metrics
