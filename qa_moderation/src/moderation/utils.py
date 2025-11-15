import logging
import os
import sys

from importlib.util import find_spec

import datasets
import numpy as np
import transformers

from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from transformers import TrainingArguments
from transformers.hf_argparser import DataClassType


def setup_logging(
    logger: logging.Logger,
    training_args: TrainingArguments,
    model_args: DataClassType,
    data_args: DataClassType,
) -> None:
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        "fp16" if training_args.fp16 else "bf16" if training_args.bf16 else False,
    )

    logger.info("Data arguments %s", data_args)
    logger.info("Model arguments %s", model_args)
    logger.info("Training arguments %s", training_args)

    if training_args.report_to is not None:
        # Setup wandb if installed
        if "wandb" in training_args.report_to and find_spec("wandb") is not None:
            os.environ["WANDB_PROJECT"] = "QA_Beavertails_" + model_args.model_name_or_path.split("/")[-1]
            os.environ["WANDB_LOG"] = "false"
            os.environ["WANDB_WATCH"] = "false"

            if training_args.run_name is not None:
                os.environ["WANDB_NAME"] = training_args.run_name
        else:
            logger.warning("wandb not installed. Install with `pip install wandb` to enable logging to wandb")


def compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    id2labels: dict[int, str] | None = None,
    thresholds: np.ndarray | None = None,
) -> dict:
    if thresholds is not None and id2labels is not None:
        preds = np.asarray([np.where(p > thresholds, 1, 0) for p in preds])
    else:
        preds = np.asarray([np.where(p > 0.5, 1, 0) for p in preds])

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
                    f"{label}/threshold": float(thresholds[idx]) if thresholds is not None else 0.5,
                }
            )

    return metrics


def tune_thresholds(preds: np.ndarray, labels: np.ndarray, id2labels: dict[int, str]) -> np.ndarray:
    thresholds = np.zeros(len(id2labels))

    for idx in id2labels:
        y_true = labels[:, idx]
        y_pred = preds[:, idx]

        precision, recall, thresh = precision_recall_curve(y_true, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_thresh = thresh[np.argmax(f1)]
        thresholds[idx] = best_thresh

    return thresholds
