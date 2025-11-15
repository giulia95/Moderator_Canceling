import argparse
import json

from pathlib import Path

import numpy as np
import torch

from moderation.generation.utils import load_qa_dataset
from moderation.utils import compute_all_metrics


# Example: map_labels({"S2": True, "S1": False}, ["S1", "S2", "S3"]) -> [0, 1, 0]
def map_labels(category: dict[str, bool], labels: list[str]) -> torch.Tensor:
    """Map a list of labels to a multi-hot vector."""
    return torch.tensor([int(category[label]) for label in labels])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--split", type=str, required=False)
    parser.add_argument("--config-name", type=str, required=False)
    parser.add_argument("--predictions-filepath", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)

    return parser.parse_args()


def compute_metrics(args: argparse.Namespace) -> None:
    # load the dataset and labels
    dataset, labels = load_qa_dataset(args.dataset_name, args.split, args.config_name)
    dataset = dataset.map(lambda x: {"label": map_labels(x["category"], labels)})
    id2labels = dict(enumerate(labels))

    # load the predictions
    predictions = torch.load(args.predictions_filepath)

    all_preds = np.asarray(predictions)
    all_labels = np.asarray(dataset["label"])

    metrics_dir = Path(args.output_dir)
    metrics_dir.mkdir(exist_ok=True)
    base_filename = metrics_dir / f"{args.model_name.replace('/', '__')}"

    if args.config_name == "multilingual":
        eng_idx = len(dataset) // 2
        it_preds = all_preds[:eng_idx]
        it_labels = all_labels[:eng_idx]

        en_preds = all_preds[eng_idx:]
        en_labels = all_labels[eng_idx:]

        # run on italian dataset
        it_metrics = compute_all_metrics(it_preds, it_labels, id2labels=id2labels, thresholds=args.thresholds)

        with Path(f"{base_filename}_italian.json").open("w") as f:
            json.dump(it_metrics, f, indent=2)

        # run on english dataset
        en_metrics = compute_all_metrics(en_preds, en_labels, id2labels=id2labels, thresholds=args.thresholds)

        with Path(f"{base_filename}_english.json").open("w") as f:
            json.dump(en_metrics, f, indent=2)

    # run on both datasets
    metrics = compute_all_metrics(all_preds, all_labels, id2labels=id2labels, thresholds=args.thresholds)

    with Path(f"{base_filename}_all.json").open("w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    compute_metrics(args)
