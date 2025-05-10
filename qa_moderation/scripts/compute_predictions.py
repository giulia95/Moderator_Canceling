import argparse

from pathlib import Path

from moderation.classification.predict import compute_predictions as run_classification
from moderation.generation.predict import compute_predictions as run_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "generation"], required=True, help="Task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--split", type=str, required=False)
    parser.add_argument("--config-name", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--out-filepath", type=Path, required=True)

    generation_group = parser.add_argument_group("Generation arguments")
    generation_group.add_argument("--adapter-name", type=str, default=None)
    generation_group.add_argument("--max-new-tokens", type=int, default=30)

    return parser.parse_args()


if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # compute predictions depending on task
    if args.task == "generation":
        run_generation(args)
    else:
        run_classification(args)
