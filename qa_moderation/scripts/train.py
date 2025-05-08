import argparse
import logging
import sys

from pathlib import Path

from transformers import HfArgumentParser

from moderation.classification.train import run as run_classification
from moderation.generation.train import run as run_generation
from moderation.utils import setup_logging


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training script for QA moderation")
    parser.add_argument("--task", choices=["classification", "generation"], required=True, help="Training mode")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (yaml)")
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    if args.mode == "classification":
        from transformers import TrainingArguments  # noqa: I001
        from moderation.classification.arguments import DataArguments, ModelArguments

        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
    else:
        from trl import SFTConfig  # noqa: I001
        from moderation.generation.arguments import DataArguments, ModelArguments

        parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))  # type: ignore

    model_args, data_args, training_args = parser.parse_yaml_file(config_path)
    setup_logging(logger, training_args, model_args, data_args)

    if args.mode == "classification":
        run_classification(model_args, data_args, training_args)
    else:
        run_generation(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
