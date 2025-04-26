import logging
import os
import sys

from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import transformers

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.hf_argparser import HfArgumentParser

from arguments import DataArguments, ModelArguments
from utils import compute_metrics, format_prompts, load_qa_dataset, tokenize


if TYPE_CHECKING:
    from transformers import TrainerCallback

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


def run(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    callbacks: "list[TrainerCallback] | None" = None,
) -> None:
    set_seed(training_args.seed)

    # Load dataset
    dataset, labels, label2id, id2label = load_qa_dataset(data_args.dataset_name, config_name=data_args.config_name)
    logger.info("Dataset loaded %s", dataset)
    logger.info("Labels: %s", labels)
    logger.info("Label2id: %s", label2id)
    logger.info("Id2label: %s", id2label)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=model_args.trust_remote_code,
    )

    if "Llama-3" in model_args.model_name_or_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"  # noqa: S105
        tokenizer.pad_token_id = 128004

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task="text-classification",
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    if getattr(model_args, "cls_dropout", None):
        config.cls_dropout = model_args.cls_dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        device_map=model_args.device_map,
        trust_remote_code=model_args.trust_remote_code,
    )

    max_seq_length = data_args.max_seq_length

    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length (%d) is larger than the maximum length for the model (%d). Using max_seq_length=%d.",
            max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    # Prepare dataset and splits
    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            partial(format_prompts, tokenizer=tokenizer, template=data_args.template),
            desc="Formatting prompts using template",
        )

        dataset = dataset.map(
            partial(
                tokenize,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                padding=data_args.padding,
                add_eos_token=model_args.add_eos_token,
            ),
            batched=True,
            desc="Tokenize dataset",
        )

        data_partition = dataset[data_args.train_split].train_test_split(
            test_size=data_args.eval_split_ratio,
            seed=training_args.seed,
            shuffle=True,
        )

    train_dataset = data_partition["train"]
    eval_dataset = data_partition["test"]
    test_dataset = dataset[data_args.test_split]

    logger.info("Train: %d", len(train_dataset))
    logger.info("Eval: %d", len(eval_dataset))
    logger.info("Test: %d", len(test_dataset))

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16 or training_args.bf16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    if callbacks is not None:
        for cb in callbacks:
            trainer.add_callback(cb)

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    # Training
    train_result = trainer.train()
    train_metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # Evaluation
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Test
    test_results = trainer.predict(test_dataset)  # type: ignore
    test_metrics = test_results.metrics
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # Push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["multi-label", "question-answering", "text-classification"],
        "tasks": "text-classification",
        "dataset": data_args.dataset_name,
        "dataset_tags": "beavertails",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def setup_logging(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> None:
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


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(Path(sys.argv[1]).resolve())
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(model_args, data_args, training_args)
    run(model_args, data_args, training_args)
