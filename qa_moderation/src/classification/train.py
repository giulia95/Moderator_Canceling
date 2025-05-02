import logging
import os
import sys

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from qa_moderation.src.utils import setup_logging
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

from classification.arguments import DataArguments, ModelArguments
from classification.utils import compute_metrics, format_prompts, load_qa_dataset, tokenize


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
    model.config.pad_token_id = tokenizer.pad_token_id

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


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(Path(sys.argv[1]).resolve())
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(logger, training_args, model_args, data_args)
    run(model_args, data_args, training_args)
