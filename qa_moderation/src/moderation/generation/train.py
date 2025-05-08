import logging
import os
import sys
import typing

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from accelerate import PartialState
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import LoraConfig, PeftMixedModel, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from transformers.tokenization_utils import PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from moderation.generation.arguments import DataArguments, ModelArguments
from moderation.generation.utils import format_prompts, load_qa_dataset, update_chat_template
from moderation.utils import setup_logging


if TYPE_CHECKING:
    from transformers import BatchEncoding, TrainerCallback


os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


def run(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: SFTConfig,
    callbacks: "list[TrainerCallback] | None" = None,
) -> None:
    set_seed(training_args.seed)

    # Load dataset
    dataset, labels = load_qa_dataset(data_args.dataset_name, config_name=data_args.config_name)
    logger.info("Dataset loaded %s", dataset)
    logger.info("Labels: %s", labels)

    # Load model and tokenizer
    model = load_base_model(model_args.model_name_or_path, training_args)
    tokenizer = init_tokenizer(model_args.model_name_or_path, include_descriptions=data_args.include_descriptions)

    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Chat template: %s", tokenizer.chat_template)
    logger.info("Model config: %s", model.config)

    max_seq_length = training_args.max_seq_length

    if max_seq_length and max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "The max_seq_length (%d) is larger than the maximum length for the model (%d). Using max_seq_length=%d.",
            max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )

    max_seq_length = min(max_seq_length or tokenizer.model_max_length, tokenizer.model_max_length)

    lora_args = {
        "lora_r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "lora_modules": model_args.lora_modules,
    }
    model = load_peft_model(model, lora_args)

    # Prepare dataset and splits
    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            partial(
                format_prompts,
                model_name_or_path=model_args.model_name_or_path,
                tokenizer=tokenizer,
                categories=labels,
            ),
            desc="Formatting prompts using chat template",
        )

        logger.debug("Dataset after formatting: %s", dataset)
        logger.debug("Example after formatting: %s", dataset[data_args.test_split][0])

        dataset = dataset.map(
            partial(
                tokenize,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                padding=data_args.padding,
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

    # Data collator
    if model_args.completion_only:
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
    else:
        data_collator = None

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.model.print_trainable_parameters()  # type: ignore

    # Callbacks
    if callbacks is not None:
        for callback in callbacks:
            trainer.add_callback(callback)

    # progress_callback = LLMSampleCB(
    #     trainer,
    #     test_dataset,
    #     num_samples=1000,
    #     max_new_tokens=200,
    #     freq=500,
    # )

    # trainer.add_callback(progress_callback)

    # Resume from checkpoint if available
    last_checkpoint = None
    if (
        training_args.output_dir is not None
        and Path(training_args.output_dir).exists()
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    checkpoint = None

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
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
        # "model_name": training_args.run_name,
        # "finetuned_from": model_args.model_name_or_path,
        "tags": ["question-answering", "text-generation"],
        # "tasks": "text-generation",
        "dataset_name": data_args.dataset_name,
        # "dataset_tags": "beavertails",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def init_tokenizer(model_name: str, include_descriptions: bool = True) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.bos_token = "<|begin_of_text|>"  # noqa: S105
    tokenizer.eos_token = "<|eot_id|>"  # noqa: S105
    tokenizer.pad_token = "<|finetune_right_pad_id|>"  # noqa: S105

    update_chat_template(tokenizer, include_descriptions)

    return tokenizer


def load_base_model(model_name: str, training_args: SFTConfig) -> PreTrainedModel:
    model_cls = AutoLigerKernelForCausalLM if training_args.use_liger_kernel else AutoModelForCausalLM

    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": PartialState().process_index},
    )
    model = typing.cast("PreTrainedModel", model)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    return model


def load_peft_model(model: PreTrainedModel, lora_args: dict) -> PeftModel | PeftMixedModel:
    peft_config = LoraConfig(
        r=lora_args["lora_r"],
        lora_alpha=lora_args["lora_alpha"],
        lora_dropout=lora_args["lora_dropout"],
        bias="none",
        target_modules=lora_args["lora_modules"],
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    return get_peft_model(model, peft_config)


def tokenize(
    examples: dict,
    tokenizer: PreTrainedTokenizer,
    padding: "str | bool | PaddingStrategy" = PaddingStrategy.DO_NOT_PAD,
    max_length: int | None = None,
) -> "BatchEncoding":
    return tokenizer(
        examples["text"],
        padding=padding,
        max_length=max_length,
        truncation=True,
        pad_to_multiple_of=8,
        add_special_tokens=False,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))  # type: ignore

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(Path(sys.argv[1]).resolve())
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    setup_logging(logger, training_args, model_args, data_args)
    run(model_args, data_args, training_args)
