import logging
import os
import sys

from functools import partial

import datasets
import transformers

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from utils import compute_metrics, load_qa_dataset, tokenize


os.environ["WANDB_PROJECT"] = "bert-qa-moderation"
os.environ["WANDB_LOG"] = "false"
os.environ["WANDB_WATCH"] = "false"

run_name = "QA-DeBERTa-v3-large"
model_name_or_path = "microsoft/deberta-v3-large"

trust_remote_code = True
device_map = "cuda"

dataset_name = "PKU-Alignment/Beavertails"
train_split = "330k_train"
test_split = "330k_test"
eval_split_ratio = 0.1

max_seq_length = 512
pad_to_max_length = False
padding = "max_length" if pad_to_max_length else False

logger = logging.getLogger(__name__)

training_args = TrainingArguments(
    run_name=run_name,
    output_dir=f"output/{run_name}",
    report_to="wandb",
    log_level="info",
    bf16=True,
    tf32=True,
    ddp_find_unused_parameters=False,
    use_liger_kernel=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    group_by_length=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_ratio=0.03,
    weight_decay=5e-4,
    learning_rate=6e-6,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=0.1,
    save_steps=0.1,
    logging_steps=10,
    metric_for_best_model="eval_f1_macro",
    load_best_model_at_end=True,
    save_total_limit=1,
    seed=42,
)

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

set_seed(training_args.seed)

dataset, labels, id2label, label2id = load_qa_dataset(dataset_name, template="Question: {question} Answer: {answer}")
logger.info("Dataset loaded %s", dataset)
logger.info("Labels: %s", labels)
logger.info("Label2id: %s", label2id)
logger.info("Id2label: %s", id2label)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=trust_remote_code,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    finetuning_task="text-classification",
    problem_type="multi_label_classification",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    device_map=device_map,
    trust_remote_code=trust_remote_code,
)

if max_seq_length > tokenizer.model_max_length:
    logger.warning(
        "The max_seq_length passed (%d) is larger than the maximum length for the model (%d). Using max_seq_length=%d.",
        max_seq_length,
        tokenizer.model_max_length,
        tokenizer.model_max_length,
    )

max_seq_length = min(max_seq_length, tokenizer.model_max_length)

with training_args.main_process_first(desc="dataset map pre-processing"):
    dataset = dataset.map(
        partial(tokenize, tokenizer=tokenizer, max_length=max_seq_length, padding=padding),
        batched=True,
        desc="Tokenize dataset",
    )

    data_partition = dataset[train_split].train_test_split(
        test_size=eval_split_ratio,
        seed=training_args.seed,
        shuffle=True,
    )

train_dataset = data_partition["train"]
eval_dataset = data_partition["test"]
test_dataset = dataset[test_split]

logger.info("Train: %d", len(train_dataset))
logger.info("Eval: %d", len(eval_dataset))
logger.info("Test: %d", len(test_dataset))


if pad_to_max_length:
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

train_result = trainer.train()
metrics = train_result.metrics

trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

kwargs = {
    "finetuned_from": model_name_or_path,
    "tags": ["multi-label", "question-answering", "text-classification"],
    "tasks": "text-classification",
    "dataset": dataset_name,
    "dataset_tags": "beavertails",
}

if training_args.push_to_hub:
    trainer.push_to_hub(kwargs=kwargs)
else:
    trainer.create_model_card(**kwargs)
