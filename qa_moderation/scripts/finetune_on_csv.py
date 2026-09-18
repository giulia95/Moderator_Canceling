import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def infer_label_mapping(series: pd.Series) -> dict[str | int, int]:
    values = [item for item in series.dropna().astype(str).tolist()]
    unique_values = sorted(set(values))

    normalized = {str(value).strip().lower(): value for value in unique_values}

    if set(normalized) == {"0", "1"}:
        return {"0": 0, "1": 1}
    if set(normalized) == {"false", "true"}:
        return {"false": 0, "true": 1}
    if set(normalized) <= {"safe", "unsafe"}:
        return {"safe": 0, "unsafe": 1}
    if set(normalized) <= {"unhealthy", "healthy"}:
        return {"healthy": 0, "unhealthy": 1}

    raise ValueError(
        "Could not infer a binary label mapping automatically. "
        "Please supply --label-mapping, for example '{\"safe\":0,\"unsafe\":1}'."
    )


def prepare_csv_dataset(
    csv_path: str,
    text_column: str = "text",
    label_column: str = "label",
    label_mapping: str | None = None,
) -> Dataset:
    df = pd.read_csv(csv_path)
    print(df)

    missing = [column for column in (text_column, label_column) if column not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    subset = df[[text_column, label_column]].dropna(subset=[text_column, label_column]).copy()
    subset = subset.rename(columns={text_column: "text", label_column: "label"})

    if label_mapping is not None:
        mapping = json.loads(label_mapping)
        subset["label"] = subset["label"].map(mapping)
    else:
        subset["label"] = subset["label"].map(infer_label_mapping(subset["label"]))

    subset = subset[subset["label"].notna()].copy()
    subset["label"] = subset["label"].astype(int)

    return Dataset.from_pandas(subset)


def tokenize_batch(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def load_yaml_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}

    with Path(config_path).open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a dictionary at the top level: {config_path}")

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a local Hugging Face model on a CSV dataset.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Local path to the base model directory.")
    parser.add_argument("--train_csv", type=str, default=None, help="Path to the training CSV file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save the fine-tuned model.")
    parser.add_argument("--text_column", type=str, default=None, help="Text column name in the CSV.")
    parser.add_argument("--label_column", type=str, default=None, help="Label column name in the CSV.")
    parser.add_argument("--label_mapping", type=str, default=None, help='Optional JSON mapping like {"safe": 0, "unsafe": 1}.')
    parser.add_argument("--eval_ratio", type=float, default=None, help="Held-out validation split ratio.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--trust_remote_code", default=None, action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    model_path = config.get("model_path", args.model_path)
    train_csv = config.get("train_csv", args.train_csv)
    output_dir = config.get("output_dir", args.output_dir)
    text_column = config.get("text_column", args.text_column) or "text"
    label_column = config.get("label_column", args.label_column) or "label"
    label_mapping = config.get("label_mapping", args.label_mapping)
    eval_ratio = config.get("eval_ratio", args.eval_ratio)
    seed = config.get("seed", args.seed)
    learning_rate = config.get("learning_rate", args.learning_rate)
    batch_size = config.get("batch_size", args.batch_size)
    epochs = config.get("epochs", args.epochs)
    max_length = config.get("max_length", args.max_length)
    trust_remote_code = config.get("trust_remote_code", args.trust_remote_code)

    if model_path is None or train_csv is None or output_dir is None:
        raise ValueError(
            "Missing required arguments. Provide --model_path, --train_csv, and --output_dir, "
            "or specify them in the YAML config file."
        )

    eval_ratio = 0.1 if eval_ratio is None else float(eval_ratio)
    seed = 42 if seed is None else int(seed)
    learning_rate = 2e-5 if learning_rate is None else float(learning_rate)
    batch_size = 8 if batch_size is None else int(batch_size)
    epochs = 3 if epochs is None else int(epochs)
    max_length = 512 if max_length is None else int(max_length)
    trust_remote_code = bool(trust_remote_code or False)

    dataset = prepare_csv_dataset(
        csv_path=train_csv,
        text_column=text_column,
        label_column=label_column,
        label_mapping=label_mapping,
    )
    print(dataset)

    if len(dataset) < 2:
        raise ValueError("The CSV must contain at least 2 examples for training and validation.")

    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "[PAD]"

    def encode(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = train_ds.map(encode, batched=True)
    eval_ds = eval_ds.map(encode, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        trust_remote_code=trust_remote_code,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
