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
    dataset_name: str, split: str, config_name: str, label_processing: str, label_column: str | None = None, problem_type: str = "multi_label_classification"
) -> tuple["Dataset", list[str], dict[str, int], dict[int, str]]: ...


"""
def cade_preprocessing(df_dataset):

  # Define the grouping columns
  grouping_cols = ['text', 'target', 'title']

  # Check if all grouping columns exist in df_dataset
  missing_cols = [col for col in grouping_cols if col not in df_dataset.columns]
  if missing_cols:
      raise RuntimeError(f"Missing grouping columns in df_dataset: {missing_cols}")

  # Drop rows where 'acceptability' is NaN before grouping
  initial_df_rows = len(df_dataset)
  df_dataset_cleaned = df_dataset.dropna(subset=['acceptability']).copy()
  print(f"Dropped {initial_df_rows - len(df_dataset_cleaned)} rows with NaN in 'acceptability' before grouping.")

  # Group by 'text', 'target', 'title' and aggregate 'acceptability' into a list
  grouped_df = df_dataset_cleaned.groupby(grouping_cols)['acceptability'].apply(list).reset_index()

  # Concatenate the acceptability lists into a single string for each group
  grouped_df['concatenated_acceptability'] = grouped_df['acceptability'].apply(lambda x: ', '.join(map(str, x)))

  # Calculate the average acceptability
  grouped_df['average_acceptability'] = grouped_df['concatenated_acceptability'].apply(
      lambda x: sum(float(val) for val in x.split(', ')) / len(x.split(', ')) if x else 0.0
  )

  # Drop the intermediate 'acceptability' list column if no longer needed
  grouped_df = grouped_df.drop(columns=['acceptability'])

  return grouped_df
  """
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value

def _process_single_dataset(dataset_obj: Dataset, cluster_filter_mode: str = "all"):
    grouping_cols = ['text', 'target', 'title']

    initial_dataset_rows = len(dataset_obj)
    dataset_cleaned = dataset_obj.filter(lambda example: example['acceptability'] is not None)
    print(f"Dropped {initial_dataset_rows - len(dataset_cleaned)} examples with None in 'acceptability' before grouping for this split.")

    current_rows = len(dataset_cleaned)
    if cluster_filter_mode == "cluster 0":
        # Drop instances where cluster is 1 (keep only cluster 0)
        dataset_cleaned = dataset_cleaned.filter(lambda example: example['cluster'] != 1)
        print(f"Dropped {current_rows - len(dataset_cleaned)} examples with cluster 1 before grouping for this split (keeping cluster 0)."
        )
    elif cluster_filter_mode == "cluster 1":
        # Drop instances where cluster is 0 (keep only cluster 1)
        dataset_cleaned = dataset_cleaned.filter(lambda example: example['cluster'] != 0)
        print(f"Dropped {current_rows - len(dataset_cleaned)} examples with cluster 0 before grouping for this split (keeping cluster 1).")
    elif cluster_filter_mode == "all":
        print("No cluster-based dropping applied as 'cluster_filter_mode' is 'all'.")
    else:
        raise ValueError(f"Invalid cluster_filter_mode: {cluster_filter_mode}. Expected 'all', 'cluster 0', or 'cluster 1'.")

    available_features = dataset_cleaned.features.keys()
    missing_cols = [col for col in grouping_cols if col not in available_features]
    if missing_cols:
        raise RuntimeError(f"Missing grouping columns in dataset: {missing_cols}")

    grouped_data_dict = {}

    for i in range(len(dataset_cleaned)):
        example = dataset_cleaned[i]
        key = (example['text'], example['target'], example['title'])

        if key not in grouped_data_dict:
            grouped_data_dict[key] = {col: example[col] for col in available_features if col != 'acceptability'}
            grouped_data_dict[key]['acceptability_values'] = [] 
        
        grouped_data_dict[key]['acceptability_values'].append(example['acceptability'])

    final_records = []
    for key_tuple, data in grouped_data_dict.items():
        acceptability_list = data['acceptability_values']
        concatenated_acceptability = ', '.join(map(str, acceptability_list))
        average_acceptability = sum(acceptability_list) / len(acceptability_list) if acceptability_list else 0.0

        new_record = {
            col: data[col] for col in data if col != 'acceptability_values'
        }
        new_record['concatenated_acceptability'] = concatenated_acceptability
        new_record['average_acceptability'] = average_acceptability
        
        final_records.append(new_record)

    original_features = dataset_obj.features.copy()
    
    if 'acceptability' in original_features:
        del original_features['acceptability']
        
    original_features['concatenated_acceptability'] = Value('string')
    original_features['average_acceptability'] = Value('float')

    grouped_dataset_result = Dataset.from_list(final_records, features=original_features)

    return grouped_dataset_result

def cade_preprocessing_dataset(dataset_dict_obj: DatasetDict, cluster_filter_mode: str = "all"):
    if not isinstance(dataset_dict_obj, DatasetDict):
        raise TypeError("Input must be a datasets.DatasetDict object.")

    processed_splits = {}
    for split_name, dataset in dataset_dict_obj.items():
        print(f"Processing split: {split_name}")
        processed_splits[split_name] = _process_single_dataset(dataset, cluster_filter_mode)

    return DatasetDict(processed_splits)

def load_qa_dataset(
    dataset_name: str,
    split: str | None = None,
    config_name: str | None = None,
    problem_type: str = "multi_label_classification",
    label_processing: str = "",
    label_column: str = "label",
    cluster_filter_mode: str = "all",
) -> tuple["Dataset | DatasetDict", list[str], dict[str, int], dict[int, str]]:
    dataset = load_dataset(dataset_name, name=config_name, split=split)
    print(split)
    # group "per text", derive "average_acceptability"
    dataset = cade_preprocessing_dataset(dataset, cluster_filter_mode)

    dataset = dataset.filter(lambda x: x[label_column] is not None)

    if problem_type == "single_label_classification":

        """
        def _format_bin_labels(example: dict[str, Any]) -> dict[str, torch.Tensor | int]:
            return {"label": 0 if example["acceptability"] <= 2 else 1} #{"label": 0 if example["is_safe"] else 1}
        """
        def _format_bin_labels(example: dict[str, Any], label_processing: str, label_column: str) -> dict[str, torch.Tensor | int]:
            # Ensure the condition and column name are valid
            if label_column not in example:
                raise ValueError(f"Column '{label_column}' not found in the example.")
            
            # Evaluate the condition against the specified column
            try:
                condition = f"example['{label_column}'] {label_processing}"
                label = 1 if eval(condition) else 0
            except Exception as e:
                raise ValueError(f"Failed to evaluate condition '{label_processing}' for column '{label_column}': {e}")
            
            return {"label": label}

        dataset = cast("Dataset", dataset)
        dataset = dataset.map(
            lambda example: _format_bin_labels(
                example,
                label_processing=label_processing,
                label_column=label_column,
            ),
            desc="Formatting labels to safe/unsafe (0/1)",
        )
        """
        dataset = cast("Dataset", dataset)
        dataset = dataset.map(_format_bin_labels, desc="Formatting labels to safe/unsafe (0/1)")
        """
        #TODO: Qui sistemare per fare funzionare su 4 labels
        labels = ["safe", "unsafe"]
        id2label = {0: "safe", 1: "unsafe"}
        label2id = {"safe": 0, "unsafe": 1}
    elif problem_type == "single_label_multiclass_classification":
        print("not yet implemented XD")

    else:
        categories: list[dict[str, bool]]

        if isinstance(dataset, DatasetDict):
            splits = list(dataset.keys())
            categories = dataset[splits[0]]["label"]
        else:
            dataset = cast("Dataset", dataset)
            categories = dataset["label"]

        labels = list(categories[0].keys())
        id2label = dict(enumerate(labels))
        label2id = {label: i for i, label in id2label.items()}

        def _format_labels(example: dict[str, Any]) -> dict[str, torch.Tensor]:
            multi_hot_vec = torch.zeros(len(label2id), dtype=torch.float32)

            for name, is_present in example["label"].items():
                if is_present is True:
                    multi_hot_vec[label2id[name]] = 1.0

            return {"label": multi_hot_vec}

        dataset = dataset.map(_format_labels, desc="Formatting labels to multi-hot")

    return dataset, labels, label2id, id2label

""" ORIGINAL
def format_prompts(
    example: dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    template: str | None = None,
) -> dict[str, str]:
    # CADE has only one field: "text"
    if "text" not in example:
        raise ValueError("The dataset must contain a 'text' column.")

    raw_text = example["text"]

    # 1. If user provides a template, apply it
    #    e.g. template="### Instruction:\n{text}\n###"
    if template is not None:
        text = template.format(text=raw_text)

    # 2. If tokenizer has a chat template, wrap the text as a user message
    elif tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": raw_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        text = str(text)

    # 3. Otherwise return the text unchanged
    else:
        text = raw_text
    return {"text": text}
"""

    # ORIGINAL + TITLE
def format_prompts(
    example: dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    template: str | None = None,
) -> dict[str, str]:
    # CADE has only one field: "text"
    if "text" not in example:
        raise ValueError("The dataset must contain a 'text' column.")

    raw_text = example["text"]
    #raw_title = example["title"]

    # 1. If user provides a template, apply it
    #    e.g. template="### Instruction:\n{text}\n###"
    if template is not None:
        text = template.format(text=raw_text)

    # 2. If tokenizer has a chat template, wrap the text as a user message
    elif tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": raw_text}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        text = str(text)

    # 3. Otherwise return the text unchanged
    else:
        #text = raw_title + '. ' + raw_text
        text = raw_text
    #print(text)
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

def compute_metrics(
    id2labels: dict[int, str], problem_type: str = "multi_label_classification"
) -> Callable[["EvalPrediction"], dict]:
    def compute_metrics(pred: "EvalPrediction") -> dict:
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        labels = pred.label_ids[0] if isinstance(pred.label_ids, tuple) else pred.label_ids

        if problem_type == "single_label_classification":
            preds = torch.softmax(torch.tensor(preds), dim=-1).numpy()

            return compute_all_metrics(preds, labels, id2labels=None, thresholds=None)
        preds = torch.sigmoid(torch.tensor(preds)).numpy()

        return compute_all_metrics(preds, labels)
    return compute_metrics


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
