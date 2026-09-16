import argparse

from functools import partial
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed, AutoConfig, AutoModelForMaskedLM

from moderation.classification.utils import format_prompts, load_qa_dataset


def compute_predictions(args: argparse.Namespace) -> None:
    """
    Main prediction pipeline that generates model predictions with uncertainty estimation.
    
    This function:
    1. Loads a dataset and pre-trained sequence classification model
    2. Splits the training data and uses only the test portion for predictions
    3. Computes prediction probabilities for each example
    4. Calculates pseudo-perplexity scores using a masked language model
    5. Performs calibration to identify uncertain predictions
    6. Saves results as both .npy and .csv files with all metrics
    
    Args:
        args: Namespace containing:
            - seed: Random seed for reproducibility
            - dataset_name: Name of the dataset to load
            - config_name: Dataset configuration
            - problem_type: 'multi_label_classification' or 'single_label_classification'
            - label_processing: Method for processing labels
            - label_column: Name of the label column
            - cluster_filter_mode: Filtering mode for clusters
            - model_name: HuggingFace model identifier
            - dtype: Model data type (e.g., torch.float32)
            - device: Device to run model on ('cuda' or 'cpu')
            - max_length: Maximum sequence length for tokenization
            - out_filepath: Path to save prediction outputs
    
    Returns:
        None. Saves predictions to disk as .npy and .csv files.
    """
    set_seed(args.seed)

    # load the dataset
    #dataset, labels, _, id2label = load_qa_dataset(args.dataset_name, args.split, args.config_name, args.problem_type,
    
    print(args)

    dataset, labels, label2id, id2label = load_qa_dataset(
        args.dataset_name,
        config_name=args.config_name,
        problem_type=args.problem_type,
        label_processing = args.label_processing,
        label_column = args.label_column,
        cluster_filter_mode = args.cluster_filter_mode,
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        finetuning_task="text-classification",
        problem_type=args.problem_type,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # load the tokenizer and model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=args.dtype,
        device_map=args.device,
        trust_remote_code=True,
        config=config,
    )
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # check if the dataset labels are the same as the model's labels
    if set(labels) != set(model.config.id2label.values()):
        msg = "Labels do not match between dataset and model"
        raise ValueError(msg)

    if id2label != model.config.id2label:
        msg = "Label2id does not match between dataset and model"
        raise ValueError(msg)

    train_test = dataset['train'].train_test_split(
            test_size=0.2,
            seed=42,
            shuffle=True,
        )
    dataset = train_test["test"] # only the test part

    # format prompts
    dataset = dataset.map(
        partial(format_prompts, tokenizer=tokenizer, template=None),
        desc="Formatting prompts using template",
    )

    # compute predictions
    predictions = torch.zeros(len(dataset), len(labels), dtype=torch.float64)

    with torch.inference_mode():
        for i, example in enumerate(tqdm(dataset.to_iterable_dataset(), total=len(dataset))):
            inputs = tokenizer(
                example["text"],
                truncation=True,
                padding=False,
                max_length=args.max_length,
                return_tensors="pt",
            )
            outputs = model(**inputs.to(model.device))
            logits = outputs.logits[0].detach()

            predictions[i, :] = (
                torch.sigmoid(logits)
                if args.problem_type == "multi_label_classification"
                else torch.softmax(logits, dim=-1)
            )

    pseudo_ppls, all_token_scores = compute_perplexity_with_tokens(dataset, args.model_name, args.device)

    predictions = predictions.detach().cpu().numpy()

    per_item_loss, is_uncertain = loss_per_item(predictions, dataset['label'])
    
    # save predictions
    filepath = Path(args.out_filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, predictions)

    # ---- Save CSV with predictions ----
    # Safety check
    assert predictions.shape[0] == len(dataset), \
        "Predictions and test dataset size mismatch"
    assert predictions.shape[1] == 2, \
        "Expected logits for exactly 2 classes"

    # Dataset -> DataFrame
    df_test = dataset.to_pandas()

    # Add one column per class
    model_name = args.model_name
    df_test[f"{model_name}_logit_class_0"] = predictions[:, 0]
    df_test[f"{model_name}_logit_class_1"] = predictions[:, 1]

    # calibration
    df_test[f"{model_name}_per_item_loss"] = per_item_loss
    df_test[f"{model_name}_flag_uncertain"] = is_uncertain

    df_test[f"{model_name}_pseudo_perplexity"] = pseudo_ppls
    df_test[f"{model_name}_token_scores"] = [
            json.dumps(ts, ensure_ascii=False)
            for ts in all_token_scores
        ]

    # Save CSV
    csv_path = filepath.with_suffix(".csv")
    df_test.to_csv(csv_path, index=False)


def loss_per_item(predictions, true_labels):
    """
    Compute per-item cross-entropy loss and identify uncertain predictions using conformal prediction.
    
    This function implements a calibration approach based on conformal prediction:
    - Calculates cross-entropy loss for each prediction as a non-conformity score
    - Determines a calibration threshold at the 95th percentile (α=0.05)
    - Flags predictions exceeding this threshold as uncertain
    
    Args:
        predictions: numpy array of shape (n_samples, n_classes) containing predicted probabilities
        true_labels: array-like of shape (n_samples,) containing true class labels
    
    Returns:
        tuple: (per_item_loss, is_uncertain)
            - per_item_loss: numpy array of cross-entropy losses for each sample
            - is_uncertain: boolean array indicating which predictions are uncertain
    """
    # loss per-item
    eps = 1e-12

    #print(predictions)
    #true_labels = np.array(true_labels).astype(int)

    # 1. non-conformity score = cross-entropy per-item
    p_true = predictions[np.arange(len(predictions)), true_labels]
    per_item_loss = -np.log(p_true + eps)

    # 2. soglia di calibrazione (95%)
    alpha = 0.05
    threshold = np.quantile(per_item_loss, 1 - alpha)
    print("======== Calibration Threshold: " + str(threshold) + "======== ")

    # 3. flag di incertezza
    is_uncertain = per_item_loss > threshold
    return per_item_loss, is_uncertain

def compute_perplexity_with_tokens(dataset, model_name, device):
    """
    Compute pseudo-perplexity scores and token-level information for all examples in the dataset.
    
    Loads a masked language model (MLM) version of the specified model and calculates
    pseudo-perplexity for each text in the dataset, along with detailed token-level scores.
    This provides an additional uncertainty metric based on language model likelihood.
    
    Args:
        dataset: HuggingFace Dataset containing examples with 'text' field
        model_name: HuggingFace model identifier for the MLM model
        device: Device to run the model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (pseudo_ppls, all_token_scores)
            - pseudo_ppls: numpy array of pseudo-perplexity scores for each example
            - all_token_scores: list of token-level score dictionaries for each example
    """

    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    #vector to store perplexities
    pseudo_ppls = np.zeros(len(dataset), dtype=np.float64)
    all_token_scores = []

    with torch.inference_mode():
        for i, example in enumerate(tqdm(dataset.to_iterable_dataset(), total=len(dataset))):
            #perplexity
            text = example["text"]
            pseudo_perplexity, token_scores = pseudo_perplexity_with_tokens(text, model, tokenizer, device)
            pseudo_ppls[i] = pseudo_perplexity
            all_token_scores.append(token_scores)
    return pseudo_ppls, all_token_scores

def pseudo_perplexity_with_tokens(text, model, tokenizer, device):
    """
    Calculate pseudo-perplexity for a single text using the masking approach.
    
    For each non-special token in the input:
    1. Mask the token
    2. Use the MLM to predict the masked position
    3. Calculate the log probability of the original token
    4. Collect detailed scores (position, token text, log prob, token perplexity)
    
    The sentence-level pseudo-perplexity is computed as exp(-mean(log_probs)).
    Lower perplexity indicates the model is more confident about the text.
    
    Args:
        text: Input text string to compute perplexity for
        model: Pre-trained masked language model
        tokenizer: Tokenizer corresponding to the model
        device: Device the model is on ('cuda' or 'cpu')
    
    Returns:
        tuple: (sent_ppl, token_scores)
            - sent_ppl: float, sentence-level pseudo-perplexity score
            - token_scores: list of dicts containing per-token information:
                - position: token position in sequence
                - token: token text
                - log_prob: log probability of the token
                - token_ppl: token-level perplexity (exp(-log_prob))
    """
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    seq_len = input_ids.size(1)
    token_scores = []

    for t in range(seq_len):
        token_id = input_ids[0, t].item()

        # skip special tokens (<s>, </s>, <pad>, etc.)
        if token_id in tokenizer.all_special_ids:
            continue

        masked_ids = input_ids.clone()
        masked_ids[0, t] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(
                input_ids=masked_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits[0, t]
            log_prob = F.log_softmax(logits, dim=-1)[token_id]

        token = tokenizer.convert_ids_to_tokens(token_id)
        token_scores.append({
            "position": t,
            "token": token,
            "log_prob": log_prob.item(),
            "token_ppl": torch.exp(-log_prob).item()
        })

    mean_log_prob = torch.tensor(
        [x["log_prob"] for x in token_scores],
        device=device
    ).mean()

    sent_ppl = torch.exp(-mean_log_prob)

    return sent_ppl.item(), token_scores