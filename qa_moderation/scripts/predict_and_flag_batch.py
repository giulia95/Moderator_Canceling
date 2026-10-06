import argparse
import json
import os
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from moderation.classification.utils import format_prompts


def parse_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"

    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid torch dtype: {dtype_name}")

    return dtype


def get_ppl_threshold(
    df_input: pd.DataFrame,
    model_name: str,
    ppl_threshold: float | None,
    ppl_percentile: float,
    reference_csv: str | None,
    reference_ppl_column: str | None,
) -> float:
    if ppl_threshold is not None:
        return ppl_threshold

    if reference_csv is not None:
        ref_df = pd.read_csv(reference_csv)
        ref_col = reference_ppl_column or f"{model_name}_pseudo_perplexity"
        if ref_col not in ref_df.columns:
            raise ValueError(
                f"Column '{ref_col}' not found in reference CSV: {reference_csv}"
            )
        return float(np.percentile(ref_df[ref_col].values, ppl_percentile))

    return float(np.percentile(df_input["_pseudo_perplexity_temp"].values, ppl_percentile))

def get_confidence_threshold(
    default_threshold: float,
    reference_csv: str | None,
    reference_confidence_column: str | None,
    alpha: float,
    model_name: str,
) -> tuple[float, str]:
    """
    Return confidence threshold for low-confidence flagging.

        If reference_csv is provided, calibration prefers per-item nonconformity
        (per_item_loss) from compute_predictions.py, analogous to predict.py:
            threshold_loss = quantile(per_item_loss, 1 - alpha)
        and maps it to confidence threshold via:
            
             = exp(-threshold_loss)

        If per_item_loss is not available, it falls back to confidence-based
        calibration using the alpha percentile on confidence values.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if reference_csv is None:
        return default_threshold, "fixed"

    ref_df = pd.read_csv(reference_csv)

    ref_conf = None
    source = ""

    # 0) Preferred path: use per_item_loss produced by compute_predictions.py.
    loss_col = f"{model_name}_per_item_loss"
    if loss_col in ref_df.columns:
        ref_loss = pd.to_numeric(ref_df[loss_col], errors="coerce").dropna().values
        if ref_loss.size > 0:
            threshold_loss = float(np.percentile(ref_loss, (1.0 - alpha) * 100.0))
            threshold_conf = float(np.exp(-threshold_loss))
            source = f"reference_csv:{loss_col}->exp(-q_{1-alpha:.2f})"
            return threshold_conf, source

    # 1) Explicit confidence column provided by user.
    if reference_confidence_column is not None:
        if reference_confidence_column not in ref_df.columns:
            raise ValueError(
                f"Column '{reference_confidence_column}' not found in reference CSV: {reference_csv}"
            )
        ref_conf = pd.to_numeric(ref_df[reference_confidence_column], errors="coerce").dropna().values
        source = f"reference_csv:{reference_confidence_column}"

    # 2) Standard confidence column from predict_and_flag outputs.
    if ref_conf is None and "prediction_confidence" in ref_df.columns:
        ref_conf = pd.to_numeric(ref_df["prediction_confidence"], errors="coerce").dropna().values
        source = "reference_csv:prediction_confidence"

    # 3) compute_predictions.py output: one column per class like <model>_logit_class_0/1/...
    if ref_conf is None:
        model_class_cols = [
            c for c in ref_df.columns if c.startswith(f"{model_name}_logit_class_")
        ]
        if model_class_cols:
            class_probs = ref_df[model_class_cols].apply(pd.to_numeric, errors="coerce")
            ref_conf = class_probs.max(axis=1).dropna().values
            source = f"reference_csv:max({model_name}_logit_class_*)"

    # 4) Fallback: any *_logit_class_* columns if model-name-specific columns are not found.
    if ref_conf is None:
        generic_class_cols = [c for c in ref_df.columns if "_logit_class_" in c]
        if generic_class_cols:
            class_probs = ref_df[generic_class_cols].apply(pd.to_numeric, errors="coerce")
            ref_conf = class_probs.max(axis=1).dropna().values
            source = "reference_csv:max(*_logit_class_*)"

    # 5) Last resort: parse class_probabilities JSON column.
    if ref_conf is None and "class_probabilities" in ref_df.columns:
        parsed = []
        for val in ref_df["class_probabilities"].dropna().values:
            try:
                arr = json.loads(val)
                if isinstance(arr, list) and len(arr) > 0:
                    parsed.append(float(np.max(arr)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        ref_conf = np.array(parsed, dtype=np.float64)
        source = "reference_csv:class_probabilities"

    if ref_conf is None:
        raise ValueError(
            "Could not derive confidence from reference CSV. Provide --reference-confidence-column, "
            "or include 'prediction_confidence', '<model>_logit_class_*', or 'class_probabilities'."
        )

    if ref_conf.size == 0:
        raise ValueError(
            f"Reference confidence values are empty or invalid in {reference_csv}"
        )

    # Equivalent formulations:
    # 1) s = 1 - confidence; threshold_s = quantile(s, 1 - alpha); flag s > threshold_s
    # 2) confidence < quantile(confidence, alpha)
    threshold_conf = float(np.percentile(ref_conf, alpha * 100.0))
    return threshold_conf, source

def predict_single_label(logits: torch.Tensor, id2label: dict[int, str]):
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = id2label.get(pred_idx, str(pred_idx))
    confidence = float(np.max(probs))
    return probs, pred_idx, pred_label, confidence


def predict_multi_label(logits: torch.Tensor, id2label: dict[int, str], threshold: float):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    predicted_indices = np.where(probs >= threshold)[0].tolist()
    predicted_labels = [id2label.get(int(i), str(int(i))) for i in predicted_indices]

    confidence = float(np.max(np.abs(probs - 0.5) * 2.0))

    return probs, predicted_indices, predicted_labels, confidence


def atomic_save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Validate written JSON before replacing checkpoint.
        with temp_path.open("r", encoding="utf-8") as f:
            json.load(f)

        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_checkpoint(main_path: Path, backup_path: Path) -> tuple[list[dict], int]:
    for path in (main_path, backup_path):
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            rows = checkpoint_data["rows"]
            start_idx = int(checkpoint_data["last_processed_idx"]) + 1
            print(f"Loaded checkpoint from {path}")
            return rows, start_idx
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"Warning: could not load checkpoint {path}: {e}")

    return [], 0


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    checkpoint_backup_path = output_path.with_suffix(".checkpoint.bak.json")


    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise ValueError(
            f"Text column '{args.text_column}' not found in input CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Check for existing checkpoint
    rows = []
    start_idx = 0

    if args.resume and (checkpoint_path.exists() or checkpoint_backup_path.exists()):
        print(
            f"Found checkpoint files at {checkpoint_path} / {checkpoint_backup_path}, attempting resume..."
        )
        rows, start_idx = load_checkpoint(checkpoint_path, checkpoint_backup_path)
        if start_idx > 0:
            print(f"Resuming from row {start_idx}/{len(df)}")
        else:
            print("No valid checkpoint payload found, starting from scratch.")

    # load tokenizers/models
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        torch_dtype=parse_torch_dtype(args.dtype),
        device_map=args.device,
        trust_remote_code=True,
    )
    classifier.eval()

    id2label = {int(k): v for k, v in classifier.config.id2label.items()}
    problem_type = args.problem_type
    if problem_type is None:
        problem_type = classifier.config.problem_type or "single_label_classification"

    remaining = len(df) - start_idx
    with torch.inference_mode():
        for batch_start in tqdm(range(start_idx, len(df), args.batch_size), total=(remaining + args.batch_size - 1) // args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            batch_texts = []
            for raw_text in batch_df[args.text_column].tolist():
                text = str(raw_text)
                example = {"text": text}
                example = format_prompts(example, tokenizer=tokenizer, template=None)
                batch_texts.append(example["text"])

            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            outputs = classifier(**inputs.to(classifier.device))
            logits_batch = outputs.logits.detach()

            if problem_type == "multi_label_classification":
                probs_batch = torch.sigmoid(logits_batch).cpu().numpy()
            else:
                probs_batch = torch.softmax(logits_batch, dim=-1).cpu().numpy()

            for i, row_idx in enumerate(range(batch_start, batch_end)):
                probs = probs_batch[i]
                text = batch_texts[i]

                if problem_type == "multi_label_classification":
                    predicted_indices = np.where(probs >= args.multi_label_threshold)[0].tolist()
                    predicted_labels = [id2label.get(int(idx), str(int(idx))) for idx in predicted_indices]
                    confidence = float(np.max(np.abs(probs - 0.5) * 2.0))
                    pred_idx_value = json.dumps(predicted_indices)
                    pred_label_value = json.dumps(predicted_labels, ensure_ascii=False)
                else:
                    pred_idx = int(np.argmax(probs))
                    pred_label = id2label.get(pred_idx, str(pred_idx))
                    confidence = float(np.max(probs))
                    pred_idx_value = pred_idx
                    pred_label_value = pred_label

                rows.append(
                    {
                        "row_id": row_idx,
                        "text": text,
                        "predicted_label": pred_label_value,
                        "predicted_label_id": pred_idx_value,
                        "prediction_confidence": confidence,
                        "class_probabilities": json.dumps(probs.tolist()),
                        #"pseudo_perplexity": float(pseudo_ppl),
                        #"max_token_ppl": max_token_ppl,
                        #"mean_token_ppl": mean_token_ppl,
                        #"num_high_ppl_tokens": num_high_ppl_tokens,
                        #"token_scores": json.dumps(token_scores, ensure_ascii=False),
                    }
                )

                # Save checkpoint periodically
                if args.checkpoint_interval > 0 and (row_idx + 1) % args.checkpoint_interval == 0:
                    checkpoint_data = {
                        'last_processed_idx': row_idx,
                        'rows': rows
                    }
                    atomic_save_json(checkpoint_path, checkpoint_data)
                    atomic_save_json(checkpoint_backup_path, checkpoint_data)
                    print(f"\nCheckpoint saved at row {row_idx + 1}/{len(df)}")

    print("Inference completed")    

    result_df = pd.DataFrame(rows)

    # Pseudo-perplexity is intentionally disabled in this script execution path.
    # threshold_ppl = get_ppl_threshold(
    #     df_input=result_df,
    #     model_name=args.model_name,
    #     ppl_threshold=args.ppl_threshold,
    #     ppl_percentile=args.ppl_percentile,
    #     reference_csv=args.reference_csv,
    #     reference_ppl_column=args.reference_ppl_column,
    # )
    # print("threshold_ppl: " + str(threshold_ppl))
    confidence_threshold, threshold_source = get_confidence_threshold(
        default_threshold=args.confidence_threshold,
        reference_csv=args.reference_csv,
        reference_confidence_column=args.reference_confidence_column,
        alpha=args.alpha,
        model_name=args.model_name,
    )
    print(f"confidence_threshold: {confidence_threshold:.6f} (source={threshold_source})")

    low_conf_flag = result_df["prediction_confidence"] < confidence_threshold
    # high_ppl_flag = result_df["pseudo_perplexity"] > threshold_ppl
    high_ppl_flag = pd.Series(False, index=result_df.index)
    #high_token_flag = result_df["num_high_ppl_tokens"] >= args.min_high_ppl_tokens

    result_df["flag_low_confidence"] = low_conf_flag
    result_df["flag_high_ppl"] = high_ppl_flag
    #result_df["flag_many_high_ppl_tokens"] = high_token_flag
    result_df["manual_review"] = low_conf_flag | high_ppl_flag #| high_token_flag

    """
    reasons = []
    for i in range(len(result_df)):
        reason = []
        if low_conf_flag.iloc[i]:
            reason.append("low_confidence")
        if high_ppl_flag.iloc[i]:
            reason.append("high_pseudo_perplexity")
        #if high_token_flag.iloc[i]:
        #    reason.append("many_high_ppl_tokens")
        reasons.append(",".join(reason) if reason else "")

    result_df["manual_review_reason"] = reasons
    """

    # Equal but faster
    low_conf_np = low_conf_flag.to_numpy()
    high_ppl_np = high_ppl_flag.to_numpy()
    result_df["manual_review_reason"] = np.where(
        low_conf_np & high_ppl_np,
        "low_confidence,high_pseudo_perplexity",
        np.where(
            low_conf_np,
            "low_confidence",
            np.where(high_ppl_np, "high_pseudo_perplexity", ""),
        ),
    )

    # merge back original columns
    out_df = df.copy()
    """
    for col in result_df.columns:
        if col == "_pseudo_perplexity_temp":
            continue
        out_df[col] = result_df[col]
    """
    # Equal but faster
    result_cols = list(result_df.columns)
    out_df[result_cols] = result_df[result_cols]

    out_df.to_csv(output_path, index=False)

    """
    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint file removed: {checkpoint_path}")
    if checkpoint_backup_path.exists():
        checkpoint_backup_path.unlink()
        print(f"Checkpoint backup file removed: {checkpoint_backup_path}")
    """
    review_count = int(out_df["manual_review"].sum())
    print(f"Saved predictions to: {output_path}")
    print("Pseudo-perplexity threshold used: DISABLED")
    print(f"Confidence threshold used: {confidence_threshold:.6f} (source={threshold_source})")
    print(f"Manual review flagged: {review_count}/{len(out_df)} ({100*review_count/len(out_df):.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a moderation model, predict new samples, and flag items for manual review."
    )

    parser.add_argument("--input-csv", type=str, required=True, help="Path to input CSV with text samples.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to save predictions + manual review flags.")
    parser.add_argument("--text-column", type=str, default="text", help="Input column containing text.")

    parser.add_argument("--model-name", type=str, required=True, help="Hugging Face model path/name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="auto", help="Torch dtype name, e.g. float32, bfloat16, auto.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for classifier inference.")
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["single_label_classification", "multi_label_classification"],
        default=None,
        help="If omitted, read from model config.",
    )
    parser.add_argument("--multi-label-threshold", type=float, default=0.5)

    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Calibration level for reference-based confidence threshold (default: 0.05).",
    )
    parser.add_argument("--token-ppl-threshold", type=float, default=500000.0)
    parser.add_argument("--min-high-ppl-tokens", type=int, default=3)


    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Save checkpoint every N rows (0 to disable).")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")


    parser.add_argument(
        "--ppl-threshold",
        type=float,
        default=None,
        help="Absolute pseudo-perplexity threshold. If set, overrides percentile-based threshold.",
    )
    parser.add_argument(
        "--ppl-percentile",
        type=float,
        default=95.0,
        help="Percentile used to derive pseudo-perplexity threshold.",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default=None,
        help="Optional reference CSV used to calibrate confidence threshold.",
    )
    parser.add_argument(
        #"--reference-ppl-column",
        "--reference-confidence-column",
        type=str,
        default=None,
        #help="Pseudo-perplexity column in reference CSV (default: <model_name>_pseudo_perplexity).",
        help="Confidence column in reference CSV (default: prediction_confidence).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
