#!/bin/bash
dataset="saiteki-kai/Beavertails-it"
split="330k_test"
config_name="multilingual"
output_dir="./output/metrics"

models=(
    "PKU-Alignment/beaver-dam-7b"
    "meta-llama/Llama-Guard-3-8B"
    "meta-llama/Llama-Guard-4-12B"
    "saiteki-kai/QA-DeBERTa-v3-large"
    "saiteki-kai/QA-Llama-3.1"
    "saiteki-kai/QA-Llama-Guard-3-8B"
)

predictions_path() {
    model="$1"
    echo "./output/predictions_${model/\//__}.npy"
}

for model in "${models[@]}"; do
    echo "Computing metrics for $model"

    python ./scripts/compute_metrics.py \
        --model-name "$model" \
        --dataset-name "$dataset" \
        --split "$split" \
        --config-name "$config_name" \
        --predictions-filepath "$(predictions_path "$model")" \
        --output-dir "$output_dir"
done
