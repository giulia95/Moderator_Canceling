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

load_thresholds() {
    # split model name by / and take the last part
    model="$1"
    model_short="${model##*/}"
    local results_file="./output/${model_short}/eval_results.json"

    if [ -f "$results_file" ]; then
        # produce a space-separated list of threshold values, e.g. "0.1 0.2"
        THRESHOLDS=$(jq -r 'to_entries | map(select(.key|contains("/threshold"))) | map(.value) | join(" ")' "$results_file" )

        if [ -z "$THRESHOLDS" ]; then
            echo "Warning: No thresholds found in $results_file for model $model" >&2
        fi
    else
        THRESHOLDS=""
    fi

    if [ -n "${THRESHOLDS}" ]; then
        # read into array and prepend the flag so expansion yields: --thresholds v1 v2 ...
        read -r -a thr_array <<< "$THRESHOLDS"
        thresholds_arg=(--thresholds "${thr_array[@]}")
    else
        thresholds_arg=()
    fi
}


for model in "${models[@]}"; do
    echo "Computing metrics for $model"

    load_thresholds "$model"

    python ./scripts/compute_metrics.py \
        --model-name "$model" \
        --dataset-name "$dataset" \
        --split "$split" \
        --config-name "$config_name" \
        --predictions-filepath "$(predictions_path "$model")" \
        --output-dir "$output_dir" \
        "${thresholds_arg[@]}"
done
