#!/bin/bash

dataset_name="saiteki-kai/Beavertails-it"
split="330k_test"
config_name="multilingual"

cls_models=(
    "PKU-Alignment/beaver-dam-7b"
    "saiteki-kai/QA-DeBERTa-v3-large"
    "saiteki-kai/QA-Llama-3.1"
)

gen_models=(
    "meta-llama/Llama-Guard-3-8B"
    "meta-llama/Llama-Guard-4-12B"
    "saiteki-kai/QA-Llama-Guard-3-8B"
)

data_args=(
    --dataset-name "$dataset_name"
    --split "$split"
    --config-name "$config_name"
)

# Classification models
for model in "${cls_models[@]}"; do
    echo "Predicting $model"

    python ./scripts/compute_predictions.py \
        --task "classification" \
        --model-name "$model" \
        --out-filepath "./output/predictions_${model/\//__}.npy" \
        "${data_args[@]}"
done

# Generation models
for model in "${gen_models[@]}"; do
    echo "Predicting $model"

    if [[ "$model" == "saiteki-kai/QA-Llama-Guard-3-8B" ]]; then
        model_args=(
            --model-name "meta-llama/Llama-Guard-3-8B"
            --adapter-name "saiteki-kai/QA-Llama-Guard-3-8B"
            --out-filepath "./output/predictions_saiteki-kai__QA-Llama-Guard-3-8B.npy"
            --max-new-tokens 30
        )
    else
        model_args=(
            --model-name "$model"
            --out-filepath "./output/predictions_${model/\//__}.npy"
            --max-new-tokens 30
        )
    fi

    python ./scripts/compute_predictions.py \
        --task "generation" \
        "${data_args[@]}" \
        "${model_args[@]}"
done
