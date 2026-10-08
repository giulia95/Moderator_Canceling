# Moderator Project

## Execution

### Model Fine-Tuning

To run model fine-tuning:

```bash
cd ./qa_moderation/

python scripts/train.py \
    --config ./configs/train_deberta_CADE.yaml
```

### Model Prediction on the CADE Test Set

To run model prediction on the CADE test set:

```bash
cd ./qa_moderation/

python scripts/compute_predictions.py \
    --model-name [PATH TO MODEL] \
    --dataset-name aequa-tech/CADE \
    --out-filepath ./predictions.npy \
    --label_processing "<2" \
    --label_column average_acceptability \
    --problem-type single_label_classification \
    --cluster_filter_mode "all"

python scripts/compute_metrics.py \
    --model-name QA-DeBERTa-v3-large \
    --dataset-name aequa-tech/CADE \
    --predictions-filepath ./predictions.npy \
    --output-dir [OUTPUT DIR] \
    --label_processing "<2" \
    --label_column average_acceptability \
    --problem-type single_label_classification \
    --cluster_filter_mode "all"
```

### Model Prediction on Unseen Data

To run model prediction on unseen data:

```bash
cd ./qa_moderation/

python scripts/predict_and_flag_batch.py \
    --input-csv ./target_group_comments.csv \
    --output-csv ./predictions_output.csv \
    --model-name [FINETUNED MODEL FOLDER] \
    --reference-csv [CSV WITH PREDICTIONS] \
    --alpha 0.05 \
    --resume \
    --batch-size 64
```

In this last example, `--reference-csv` is an optional reference CSV used to calibrate the confidence threshold.

## Citation

If you found our work useful, please cite our papers.

### Original Moderator

**Uncovering Unsafety Traits in Italian Language Models**

[Paper](https://aclanthology.org/2025.clicit-1.91.pdf)

```bibtex
@inproceedings{rizzi2025uncovering,
  title={Uncovering Unsafety Traits in Italian Language Models},
  author={Rizzi, Giulia and Magazz{\`u}, Giuseppe and Sormani, Alberto and Puler{\`a}, Francesca and Scalena, Daniel and Fersini, Elisabetta},
  booktitle={Proceedings of the Eleventh Italian Conference on Computational Linguistics (CLiC-it 2025)},
  pages={974--982},
  year={2025}
}
```

### BeaverTails-IT

**BeaverTails-IT: Towards A Safety Benchmark for Evaluating Italian Large Language Models**

[Paper](https://aclanthology.org/2025.clicit-1.60.pdf)

```bibtex
@inproceedings{magazzu2025beavertails,
  title={BeaverTails-IT: Towards A Safety Benchmark for Evaluating Italian Large Language Models},
  author={Magazz{\`u}, Giuseppe and Sormani, Alberto and Rizzi, Giulia and Puler{\`a}, Francesca and Scalena, Daniel and Cariddi, Stefano and Michielon, Edoardo and Pasqualini, Marco and Stamile, Claudio and Fersini, Elisabetta},
  booktitle={Proceedings of the Eleventh Italian Conference on Computational Linguistics (CLiC-it 2025)},
  pages={625--635},
  year={2025}
}
```