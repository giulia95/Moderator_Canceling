# QA Moderation

Deberta moderator fine-tuned on [Beavertails](https://huggingface.co/datasets/PKU-Alignment/Beavertails)

## Requirements

- transformers
- datasets
- sentencepiece
- accelerate
- liger-kernel
- scikit-learn
- wandb (optional)
- hf_xet (optional)

Install the requirements using the following command:

```bash
pip install -r requirements.txt
```

## Usage

You can specify a config file to fine-tune the model using the following command:

```bash
python train.py configs/fine_tuning.yaml
```

or you can specify the arguments manually:

```bash
python train.py \
    --model_name_or_path microsoft/deberta-v3-large \
    --dataset_name PKU-Alignment/Beavertails \
    --num_train_epochs 3 \
    --learning_rate 6e-6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    ...
```
