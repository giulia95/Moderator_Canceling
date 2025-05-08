from functools import partial
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from classification.utils import compute_all_metrics, format_prompts, load_qa_dataset


set_seed(42)

dataset_name = "saiteki-kai/Beavertails-it"
split = "330k_test"
config_name = "multilingual"

# load the dataset
dataset, labels, label2id, id2label = load_qa_dataset(dataset_name, split, config_name)

# load the tokenizer and model
# model_name = "PKU-Alignment/beaver-dam-7b"
model_name = "./output/DeBERTa-QA/checkpoint-16908"
model_name = "./output/checkpoints/5e180171-2f1d-4cf6-bbff-5284e537c23a/checkpoint-12681"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# check if the labels are the same as the model's labels
if set(labels) != set(model.config.id2label.values()):
    msg = "Labels do not match between dataset and model"
    raise ValueError(msg)

if id2label != model.config.id2label:
    msg = "Label2id does not match between dataset and model"
    raise ValueError(msg)

# format prompts
dataset = dataset.map(
    partial(format_prompts, tokenizer=tokenizer, template=None),
    desc="Formatting prompts using template",
)

# compute predictions
predictions = torch.zeros(len(dataset), len(labels), dtype=torch.bool)
with torch.inference_mode():
    for i, example in enumerate(tqdm(dataset.to_iterable_dataset(), total=len(dataset))):
        inputs = tokenizer(example["text"], truncation=True, padding=False, max_length=1024, return_tensors="pt")
        outputs = model(**inputs.to(model.device))
        logits = outputs.logits[0].detach()
        predictions[i, :] = logits > 0

    predictions = predictions.to(torch.int)

    filepath = Path(f"output/predictions_{model_name.replace('/', '__')}.pt")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, filepath)

metrics = compute_all_metrics(np.asarray(predictions), np.asarray(dataset["label"]), id2labels=id2label)

filepath = Path(f"output/metrics_{model_name.replace('/', '__')}.pt")
torch.save(metrics, filepath)

print(metrics)
