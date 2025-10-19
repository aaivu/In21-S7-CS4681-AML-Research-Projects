# qat_train_disaster_no_benchmark.py
"""
QAT Fine-tuning (training + final evaluation) for Qwen2.5-0.5B on disaster-domain dataset.
This file is the same as your experimental script but WITHOUT the benchmarking utilities.
It injects FakeQuantLinear modules for QAT simulation, trains, reverts layers, saves model,
and computes final validation perplexity.

Expect input JSON format: list[{"chunk": "..."}]
"""

import os
import json
import math
import random
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================================================
# CONFIGURATION (kept from your experiment)
# =========================================================
HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
QAT_OUTPUT_DIR = "model_checkpoints/resqedge"
FLOOD_JSON = "../data/disaster_corpus/flood.json"

EPOCHS = 2
BATCH_SIZE = 4
LR = 1e-6
BLOCK_SIZE = 128
NUM_BITS = 8
SEED = 42
USE_FP16 = False   # disable mixed precision for QAT
SAVE_EVERY = 1     # (kept, not used for benchmarking)

torch.manual_seed(SEED)
random.seed(SEED)

# =========================================================
# DEVICE
# =========================================================
def choose_device(pref="auto"):
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = choose_device("cuda")
print(f"Using device: {device}")

# =========================================================
# FAKE QUANT WRAPPERS
# =========================================================
class FakeQuantLinear(nn.Module):
    def __init__(self, linear_module, num_bits=8, fake_quant_activations=True):
        super().__init__()
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.bias = linear_module.bias is not None
        self.weight = linear_module.weight
        self.bias_param = linear_module.bias
        self.num_bits = num_bits
        self.fake_quant_activations = fake_quant_activations

    def _get_scale_zero_point(self, tensor):
        qmin, qmax = 0, 2**self.num_bits - 1
        # move to cpu for stable min/max extraction
        min_val = float(tensor.min().detach().cpu())
        max_val = float(tensor.max().detach().cpu())
        if max_val - min_val == 0:
            scale = 1.0
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
        zero_point = round(qmin - min_val / scale) if scale != 0 else 0
        return float(scale), int(zero_point)

    def forward(self, x):
        qmin, qmax = 0, 2**self.num_bits - 1
        w_scale, w_zp = self._get_scale_zero_point(self.weight)
        fq_w = torch.fake_quantize_per_tensor_affine(self.weight, w_scale, w_zp, qmin, qmax)

        if self.fake_quant_activations:
            a_scale, a_zp = self._get_scale_zero_point(x)
            fq_x = torch.fake_quantize_per_tensor_affine(x, a_scale, a_zp, qmin, qmax)
            return nn.functional.linear(fq_x, fq_w, self.bias_param)
        else:
            return nn.functional.linear(x, fq_w, self.bias_param)


def replace_linear_with_fakequant(module, num_bits=8, fake_quant_activations=True):
    replaced = 0
    for name, child in list(module.named_children()):
        # recurse into children
        if len(list(child.children())) > 0:
            replaced += replace_linear_with_fakequant(child, num_bits, fake_quant_activations)
        # replace torch.nn.Linear with FakeQuantLinear
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child, num_bits, fake_quant_activations))
            replaced += 1
    return replaced


def revert_fakequant_to_linear(module):
    reverted = 0
    for name, child in list(module.named_children()):
        if len(list(child.children())) > 0:
            reverted += revert_fakequant_to_linear(child)
        if isinstance(child, FakeQuantLinear):
            linear = nn.Linear(child.in_features, child.out_features, bias=child.bias)
            # copy weights/bias back
            linear.weight = child.weight
            linear.bias = child.bias_param
            setattr(module, name, linear)
            reverted += 1
    return reverted

# =========================================================
# DATASET PREPARATION (safe grouping to avoid mismatch)
# =========================================================
def prepare_disaster_lm_dataset(tokenizer, flood_json: str, block_size: int = 128):
    """Prepare causal LM dataset from disaster-domain JSON (flood.json)."""
    with open(flood_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    disaster_texts = [entry.get("chunk", "").strip() for entry in data if entry.get("chunk") and entry["chunk"].strip()]
    print(f"Loaded {len(disaster_texts)} disaster text chunks.")

    ds = Dataset.from_dict({"text": disaster_texts})

    def tokenize_function(examples):
        # no truncation here — we will group to blocks manually
        return tokenizer(examples["text"], truncation=False)

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # concatenate lists
        concatenated_ids = sum(examples["input_ids"], [])
        concatenated_masks = sum(examples["attention_mask"], [])
        total_length = (len(concatenated_ids) // block_size) * block_size

        # create blocks (ensure same slicing for both)
        input_blocks = [concatenated_ids[i:i+block_size] for i in range(0, total_length, block_size)]
        mask_blocks = [concatenated_masks[i:i+block_size] for i in range(0, total_length, block_size)]

        # safety: ensure same number of blocks
        min_len = min(len(input_blocks), len(mask_blocks))
        if min_len == 0:
            return {"input_ids": [], "attention_mask": []}
        input_blocks = input_blocks[:min_len]
        mask_blocks = mask_blocks[:min_len]

        return {"input_ids": input_blocks, "attention_mask": mask_blocks}

    lm_ds = tokenized.map(group_texts, batched=True, batch_size=1000)
    # keep only full-length blocks
    lm_ds = lm_ds.filter(lambda x: len(x["input_ids"]) == block_size)
    lm_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"Prepared {len(lm_ds)} training samples.")
    return lm_ds

# =========================================================
# DATA COLLATOR
# =========================================================
def data_collator(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

# =========================================================
# MAIN TRAINING LOOP
# =========================================================
def main():
    os.makedirs(QAT_OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME).to(device)

    print("Injecting FakeQuant layers for QAT...")
    replaced = replace_linear_with_fakequant(model, num_bits=NUM_BITS, fake_quant_activations=True)
    print(f"Replaced {replaced} Linear layers with FakeQuantLinear.")

    train_ds = prepare_disaster_lm_dataset(tokenizer, FLOOD_JSON, block_size=BLOCK_SIZE)
    split = train_ds.train_test_split(test_size=0.1, seed=SEED)
    train_dataset, eval_dataset = split["train"], split["test"]

    training_args = TrainingArguments(
        output_dir=QAT_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=False,  # disable mixed precision for QAT
        save_strategy="no",
        report_to=[],
        logging_steps=10,
        # evaluation compatible for older transformers
        do_eval=True,
        eval_steps=500,  # evaluate every 500 steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting QAT training on disaster-domain data...")
    trainer.train()

    print("Training complete. Reverting fake quant layers to linear for export...")
    revert_fakequant_to_linear(model)

    print("Saving final QAT checkpoint...")
    trainer.save_model(QAT_OUTPUT_DIR)
    tokenizer.save_pretrained(QAT_OUTPUT_DIR)

    # compute final perplexity
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(DataLoader(eval_dataset, batch_size=1), desc="Evaluating"):
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in batch.items()}, labels=batch["input_ids"].to(device))
        loss = outputs.loss
        total_loss += loss.item() * batch["input_ids"].numel()
        total_tokens += batch["input_ids"].numel()
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    print(f"Final Validation Perplexity: {ppl:.3f}")

if __name__ == "__main__":
    main()