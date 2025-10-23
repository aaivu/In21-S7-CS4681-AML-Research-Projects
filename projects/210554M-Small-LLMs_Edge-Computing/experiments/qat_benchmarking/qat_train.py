# qat_train.py
import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset
import benchmark_utils as bu

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
QAT_OUTPUT_DIR = "qat_checkpoints/qwen_qat_fp"
NUM_BITS = 8
FAKE_QUANT_ACTIVATIONS = True
EPOCHS = 100
BATCH_SIZE = 2
LR = 2e-5
BLOCK_SIZE = 128
TRAIN_SAMPLES = 2000
USE_FP16 = True

# -----------------------------
def prepare_wikitext_dataset(tokenizer, block_size=128, train_samples=2000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    if train_samples > 0:
        ds = ds.select(range(min(len(ds), train_samples)))

    def tokenize(x):
        return tokenizer(x["text"], truncation=True, max_length=block_size)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated_ids = sum(examples["input_ids"], [])
        concatenated_mask = sum(examples["attention_mask"], [])
        total_length = (len(concatenated_ids) // block_size) * block_size

        result = {
            "input_ids": [
                concatenated_ids[i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
            "attention_mask": [
                concatenated_mask[i : i + block_size]
                for i in range(0, total_length, block_size)
            ],
        }
        return result

    lm_ds = tokenized.map(group_texts, batched=True)
    lm_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return lm_ds

def main():
    os.makedirs(QAT_OUTPUT_DIR, exist_ok=True)
    device = bu.choose_device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME).to(device)

    # Wrap with FakeQuantLinear
    bu.replace_linear_with_fakequant(model, num_bits=NUM_BITS, fake_quant_activations=FAKE_QUANT_ACTIVATIONS)

    train_ds = prepare_wikitext_dataset(tokenizer, block_size=BLOCK_SIZE, train_samples=TRAIN_SAMPLES)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=QAT_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=(device.type == "cuda" and USE_FP16),
        save_strategy="no",  # we will revert & save manually
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator
    )

    trainer.train()

    # Revert fake quant → plain Linear before saving
    bu.revert_fakequant_to_linear(model)

    trainer.save_model(QAT_OUTPUT_DIR)
    tokenizer.save_pretrained(QAT_OUTPUT_DIR)
    print("Saved FP QAT checkpoint to", QAT_OUTPUT_DIR)

if __name__ == "__main__":
    main()