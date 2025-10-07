#!/usr/bin/env python3
"""
Distill knowledge from ClinicalBERT into an extended DistilBERT student.

Requirements:
    pip install torch>=2.6 transformers>=5.0 datasets accelerate

Usage:
    python distill_clinicalbert.py \
        --train_file data/pubmed.jsonl \
        --output_dir artifacts/distilled_model \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# -----------------------
# Distillation Trainer
# -----------------------
class DistillTrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    # Accept num_items_in_batch for newer transformers
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs_s = model(**inputs, output_hidden_states=True)
        logits_s = outputs_s.logits

        with torch.no_grad():
            outputs_t = self.teacher(**inputs, output_hidden_states=True)
            logits_t = outputs_t.logits

        # MLM loss
        mlm_loss = self.ce_loss(
            logits_s.view(-1, logits_s.size(-1)),
            labels.view(-1),
        )

        # Distillation loss (KL)
        t_logits = logits_t / self.temperature
        s_logits = logits_s / self.temperature

        loss_kd = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Combine
        loss = self.alpha * mlm_loss + (1 - self.alpha) * loss_kd

        return (loss, outputs_s) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    student_tok = AutoTokenizer.from_pretrained("./artifacts/medtok/tokenizer")
    student = AutoModelForMaskedLM.from_pretrained("./artifacts/medtok/model")
    student.resize_token_embeddings(len(student_tok))
    student.to(device)

    teacher_name = "emilyalsentzer/Bio_ClinicalBERT"
    teacher = AutoModelForMaskedLM.from_pretrained(teacher_name)
    teacher.resize_token_embeddings(len(student_tok))  # Align vocab sizes
    teacher.eval()
    teacher.to(device)

    if args.train_file.endswith(".jsonl"):
        dataset = load_dataset("json", data_files={"train": args.train_file})
        column_name = "text"
    else:
        dataset = load_dataset("text", data_files={"train": args.train_file})
        column_name = "text"

    def tokenize_fn(examples):
        return student_tok(
            examples[column_name],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=[column_name])


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tok,
        mlm=True,
        mlm_probability=0.15,
    )

    # -----------------------
    # Training arguments
    # -----------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        report_to="none",  # disable wandb
    )

    trainer = DistillTrainer(
        model=student,
        teacher=teacher,
        args=training_args,
        train_dataset=tokenized["train"],
        processing_class=student_tok,  # replace deprecated tokenizer arg
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    student_tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
