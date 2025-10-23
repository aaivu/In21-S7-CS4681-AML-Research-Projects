# Main training script using Hugging Face Trainer
# Prints logs and final metrics to terminal.

import argparse
import os
from model import Model
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config, TrainingArguments, Trainer
from dataset_loader import load_librispeech_hf
from utils import DataCollatorCTCWithPadding, compute_metrics_logits, save_json
import torch
import numpy as np

def preprocess_function(batch, processor):
    # accept HF LibriSpeech dataset entries with 'audio' and 'text'
    speech = batch["audio"]["array"]
    inputs = processor(speech, sampling_rate=16000)
    with processor.as_target_processor():
        labels = processor(batch["text"]).input_ids
    return {"input_values": inputs.input_values[0], "labels": labels}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/wavlm-large")
    parser.add_argument("--processor_name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--output_dir", default="../results/experiment_1")
    parser.add_argument("--train_split", default="train.clean.100")
    parser.add_argument("--eval_split", default="validation.clean")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    asr = Model(model_dir=args.model_name, processor_dir=args.processor_name, dropout=args.dropout)
    processor = asr.get_processor()
    model = asr.get_model()

    print("Loading datasets...")
    train_ds = load_librispeech_hf(args.train_split)
    eval_ds = load_librispeech_hf(args.eval_split)

    print("Preprocessing datasets... (this may take a while)")
    train_ds = train_ds.map(asr.prepare, batched=True, batch_size=args.per_device_train_batch_size, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(asr.prepare, batched=True, batch_size=args.per_device_eval_batch_size, remove_columns=eval_ds.column_names)

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        push_to_hub=False,
        report_to="none"
    )

    # set compute_metrics closure
    def compute_metrics(pred):
        # attach processor to pred for use in utils.compute
        pred.processor = processor
        return compute_metrics_logits(pred)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,  # tokenizer not used by HF Trainer but pass for safety
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()
    print("Training complete. Saving model...")
    trainer.save_model(args.output_dir)
    # final evaluation
    print("Running final evaluation on eval split...")
    metrics = trainer.evaluate(eval_ds)
    print("Final evaluation metrics:", metrics)
    save_json(os.path.join(args.output_dir, "final_metrics.json"), metrics)

if __name__ == "__main__":
    main()
