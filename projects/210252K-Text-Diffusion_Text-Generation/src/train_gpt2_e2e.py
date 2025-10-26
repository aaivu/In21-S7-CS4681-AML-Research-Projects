# # train_gpt2_e2e_custom.py
# import torch
# from transformers import (
#     GPT2LMHeadModel, 
#     GPT2Tokenizer, 
#     Trainer, 
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from torch.utils.data import Dataset
# import pandas as pd
# import os

# class E2EDataset(Dataset):
#     def __init__(self, data_file, tokenizer, max_length=64):
#         self.samples = []
#         with open(data_file, encoding="utf-8") as f:
#             for line in f:
#                 parts = line.strip().split("||")
#                 if len(parts) == 2:
#                     mr, ref = parts
#                     self.samples.append(ref)
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         text = self.samples[idx] + self.tokenizer.eos_token
#         encodings = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encodings['input_ids'].flatten(),
#             'attention_mask': encodings['attention_mask'].flatten(),
#             'labels': encodings['input_ids'].flatten()
#         }

# def main():
#     # Paths - adjust these to your E2E data location
#     train_file = "datasets/e2e_data/src1_train.txt"
#     val_file = "datasets/e2e_data/src1_valid.txt"
#     output_dir = "./finetuned_gpt2_e2e"
    
#     # Load tokenizer and model
#     print("Loading GPT-2...")
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
    
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#     model.resize_token_embeddings(len(tokenizer))
    
#     # Create datasets
#     print("Loading E2E data...")
#     train_dataset = E2EDataset(train_file, tokenizer)
#     eval_dataset = E2EDataset(val_file, tokenizer)
    
#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=20,
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=16,
#         per_device_eval_batch_size=4,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=5e-5,
#         fp16=True,
#         logging_steps=100,
#         save_total_limit=2,
#         load_best_model_at_end=True,
#     )
    
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=False
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#     )
    
#     # Train
#     print("Training...")
#     trainer.train()
    
#     # Save
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print(f"Model saved to {output_dir}")

# if __name__ == "__main__":
#     main()

## new code ---------------------------------------------------



# train_gpt2_e2e_optimized.py
# train_gpt2_e2e_fixed.py
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
)
from torch.utils.data import Dataset
import os

class E2EDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_length=64):
        """
        E2E data format: source||target (separated by ||)
        We only use the target (reference text) for language modeling
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
        print(f"Loading from {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '||' in line:
                    # Split by || and take the target text (second part)
                    parts = line.split('||')
                    if len(parts) >= 2:
                        target_text = parts[1].strip()
                        self.texts.append(target_text)
        
        print(f"Loaded {len(self.texts)} samples")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Add BOS and EOS tokens
        text = self.tokenizer.bos_token + " " + text + " " + self.tokenizer.eos_token
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

def main():
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Paths - UPDATE THESE to your actual file paths
    train_file = "datasets/e2e_data/src1_train.txt"
    val_file = "datasets/e2e_data/src1_valid.txt"
    output_dir = "./finetuned_gpt2_e2e"

    # Check if files exist
    if not os.path.exists(train_file):
        print(f"Error: Training file not found: {train_file}")
        print("Please update the train_file path in the script")
        return
    if not os.path.exists(val_file):
        print(f"Error: Validation file not found: {val_file}")
        print("Please update the val_file path in the script")
        return
    
    # Load model
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    print("\nLoading E2E data...")
    train_dataset = E2EDataset(train_file, tokenizer)
    eval_dataset = E2EDataset(val_file, tokenizer)
    
    # Training arguments optimized for RTX 3050 4GB
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        
        # Batch settings for 4GB GPU
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,  # Effective batch = 64
        per_device_eval_batch_size=8,
        
        # Speed optimizations
        fp16=True,                      # Mixed precision training
        dataloader_num_workers=4,       # Parallel data loading
        dataloader_pin_memory=True,     # Faster GPU transfer
        
        # Evaluation and saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,             # Keep only 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Optimizer settings
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        
        # Logging
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        report_to="none",               # No wandb/tensorboard
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training on GPU...")
    print("="*50 + "\n")
    
    trainer.train()
    
    # Save
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"\nTo use for evaluation:")
    print(f"  python classifiers/eval_control.py \\")
    print(f"    --input_text generated_samples/your_file.txt \\")
    print(f"    --input_format file \\")
    print(f"    --model_name_or_path {output_dir}")

if __name__ == "__main__":
    main()