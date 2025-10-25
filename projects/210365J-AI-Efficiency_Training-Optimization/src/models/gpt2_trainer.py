import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader
from datasets import load_dataset
import time
from .enhancements import TrainingEnhancements

class GPT2Trainer:
    def __init__(self, config):
        self.config = config
        self.enhancements = TrainingEnhancements(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_data()
        self.setup_model()
    
    def setup_data(self):
        """Setup WikiText-2 dataset"""
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Tokenization (simplified - in practice use GPT2Tokenizer)
        self.vocab_size = 50257  # GPT-2 vocab size
        
        def tokenize_function(examples):
            # Simple character-level tokenization for demonstration
            text = " ".join(examples["text"])
            tokens = [min(ord(c) % self.vocab_size, self.vocab_size-1) for c in text if c.strip()]
            return {"input_ids": tokens[:512]}  # Truncate to max_length
        
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        
        self.train_loader = DataLoader(
            tokenized_datasets["train"], batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            tokenized_datasets["validation"], batch_size=self.config.batch_size)
    
    def setup_model(self):
        """Setup GPT-2 model"""
        configuration = GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=512,
            n_ctx=512,
            n_embd=768,  # Small GPT-2
            n_layer=12,
            n_head=12,
        )
        self.model = GPT2LMHeadModel(configuration)
        self.model = self.model.to(self.device)
        
        if self.config.use_activation_checkpointing:
            self.model = self.enhancements.apply_activation_checkpointing(self.model)
            
        self.optimizer = self.enhancements.create_optimizer(self.model)
        self.criterion = nn.CrossEntropyLoss()
        
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = self.enhancements.create_scheduler(self.optimizer, num_training_steps)
        self.enhancements.setup_mixed_precision()
    
    def train_epoch(self, epoch):
        """Train one epoch for GPT-2"""
        self.model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, batch in enumerate(self.train_loader):
            inputs = batch["input_ids"].to(self.device)
            targets = inputs.clone()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(inputs, labels=targets)
                loss = outputs.loss
                loss = loss / self.enhancements.gradient_accumulation_steps
            
            # Mixed precision backward pass
            self.enhancements.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % self.enhancements.gradient_accumulation_steps == 0:
                self.enhancements.scaler.step(self.optimizer)
                self.enhancements.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            running_loss += loss.item() * self.enhancements.gradient_accumulation_steps
        
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(self.train_loader)
        
        return avg_loss, epoch_time
    
    def validate(self):
        """Calculate perplexity"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input_ids"].to(self.device)
                targets = inputs.clone()
                
                outputs = self.model(inputs, labels=targets)
                loss = outputs.loss
                
                total_loss += loss.item() * inputs.size(0)
                total_tokens += inputs.size(0)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return perplexity.item()
    
    def train(self):
        """Complete training loop for GPT-2"""
        results = {
            'train_loss': [], 'perplexity': [], 
            'epoch_times': [], 'memory_usage': []
        }
        
        for epoch in range(self.config.num_epochs):
            train_loss, epoch_time = self.train_epoch(epoch)
            perplexity = self.validate()
            
            # Record memory usage
            memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            results['train_loss'].append(train_loss)
            results['perplexity'].append(perplexity)
            results['epoch_times'].append(epoch_time)
            results['memory_usage'].append(memory_allocated)
            
            print(f'Epoch {epoch+1}/{self.config.num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Perplexity: {perplexity:.2f}, '
                  f'Time: {epoch_time:.2f}s, Memory: {memory_allocated:.2f}GB')
        
        return results