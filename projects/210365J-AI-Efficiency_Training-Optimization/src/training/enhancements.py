import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_scheduler

class TrainingEnhancements:
    def __init__(self, config):
        self.config = config
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
    def setup_mixed_precision(self):
        """Configure mixed precision training"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_mixed_precision)
        
    def create_optimizer(self, model):
        """Create optimizer with weight decay"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def create_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler with warmup"""
        warmup_steps = int(self.config.warmup_epochs * num_training_steps / self.config.num_epochs)
        
        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return scheduler
    
    def dynamic_batch_schedule(self, current_epoch, total_epochs):
        """Dynamic batch size scheduling based on epoch"""
        base_batch = self.config.batch_size
        if current_epoch < total_epochs * 0.3:
            return base_batch  # Start with base batch size
        elif current_epoch < total_epochs * 0.6:
            return base_batch * 2  # Increase batch size
        else:
            return base_batch * 4  # Maximum batch size
    
    def apply_activation_checkpointing(self, model):
        """Apply activation checkpointing to save memory"""
        if hasattr(model, 'transformer'):
            # For GPT-2 models
            model.transformer.gradient_checkpointing = True
        return model