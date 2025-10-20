"""
Training Script with 5-Fold Cross-Validation for Debiased Toxicity Classifier
Memory-Optimized Version with Mixed Precision and Gradient Accumulation
RoBERTa-base Version
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from RoBERTaForToxicity import RoBERTaForToxicity
from JigsawDataset import JigsawDataset, calculate_sample_weights


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss for toxicity classification.
    Uses BCE with logits for all 7 outputs.
    """
    
    def __init__(self, loss_weights=None):
        """
        Initialize multi-task loss.
        
        Args:
            loss_weights (dict): Weights for each task loss
        """
        super(MultiTaskLoss, self).__init__()
        
        if loss_weights is None:
            # Default weights: primary target has higher weight
            loss_weights = {
                'toxicity': 1.0,
                'severe_toxicity': 0.5,
                'obscene': 0.5,
                'identity_attack': 0.5,
                'insult': 0.5,
                'threat': 0.5,
                'sexual_explicit': 0.5
            }
        
        self.loss_weights = loss_weights
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, labels, sample_weights=None):
        """
        Calculate weighted multi-task loss.
        
        Args:
            predictions (dict): Model predictions for each task
            labels (dict): Ground truth labels for each task
            sample_weights (torch.Tensor): Per-sample weights for fairness
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        total_loss = 0.0
        loss_dict = {}
        
        for task, weight in self.loss_weights.items():
            # Calculate BCE loss for this task
            task_loss = self.bce_loss(predictions[task], labels[task])
            
            # Apply sample weights if provided
            if sample_weights is not None:
                task_loss = task_loss * sample_weights
            
            # Average the loss
            task_loss = task_loss.mean()
            
            # Add to total loss with task weight
            total_loss += weight * task_loss
            loss_dict[task] = task_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, sample_weights_dict, 
                scaler, accumulation_steps):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    task_losses = {
        'toxicity': 0.0, 'severe_toxicity': 0.0, 'obscene': 0.0,
        'identity_attack': 0.0, 'insult': 0.0, 'threat': 0.0, 'sexual_explicit': 0.0
    }
    
    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}
        
        # Get sample weights for this batch
        batch_size = input_ids.size(0)
        start_idx = batch_idx * dataloader.batch_size
        end_idx = min(start_idx + batch_size, len(dataloader.dataset))
        batch_weights = torch.tensor(
            [sample_weights_dict.get(i, 1.0) for i in range(start_idx, end_idx)], 
            dtype=torch.float32, 
            device=device
        )
        
        # Determine if this is the last step in accumulation cycle
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)
        
        # Forward pass with mixed precision (only on CUDA)
        if device.type == 'cuda' and scaler is not None:
            with torch.cuda.amp.autocast():
                predictions = model(input_ids, attention_mask)
                loss, loss_dict = criterion(predictions, labels, batch_weights)
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights at accumulation steps
            if is_accumulation_step or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            # CPU training without mixed precision
            predictions = model(input_ids, attention_mask)
            loss, loss_dict = criterion(predictions, labels, batch_weights)
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights at accumulation steps
            if is_accumulation_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # Track losses
        total_loss += loss_dict['total']
        for task in task_losses.keys():
            task_losses[task] += loss_dict[task]
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses


def evaluate_epoch(model, dataloader, criterion, device, scaler=None):
    """Evaluate on validation set with mixed precision."""
    model.eval()
    total_loss = 0.0
    task_losses = {
        'toxicity': 0.0, 'severe_toxicity': 0.0, 'obscene': 0.0,
        'identity_attack': 0.0, 'insult': 0.0, 'threat': 0.0, 'sexual_explicit': 0.0
    }
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            
            # Forward pass with mixed precision (only on CUDA)
            if device.type == 'cuda' and scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = model(input_ids, attention_mask)
                    loss, loss_dict = criterion(predictions, labels, sample_weights=None)
            else:
                predictions = model(input_ids, attention_mask)
                loss, loss_dict = criterion(predictions, labels, sample_weights=None)
            
            total_loss += loss_dict['total']
            for task in task_losses.keys():
                task_losses[task] += loss_dict[task]
            
            progress_bar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
    
    # Average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses


def train_fold(fold_num, train_df, val_df, config):
    """Train a single fold."""
    print(f"\n{'='*80}")
    print(f"Training Fold {fold_num + 1}/{config['n_folds']}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config['model_name'])
    
    # Calculate sample weights for training data
    sample_weights = calculate_sample_weights(train_df)
    sample_weights_dict = {i: sample_weights[i] for i in range(len(sample_weights))}
    
    # Create datasets
    train_dataset = JigsawDataset(train_df, tokenizer, max_length=config['max_length'])
    val_dataset = JigsawDataset(val_df, tokenizer, max_length=config['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=JigsawDataset.collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=JigsawDataset.collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    model = RoBERTaForToxicity(model_name=config['model_name'], dropout_rate=config['dropout'])
    
    # Disable gradient checkpointing if enabled (prevents double backward issues)
    if hasattr(model.roberta, 'gradient_checkpointing_disable'):
        model.roberta.gradient_checkpointing_disable()
    
    model = model.to(device)
    
    print(f"Model parameters: {model.get_trainable_parameters():,}")
    
    # Initialize loss and optimizer
    criterion = MultiTaskLoss(loss_weights=config['loss_weights'])
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize gradient scaler for mixed precision (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Calculate total steps and create scheduler
    total_steps = (len(train_loader) // config['accumulation_steps']) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Gradient accumulation steps: {config['accumulation_steps']}")
    print(f"Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
    print(f"Mixed precision: {'Enabled (CUDA)' if device.type == 'cuda' else 'Disabled (CPU)'}")
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = f"models/roberta_fold{fold_num}_best.pt"
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        # Train
        train_loss, train_task_losses = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, 
            sample_weights_dict, scaler, config['accumulation_steps']
        )
        
        # Validate
        val_loss, val_task_losses = evaluate_epoch(model, val_loader, criterion, device, scaler)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Toxicity Loss: {val_task_losses['toxicity']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'fold': fold_num
            }, best_model_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\nFold {fold_num + 1} complete. Best val loss: {best_val_loss:.4f}")
    
    # Clean up
    del model, optimizer, scheduler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_val_loss


def main():
    """Main training script with 5-fold cross-validation."""
    
    # Optimized configuration for RoBERTa-base (smaller model)
    config = {
        'model_name': 'roberta-base',
        'max_length': 256,  # Can increase to 512 if memory allows
        'batch_size': 8,    # Increased from 2 (RoBERTa-base is smaller)
        'accumulation_steps': 2,  # Reduced (effective batch size = 8 * 2 = 16)
        'epochs': 3,        # Can train for more epochs with smaller model
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'dropout': 0.1,
        'n_folds': 5,
        'sample_size': 1000,  # Set to None to use full dataset, or specify number of samples
        'loss_weights': {
            'toxicity': 1.0,
            'severe_toxicity': 0.5,
            'obscene': 0.5,
            'identity_attack': 0.5,
            'insult': 0.5,
            'threat': 0.5,
            'sexual_explicit': 0.5
        }
    }
    
    print("RoBERTa-base Training Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation steps: {config['accumulation_steps']}")
    print(f"  Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
    print(f"  Max sequence length: {config['max_length']}")
    print(f"  Mixed precision: Enabled")
    print(f"  Sample size: {config['sample_size'] if config['sample_size'] else 'Full dataset'}")
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    df = pd.read_parquet('jigsaw_train_preprocessed.parquet')
    
    # Sample data if specified
    if config['sample_size'] is not None and config['sample_size'] < len(df):
        print(f"Sampling {config['sample_size']} rows from {len(df)} total...")
        df = df.sample(n=config['sample_size'], random_state=42).reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    print(f"Target distribution:")
    print(df['target'].describe())
    
    # Initialize K-Fold
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    
    # Store validation losses for each fold
    fold_val_losses = []
    
    # Cross-validation loop
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"\nFold {fold_num + 1}: Train size = {len(train_df)}, Val size = {len(val_df)}")
        
        # Train fold
        val_loss = train_fold(fold_num, train_df, val_df, config)
        fold_val_losses.append(val_loss)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Cross-Validation Complete!")
    print(f"{'='*80}")
    print(f"Mean validation loss: {np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}")
    for i, loss in enumerate(fold_val_losses):
        print(f"Fold {i+1}: {loss:.4f}")
    
    print(f"\nAll models saved in 'models/' directory")


if __name__ == "__main__":
    main()