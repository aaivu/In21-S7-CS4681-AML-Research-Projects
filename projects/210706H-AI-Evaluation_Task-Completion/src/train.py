import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Any
import os

from model import CAFEModel
from loss_functions import CAFELoss
from utils import MetricsCalculator, save_results, set_random_seeds

logger = logging.getLogger(__name__)

class ToxicityDataset(Dataset):
    """Dataset class for toxicity evaluation."""
    
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Combine prompt and continuation
        text = f"{row['prompt']} {row['continuation']}".strip()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'toxicity_score': torch.tensor(row['toxicity'], dtype=torch.float),
            'sensitive_group': torch.tensor(row.get('identity_mention', 0), dtype=torch.long),
            'context_label': torch.tensor(row.get('context_label', 0), dtype=torch.long)
        }

class CAFETrainer:
    """Trainer class for CAFE model."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 loss_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 scheduler=None):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {'toxicity_loss': 0, 'fairness_loss': 0, 'context_loss': 0}
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            toxicity_scores = batch['toxicity_score'].to(self.device)
            sensitive_groups = batch['sensitive_group'].to(self.device)
            context_labels = batch['context_label'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, return_embeddings=True)
            predicted_scores = outputs['toxicity_scores']
            embeddings = outputs['embeddings']
            
            # Calculate loss
            if isinstance(self.loss_function, CAFELoss):
                loss, components = self.loss_function(
                    predicted_scores, toxicity_scores, embeddings,
                    sensitive_groups, context_labels
                )
                
                # Update component tracking
                for key, value in components.items():
                    if key in loss_components:
                        loss_components[key] += value
            else:
                # Simple loss (for baseline)
                loss = self.loss_function(predicted_scores, toxicity_scores)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_loss = total_loss / len(self.train_dataloader)
        
        if loss_components['toxicity_loss'] > 0:
            for key in loss_components:
                loss_components[key] /= len(self.train_dataloader)
        
        if self.scheduler:
            self.scheduler.step()
        
        result = {'avg_loss': avg_loss}
        result.update(loss_components)
        
        return result
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_sensitive_groups = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                toxicity_scores = batch['toxicity_score'].to(self.device)
                sensitive_groups = batch['sensitive_group'].to(self.device)
                context_labels = batch['context_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, return_embeddings=True)
                predicted_scores = outputs['toxicity_scores']
                embeddings = outputs['embeddings']
                
                # Calculate loss
                if isinstance(self.loss_function, CAFELoss):
                    loss, _ = self.loss_function(
                        predicted_scores, toxicity_scores, embeddings,
                        sensitive_groups, context_labels
                    )
                else:
                    loss = self.loss_function(predicted_scores, toxicity_scores)
                
                total_loss += loss.item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted_scores.cpu().numpy())
                all_targets.extend(toxicity_scores.cpu().numpy())
                all_sensitive_groups.extend(sensitive_groups.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)
        
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        sensitive_groups_np = np.array(all_sensitive_groups)
        
        # Convert to binary for F1 calculation
        targets_binary = (targets_np >= 0.5).astype(int)
        
        f1 = MetricsCalculator.calculate_f1_score(targets_binary, predictions_np)
        fairness_gap = MetricsCalculator.calculate_fairness_gap(predictions_np, sensitive_groups_np)
        
        return {
            'avg_loss': avg_loss,
            'f1_score': f1,
            'fairness_gap': fairness_gap
        }
    
    def train(self, num_epochs: int, save_dir: str = "results/models") -> Dict[str, Any]:
        """Train the model for specified number of epochs."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': []
        }
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            training_history['train_losses'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            training_history['val_losses'].append(val_metrics['avg_loss'])
            training_history['val_metrics'].append(val_metrics)
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['avg_loss']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1_score']:.4f}")
            logger.info(f"Fairness Gap: {val_metrics['fairness_gap']:.4f}")
            
            # Save best model
            if val_metrics['avg_loss'] < best_val_loss:
                best_val_loss = val_metrics['avg_loss']
                self.save_model(save_dir, "best_model.pt")
                logger.info("Saved new best model")
        
        # Save final model and training history
        self.save_model(save_dir, "final_model.pt")
        save_results(training_history, f"{save_dir}/training_history.json")
        
        logger.info("Training completed!")
        return training_history
    
    def save_model(self, save_dir: str, filename: str):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_function': self.loss_function
        }, os.path.join(save_dir, filename))

def prepare_data(df: pd.DataFrame, tokenizer, batch_size: int = 16, 
                test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation dataloaders."""
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, 
                                       stratify=(df['toxicity'] >= 0.5).astype(int))
    
    # Create datasets
    train_dataset = ToxicityDataset(train_df, tokenizer)
    val_dataset = ToxicityDataset(val_df, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader

def main():
    """Main training function."""
    from utils import setup_logging
    from data_augmentation import load_rtp_dataset
    
    # Setup
    setup_logging()
    set_random_seeds(42)
    
    # Load data
    logger.info("Loading data...")
    if os.path.exists("data/augmented/augmented_rtp.csv"):
        df = pd.read_csv("data/augmented/augmented_rtp.csv")
    else:
        df = load_rtp_dataset()
    
    # Subset for faster training (remove in production)
    df = df.head(1000)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = CAFEModel().to(device)
    
    # Prepare data
    train_dataloader, val_dataloader = prepare_data(df, model.tokenizer, batch_size=8)
    
    # Setup training
    loss_function = CAFELoss(alpha=1.0, beta=0.5, gamma=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Create trainer
    trainer = CAFETrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    
    # Train model
    training_history = trainer.train(num_epochs=5)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()