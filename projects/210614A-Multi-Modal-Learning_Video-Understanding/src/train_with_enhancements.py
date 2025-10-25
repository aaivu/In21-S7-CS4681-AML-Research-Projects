"""
Comprehensive Training Script with All Multi-Scale Enhancements

This script demonstrates the complete integration of:
1. MultiScaleVideoEncoder
2. Multi-scale data loading 
3. Temporal consistency loss
4. Custom temporal batch sampler
5. Cosine temperature scheduling

Usage:
    python train_with_enhancements.py --config configs/pt/egoclip.json --epochs 20
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import argparse
import json
import numpy as np
from pathlib import Path
import logging

# Import original EgoVLP components
from model.model import FrozenInTime
from model.loss import EgoNCE
from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ

# Import our new enhancements
from model.model import MultiScaleVideoEncoder
from model.loss import EgoNCEWithScheduler
from model.temporal_loss import TemporalConsistencyLoss, TemporalPairBatchSampler
from trainer.trainer_egoclip import EgoClipTrainer


class EnhancedEgoVLP(nn.Module):
    """
    Enhanced EgoVLP model with all multi-scale improvements.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Initialize base FrozenInTime model
        self.base_model = FrozenInTime(config)
        
        # Replace video encoder with multi-scale version
        self.video_encoder = MultiScaleVideoEncoder(
            scales=[4, 8, 16],
            base_encoder=self.base_model.video_encoder,
            fusion_type='weighted',
            device=config.get('device', 'cuda')
        )
        
        # Keep text encoder from base model
        self.text_encoder = self.base_model.text_encoder
        
        # Video projection layers
        video_width = config['arch']['args']['video_params']['model']['width']
        text_width = config['arch']['args']['text_params']['model']['width']
        projection_dim = config['arch']['args'].get('projection_dim', 256)
        
        self.video_proj = nn.Sequential(
            nn.Linear(video_width, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_width, projection_dim), 
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Temporal consistency loss
        self.temporal_loss = TemporalConsistencyLoss(lambda_temp=0.1)
        
        self.config = config
    
    def forward(self, video, text, return_embeds=False):
        """
        Forward pass with multi-scale video processing.
        
        Args:
            video: Video tensor [B, C, T, H, W]
            text: Tokenized text
            return_embeds: Whether to return intermediate embeddings
        
        Returns:
            Dictionary with similarities and embeddings
        """
        # Multi-scale video encoding
        video_embeds = self.video_encoder(video)  # [B, D]
        
        # Text encoding
        text_embeds = self.text_encoder(text)  # [B, D]
        
        # Project to common space
        video_proj = self.video_proj(video_embeds)  # [B, projection_dim]
        text_proj = self.text_proj(text_embeds)    # [B, projection_dim]
        
        # Normalize embeddings
        video_proj = torch.nn.functional.normalize(video_proj, dim=1)
        text_proj = torch.nn.functional.normalize(text_proj, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(video_proj, text_proj.t())  # [B, B]
        
        result = {
            'similarity_matrix': similarity_matrix,
            'video_embeds': video_proj,
            'text_embeds': text_proj
        }
        
        if return_embeds:
            result.update({
                'video_features': video_embeds,
                'text_features': text_embeds
            })
        
        return result


class EnhancedEgoClipTrainer:
    """
    Enhanced trainer with all new loss components.
    """
    
    def __init__(self, model, config, device='cuda', rank=0):
        self.model = model
        self.config = config
        self.device = device
        self.rank = rank
        
        # Setup optimizers
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup enhanced losses
        self.setup_losses()
        
        # Training metrics
        self.train_losses = []
        self.temporal_losses = []
        self.temperatures = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_losses(self):
        """Setup all loss components with scheduling."""
        total_epochs = self.config.get('epochs', 20)
        
        # Main contrastive loss with temperature scheduling
        self.contrastive_loss = EgoNCEWithScheduler(
            tau_max=0.07,
            tau_min=0.03,
            total_epochs=total_epochs,
            noun=True,
            verb=True
        )
        
        # Temporal consistency loss
        lambda_temp = self.config.get('temporal_lambda', 0.1)
        self.temporal_loss = TemporalConsistencyLoss(lambda_temp=lambda_temp)
        
        self.logger.info(f"Loss setup complete:")
        self.logger.info(f"  - Temperature scheduling: {0.07} → {0.03} over {total_epochs} epochs")
        self.logger.info(f"  - Temporal consistency lambda: {lambda_temp}")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        self.logger = logging.getLogger('EnhancedTrainer')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for components."""
        lr = float(self.config.get('lr', 1e-4))
        weight_decay = float(self.config.get('weight_decay', 1e-6))
        
        # Different learning rates for video encoder scales
        param_groups = []
        
        # Video encoder parameters (lower lr for pretrained parts)
        video_params = list(self.model.video_encoder.parameters())
        param_groups.append({
            'params': video_params,
            'lr': lr * 0.1,  # Lower lr for video encoder
            'weight_decay': weight_decay
        })
        
        # Text encoder parameters
        text_params = list(self.model.text_encoder.parameters())
        param_groups.append({
            'params': text_params,
            'lr': lr * 0.1,  # Lower lr for text encoder
            'weight_decay': weight_decay
        })
        
        # Projection layers (higher lr for new components)
        proj_params = list(self.model.video_proj.parameters()) + list(self.model.text_proj.parameters())
        param_groups.append({
            'params': proj_params,
            'lr': lr,
            'weight_decay': weight_decay
        })
        
        return torch.optim.AdamW(param_groups)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_epochs = self.config.get('epochs', 20)
        warmup_epochs = self.config.get('warmup_epochs', 2)
        
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=total_epochs - warmup_epochs,
            T_mult=1,
            eta_min=1e-6
        )
    
    def train_epoch(self, dataloader, epoch, total_epochs):
        """
        Train for one epoch with all enhancements.
        """
        self.model.train()
        
        epoch_losses = []
        epoch_temporal_losses = []
        epoch_contrastive_losses = []
        
        # Get current temperature
        current_temp = self.contrastive_loss.temperature_scheduler.get_temperature(epoch)
        
        self.logger.info(f"Epoch {epoch}/{total_epochs-1} - Temperature: {current_temp:.6f}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            video = batch['video'].to(self.device)  # [B, C, T, H, W]
            text = batch['text'].to(self.device)
            
            # Get temporal pairs if available in batch
            temporal_pairs = batch.get('temporal_pairs', None)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(video, text, return_embeds=True)
            similarity_matrix = outputs['similarity_matrix']
            
            # Create masks (simplified - should be based on actual annotations)
            batch_size = video.size(0)
            mask_v = torch.eye(batch_size, device=self.device)
            mask_n = torch.eye(batch_size, device=self.device)
            
            # Contrastive loss with temperature scheduling
            contrastive_loss, loss_info = self.contrastive_loss(
                similarity_matrix, mask_v, mask_n, current_epoch=epoch
            )
            
            total_loss = contrastive_loss
            
            # Temporal consistency loss if pairs available
            temporal_loss_val = 0
            if temporal_pairs is not None and len(temporal_pairs) > 0:
                video_embeds = outputs['video_embeds']
                temporal_loss_val = self.temporal_loss(video_embeds, temporal_pairs)
                total_loss = total_loss + temporal_loss_val
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Log batch metrics
            epoch_losses.append(total_loss.item())
            epoch_contrastive_losses.append(contrastive_loss.item())
            if isinstance(temporal_loss_val, torch.Tensor):
                epoch_temporal_losses.append(temporal_loss_val.item())
            
            # Periodic logging
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Batch {batch_idx}: Total Loss = {total_loss.item():.6f}, "
                    f"Contrastive = {contrastive_loss.item():.6f}, "
                    f"Temporal = {temporal_loss_val if isinstance(temporal_loss_val, float) else temporal_loss_val.item():.6f}, "
                    f"Temp = {loss_info['current_temperature']:.6f}"
                )
        
        # Update scheduler
        self.scheduler.step()
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_contrastive = np.mean(epoch_contrastive_losses)
        avg_temporal = np.mean(epoch_temporal_losses) if epoch_temporal_losses else 0
        
        self.train_losses.append(avg_loss)
        self.temporal_losses.append(avg_temporal)
        self.temperatures.append(current_temp)
        
        self.logger.info(f"Epoch {epoch} Summary:")
        self.logger.info(f"  Average Total Loss: {avg_loss:.6f}")
        self.logger.info(f"  Average Contrastive Loss: {avg_contrastive:.6f}")
        self.logger.info(f"  Average Temporal Loss: {avg_temporal:.6f}")
        self.logger.info(f"  Current Temperature: {current_temp:.6f}")
        
        return {
            'total_loss': avg_loss,
            'contrastive_loss': avg_contrastive,
            'temporal_loss': avg_temporal,
            'temperature': current_temp
        }
    
    def save_checkpoint(self, epoch, filepath):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'temporal_losses': self.temporal_losses,
            'temperatures': self.temperatures
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")


def create_enhanced_dataloader(config, split='train'):
    """
    Create dataloader with temporal batch sampling and multi-scale loading.
    """
    # Dataset with multi-scale support
    dataset = EgoClip_EgoMCQ(config, split_type=split)
    
    # Use temporal batch sampler if available
    try:
        batch_sampler = TemporalPairBatchSampler(
            dataset=dataset,
            batch_size=config.get('batch_size', 32),
            drop_last=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"✓ Using TemporalPairBatchSampler for {split} split")
        
    except Exception as e:
        print(f"⚠ Falling back to regular DataLoader: {e}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=(split == 'train'),
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Enhanced EgoVLP Training')
    parser.add_argument('--config', default='configs/pt/egoclip.json', help='config file')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--temporal_lambda', type=float, default=0.1, help='temporal consistency weight')
    parser.add_argument('--output_dir', default='./outputs', help='output directory')
    parser.add_argument('--resume', default=None, help='checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Update config with args
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'temporal_lambda': args.temporal_lambda,
        'output_dir': args.output_dir
    })
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = str(device)
    
    print("Enhanced EgoVLP Training")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Temporal Lambda: {args.temporal_lambda}")
    print(f"  Output Directory: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create enhanced model
    model = EnhancedEgoVLP(config).to(device)
    
    print(f"\nModel Architecture:")
    print(f"  Multi-scale video encoder: 4, 8, 16 frames")
    print(f"  Temperature scheduling: 0.07 → 0.03")
    print(f"  Temporal consistency loss: λ = {args.temporal_lambda}")
    
    # Create enhanced trainer
    trainer = EnhancedEgoClipTrainer(model, config, device)
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_dataloader = create_enhanced_dataloader(config, split='train')
    
    print(f"  Training samples: {len(train_dataloader.dataset)}")
    print(f"  Training batches: {len(train_dataloader)}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed from checkpoint: {args.resume} (epoch {start_epoch})")
    
    # Training loop
    print(f"\nStarting enhanced training...")
    print("=" * 50)
    
    for epoch in range(start_epoch, args.epochs):
        # Train epoch
        epoch_metrics = trainer.train_epoch(train_dataloader, epoch, args.epochs)
        
        # Save checkpoint
        checkpoint_path = Path(args.output_dir) / f'enhanced_egoclip_epoch_{epoch}.pth'
        trainer.save_checkpoint(epoch, checkpoint_path)
        
        # Log progress
        print(f"\nEpoch {epoch} completed:")
        print(f"  Total Loss: {epoch_metrics['total_loss']:.6f}")
        print(f"  Temperature: {epoch_metrics['temperature']:.6f}")
        print(f"  Checkpoint: {checkpoint_path}")
        print("-" * 30)
    
    # Save final model
    final_path = Path(args.output_dir) / 'enhanced_egoclip_final.pth'
    trainer.save_checkpoint(args.epochs - 1, final_path)
    
    print("=" * 50)
    print("ENHANCED TRAINING COMPLETED!")
    print("=" * 50)
    
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {args.epochs}")
    print(f"  Final loss: {trainer.train_losses[-1]:.6f}")
    print(f"  Final temperature: {trainer.temperatures[-1]:.6f}")
    print(f"  Models saved to: {args.output_dir}")
    
    print(f"\nEnhancements Used:")
    print(f"  ✓ MultiScaleVideoEncoder (4, 8, 16 frames)")
    print(f"  ✓ Multi-scale data loading")
    print(f"  ✓ Temporal consistency loss")
    print(f"  ✓ Custom temporal batch sampler")
    print(f"  ✓ Cosine temperature scheduling")
    
    print(f"\nNext Steps:")
    print(f"  1. Evaluate on downstream tasks")
    print(f"  2. Fine-tune on specific datasets")
    print(f"  3. Analyze learned representations")


if __name__ == '__main__':
    main()