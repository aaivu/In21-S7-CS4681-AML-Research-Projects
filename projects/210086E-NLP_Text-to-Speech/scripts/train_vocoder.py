"""
Training script for iSTFT Vocoder

Train the iSTFT-based vocoder on VCTK dataset with proper logging,
checkpointing, and evaluation.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.istft_vocoder import iSTFTVocoder
from src.models.vocoder_utils import VocoderLoss, compute_mcd, count_parameters, mel_spectrogram
from src.data.vctk_dataset import get_vocoder_dataloaders

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')

class VocoderTrainer:
    """
    Trainer class for iSTFT Vocoder.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = VocoderLoss(
            lambda_time=config['lambda_time'],
            lambda_mel=config['lambda_mel'],
            lambda_stft=config['lambda_stft']
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(config['adam_b1'], config['adam_b2']),
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler (exponential decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=config['lr_decay']
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_mcd = float('inf')
        
        # Sample directory for audio logging
        self.sample_dir = self.log_dir / 'samples'
        self.sample_dir.mkdir(exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (mel, audio) in enumerate(pbar):
            mel = mel.to(self.device)
            audio = audio.to(self.device)
            
            # Forward pass
            audio_pred = self.model(mel)
            
            # Compute loss
            total_loss, loss_dict = self.criterion(audio_pred, audio)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Logging
            epoch_losses.append(total_loss.item())
            
            if self.global_step % self.config['log_interval'] == 0:
                # Log to tensorboard
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.global_step)
                for name, value in loss_dict.items():
                    if name != 'total':
                        loss_value = value.item() if isinstance(value, torch.Tensor) else value
                        self.writer.add_scalar(f'train/{name}_loss', loss_value, self.global_step)
                
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Validation
            if self.global_step % self.config['val_interval'] == 0:
                val_loss, val_mcd = self.validate()
                self.model.train()
                
                # Save checkpoint if best
                if val_mcd < self.best_val_mcd:
                    self.best_val_mcd = val_mcd
                    self.save_checkpoint('best_mcd.pt')
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_loss.pt')
            
            # Save periodic checkpoint
            if self.global_step % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(f'checkpoint_{self.global_step}.pt')
            
            self.global_step += 1
        
        # Learning rate decay per epoch
        self.scheduler.step()
        
        return np.mean(epoch_losses)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        val_losses = []
        val_mcds = []
        
        print("\nRunning validation...")
        
        for mel, audio in tqdm(self.val_loader, desc="Validation"):
            mel = mel.to(self.device)
            audio = audio.to(self.device)
            
            # Forward pass
            audio_pred = self.model(mel)
            
            # Compute loss
            total_loss, loss_dict = self.criterion(audio_pred, audio)
            val_losses.append(total_loss.item())
            
            # Compute MCD using mel_spectrogram function
            mel_config = {
                'n_fft': 1024,
                'hop_length': 256,
                'win_length': 1024,
                'n_mels': 80,
                'sample_rate': 22050,
                'f_min': 0.0,
                'f_max': 8000.0
            }
            mel_pred = mel_spectrogram(audio_pred.cpu(), **mel_config)
            mel_target = mel_spectrogram(audio.cpu(), **mel_config)
            
            mcd = compute_mcd(mel_pred, mel_target)
            val_mcds.append(mcd)
        
        avg_loss = np.mean(val_losses)
        avg_mcd = np.mean(val_mcds)
        
        # Log to tensorboard
        self.writer.add_scalar('val/total_loss', avg_loss, self.global_step)
        self.writer.add_scalar('val/mcd', avg_mcd, self.global_step)
        
        print(f"Validation - Loss: {avg_loss:.4f}, MCD: {avg_mcd:.3f} dB")
        
        # Log audio samples
        if self.global_step % (self.config['val_interval'] * 5) == 0:
            self._log_audio_samples(mel[:4], audio[:4], audio_pred[:4])
        
        return avg_loss, avg_mcd
    
    def _log_audio_samples(self, mel, audio_gt, audio_pred):
        """Log audio samples to tensorboard."""
        for i in range(min(4, mel.shape[0])):
            # Save audio
            self.writer.add_audio(
                f'audio/gt_{i}',
                audio_gt[i].cpu().unsqueeze(0),
                self.global_step,
                sample_rate=22050
            )
            self.writer.add_audio(
                f'audio/pred_{i}',
                audio_pred[i].cpu().unsqueeze(0),
                self.global_step,
                sample_rate=22050
            )
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mcd': self.best_val_mcd,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_mcd = checkpoint['best_val_mcd']
        
        print(f"Loaded checkpoint from step {self.global_step}")
    
    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Total steps: ~{num_epochs * len(self.train_loader)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*70}")
            
            epoch_loss = self.train_epoch()
            
            print(f"\nEpoch {epoch + 1} - Average Loss: {epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')
        
        print("\n" + "="*70)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation MCD: {self.best_val_mcd:.3f} dB")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train iSTFT Vocoder')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/VCTK-Corpus-0.92',
                        help='Path to VCTK dataset')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for caching preprocessed data')
    
    # Model arguments
    parser.add_argument('--mel_channels', type=int, default=80)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=6)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.999)
    parser.add_argument('--adam_b1', type=float, default=0.9)
    parser.add_argument('--adam_b2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Loss weights
    parser.add_argument('--lambda_time', type=float, default=1.0)
    parser.add_argument('--lambda_mel', type=float, default=45.0)
    parser.add_argument('--lambda_stft', type=float, default=1.0)
    
    # Logging and checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/istft_vocoder')
    parser.add_argument('--log_dir', type=str, default='logs/istft_vocoder')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1000)
    parser.add_argument('--checkpoint_interval', type=int, default=5000)
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--segment_length', type=int, default=16000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Configuration dictionary
    config = vars(args)
    
    print("="*70)
    print("iSTFT VOCODER TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)
    
    # Save config
    config_file = Path(config['checkpoint_dir']) / 'config.json'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config to: {config_file}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_vocoder_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_length=args.segment_length,
        cache_dir=args.cache_dir
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = iSTFTVocoder(
        mel_channels=args.mel_channels,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        dilation_pattern=[1, 3, 9, 27, 1, 3]
    )
    
    trainable, total = count_parameters(model)
    print(f"Model parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Model size: {total * 4 / (1024**2):.2f} MB")
    
    # Create trainer
    trainer = VocoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.num_epochs)
    
    print("\n✅ Training finished successfully!")


if __name__ == '__main__':
    main()
