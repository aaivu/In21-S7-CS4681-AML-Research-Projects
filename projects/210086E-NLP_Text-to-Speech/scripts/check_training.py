"""
Utility script to check training status and best checkpoints.
"""

import json
from pathlib import Path
import torch

def check_training_status(checkpoint_dir='checkpoints/istft_vocoder'):
    """Check current training status."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print("❌ No checkpoints found. Training not started yet.")
        return
    
    print("="*70)
    print("TRAINING STATUS CHECK")
    print("="*70)
    
    # Load config
    config_file = checkpoint_dir / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("\nTraining Configuration:")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Total epochs: {config['num_epochs']}")
    
    # Find all checkpoints
    checkpoints = {
        'best_mcd': checkpoint_dir / 'best_mcd.pt',
        'best_loss': checkpoint_dir / 'best_loss.pt',
        'periodic': sorted(checkpoint_dir.glob('checkpoint_*.pt')),
        'epoch': sorted(checkpoint_dir.glob('epoch_*.pt'))
    }
    
    print("\nAvailable Checkpoints:")
    
    # Best checkpoints
    if checkpoints['best_mcd'].exists():
        ckpt = torch.load(checkpoints['best_mcd'], map_location='cpu')
        print(f"  ✓ best_mcd.pt")
        print(f"    - Step: {ckpt['global_step']}")
        print(f"    - Epoch: {ckpt['epoch']}")
        print(f"    - Best MCD: {ckpt['best_val_mcd']:.3f} dB")
    
    if checkpoints['best_loss'].exists():
        ckpt = torch.load(checkpoints['best_loss'], map_location='cpu')
        print(f"  ✓ best_loss.pt")
        print(f"    - Step: {ckpt['global_step']}")
        print(f"    - Best Loss: {ckpt['best_val_loss']:.4f}")
    
    # Periodic checkpoints
    if checkpoints['periodic']:
        print(f"\n  Periodic Checkpoints: {len(checkpoints['periodic'])} files")
        latest = checkpoints['periodic'][-1]
        ckpt = torch.load(latest, map_location='cpu')
        print(f"  ✓ Latest: {latest.name}")
        print(f"    - Step: {ckpt['global_step']}")
        print(f"    - Epoch: {ckpt['epoch']}")
    
    # Epoch checkpoints
    if checkpoints['epoch']:
        print(f"\n  Epoch Checkpoints: {len(checkpoints['epoch'])} files")
        latest = checkpoints['epoch'][-1]
        print(f"  ✓ Latest: {latest.name}")
    
    print("\n" + "="*70)
    
    # Recommendations
    print("\nRecommendations:")
    if checkpoints['best_mcd'].exists():
        print("  → Use best_mcd.pt for evaluation (best audio quality)")
    if checkpoints['best_loss'].exists():
        print("  → Use best_loss.pt for integration (most stable)")
    if checkpoints['periodic']:
        latest = checkpoints['periodic'][-1]
        print(f"  → Resume training from: {latest.name}")
    
    print("\n" + "="*70)


def load_checkpoint_info(checkpoint_path):
    """Load and display checkpoint information."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint: {Path(checkpoint_path).name}")
    print("="*50)
    print(f"Global Step: {ckpt['global_step']}")
    print(f"Epoch: {ckpt['epoch']}")
    print(f"Best Validation Loss: {ckpt['best_val_loss']:.4f}")
    print(f"Best Validation MCD: {ckpt['best_val_mcd']:.3f} dB")
    
    # Model info
    model_params = sum(p.numel() for p in ckpt['model_state_dict'].values())
    print(f"\nModel Parameters: {model_params:,}")
    print(f"Model Size: {model_params * 4 / (1024**2):.2f} MB")
    
    # Config
    if 'config' in ckpt:
        config = ckpt['config']
        print(f"\nTraining Config:")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Batch Size: {config['batch_size']}")
    
    print("="*50)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check training status')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/istft_vocoder',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint to inspect')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        load_checkpoint_info(args.checkpoint)
    else:
        check_training_status(args.checkpoint_dir)
