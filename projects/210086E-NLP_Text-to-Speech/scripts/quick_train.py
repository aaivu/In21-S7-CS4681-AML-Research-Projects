"""
Quick training script with default settings.
Run this to start training immediately.
"""

import subprocess
import sys
from pathlib import Path

# Change to project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import and run training
from scripts.train_vocoder import main

if __name__ == '__main__':
    print("Starting iSTFT Vocoder training with default settings...")
    print("(Press Ctrl+C to stop)\n")
    
    # Set default arguments
    sys.argv = [
        'train_vocoder.py',
        '--data_dir', 'data/VCTK-Corpus-0.92',
        '--batch_size', '16',
        '--num_epochs', '100',
        '--learning_rate', '2e-4',
        '--checkpoint_dir', 'checkpoints/istft_vocoder',
        '--log_dir', 'logs/istft_vocoder',
        '--num_workers', '4'
    ]
    
    main()
