"""
Training wrapper script for EgoVLP Multi-Scale Enhancement

This script properly sets up the Python path and runs training with
the enhanced multi-scale components and comprehensive logging.

Usage:
    python train_wrapper.py --config configs/pt/egoclip_rtx3090_optimized.json
    python train_wrapper.py --config configs/pt/egoclip.json --resume checkpoint.pth
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

print("=" * 70)
print("🚀 EgoVLP Multi-Scale Enhancement - Training Wrapper")
print("=" * 70)
print(f"Project Root: {project_root}")
print(f"Python Path: {sys.path[0]}")
print()

# Check if we can import the required modules
try:
    import torch
    import data_loader.data_loader as module_data
    import model.model as module_arch
    print("✅ All core modules can be imported")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease make sure you're in the project root directory:")
    print(f"  cd {project_root}")
    sys.exit(1)

# Parse arguments
import argparse
parser = argparse.ArgumentParser(description='EgoVLP Multi-Scale Training')
parser.add_argument('-c', '--config', required=True, type=str, help='config file path')
parser.add_argument('-r', '--resume', default=None, type=str, help='path to checkpoint')
parser.add_argument('--device', default=None, type=str, help='device to use (e.g., cuda:0)')
parser.add_argument('--test-mode', action='store_true', help='run in test mode (synthetic data)')

args = parser.parse_args()

# Check if config file exists
config_path = Path(args.config)
if not config_path.exists():
    print(f"❌ Config file not found: {config_path}")
    print("\nAvailable configs:")
    for cfg in Path("configs/pt").glob("*.json"):
        print(f"  • {cfg}")
    sys.exit(1)

print(f"📋 Configuration: {config_path}")
if args.resume:
    print(f"📦 Resuming from: {args.resume}")
if args.test_mode:
    print("🧪 Running in TEST MODE (synthetic data)")
print()

# Check if dataset is available (unless test mode)
if not args.test_mode:
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_path = config.get('data_loader', {}).get('args', {}).get('data_path', '')
    if data_path and not Path(data_path).exists():
        print(f"⚠️  WARNING: Dataset not found at: {data_path}")
        print("\nOptions:")
        print("  1. Download the EgoClip dataset and update the path in the config")
        print("  2. Run in test mode with synthetic data: --test-mode")
        print("  3. Update the 'data_path' in the config file")
        print()
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("❌ Training cancelled")
            sys.exit(1)

# Import and run the training script
print("=" * 70)
print("🚀 Starting Training...")
print("=" * 70)
print()

try:
    # Import the training function
    from run import train_egoclip
    
    # Override sys.argv for the training script
    sys.argv = ['train_egoclip.py', '-c', args.config]
    if args.resume:
        sys.argv.extend(['-r', args.resume])
    if args.device:
        sys.argv.extend(['--device', args.device])
    
    # Run training
    train_egoclip.run()
    
except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("\nCommon issues:")
    print("  • Dataset path not configured correctly")
    print("  • Checkpoint file doesn't exist")
    print("  • Missing data files")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nPossible solutions:")
    print("  • Make sure you're running from the project root")
    print("  • Check that all dependencies are installed")
    print("  • Try: pip install sacred pymongo tensorboardX")
    
except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
    print("Training failed! Check the error above.")
    print("=" * 70)
    sys.exit(1)
