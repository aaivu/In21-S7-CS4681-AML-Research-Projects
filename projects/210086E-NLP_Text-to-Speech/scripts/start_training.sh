#!/bin/bash
# Quick start script for training iSTFT vocoder

echo "=========================================="
echo "iSTFT Vocoder Training - Quick Start"
echo "=========================================="

# Activate Python environment (modify as needed)
source .venv/bin/activate

cd "$(dirname "$0")/.."

# Check if VCTK dataset exists
if [ ! -d "data/VCTK-Corpus-0.92" ]; then
    echo "❌ VCTK dataset not found at data/VCTK-Corpus-0.92"
    echo "Please download and extract VCTK dataset first."
    exit 1
fi

echo "✓ VCTK dataset found"

# Test dataset loading
echo ""
echo "Testing dataset loading..."
python src/data/vctk_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset test failed"
    exit 1
fi

echo ""
echo "✓ Dataset test passed"

# Start training
echo ""
echo "Starting training..."
echo ""

python scripts/train_vocoder.py \
    --data_dir data/VCTK-Corpus-0.92 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 2e-4 \
    --checkpoint_dir checkpoints/istft_vocoder \
    --log_dir logs/istft_vocoder \
    --checkpoint_interval 5000 \
    --val_interval 1000 \
    --num_workers 4

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
