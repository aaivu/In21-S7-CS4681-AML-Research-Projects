# Training the iSTFT Vocoder

This guide shows how to train the iSTFT vocoder on the VCTK dataset.

## Prerequisites

1. **VCTK Dataset**: Download and extract to `data/VCTK-Corpus-0.92/`
2. **Python Environment**: Python 3.8+ with PyTorch, torchaudio
3. **GPU**: CUDA-capable GPU recommended (training on CPU will be very slow)

## Quick Start

### Option 1: Run with default settings

```bash
cd /mnt/d/Academic/AML/In21-S7-CS4681-AML-Research-Projects/projects/210086E-NLP_Text-to-Speech

# Activate your virtual environment
source .venv/bin/activate

# Start training
python scripts/train_vocoder.py
```

### Option 2: Custom training configuration

```bash
python scripts/train_vocoder.py \
    --data_dir data/VCTK-Corpus-0.92 \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 2e-4 \
    --checkpoint_dir checkpoints/istft_vocoder \
    --log_dir logs/istft_vocoder \
    --num_workers 4
```

## Training Configuration

### Default Settings

- **Batch size**: 16
- **Learning rate**: 2e-4 (with exponential decay of 0.999 per epoch)
- **Optimizer**: AdamW (β1=0.9, β2=0.999, weight_decay=1e-6)
- **Loss weights**: λ_time=1.0, λ_mel=45.0, λ_stft=1.0
- **Segment length**: 16,000 samples (~0.73s at 22050 Hz)
- **Gradient clipping**: 1.0

### Training Schedule

- **Epochs**: 100 (can be adjusted)
- **Steps per epoch**: ~5,000 (depends on dataset size and batch size)
- **Total steps**: ~500,000
- **Validation**: Every 1,000 steps
- **Checkpointing**: Every 5,000 steps

### Expected Training Time

- **GPU (RTX 3090)**: ~12-24 hours for 100 epochs
- **GPU (RTX 2080)**: ~24-36 hours
- **CPU**: Not recommended (>1 week)

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir logs/istft_vocoder
```

Then open http://localhost:6006 in your browser.

### Metrics to Monitor

1. **Total Loss**: Should decrease steadily
2. **MCD (Mel Cepstral Distortion)**: Target < 6 dB (lower is better)
3. **Time-domain Loss**: L1 loss between predicted and ground truth audio
4. **Mel Loss**: L1 loss between mel-spectrograms
5. **STFT Loss**: Multi-resolution spectral loss

### Audio Samples

TensorBoard will log audio samples every 5,000 steps:
- Ground truth audio
- Reconstructed audio from vocoder

Listen to these to assess quality!

## Checkpoints

Checkpoints are saved to `checkpoints/istft_vocoder/`:

- `best_mcd.pt`: Best model based on validation MCD
- `best_loss.pt`: Best model based on validation loss
- `checkpoint_*.pt`: Periodic checkpoints every 5,000 steps
- `epoch_*.pt`: Checkpoints at the end of each epoch

### Loading a Checkpoint

To resume training from a checkpoint:

```bash
python scripts/train_vocoder.py --resume checkpoints/istft_vocoder/checkpoint_50000.pt
```

## File Structure

```
210086E-NLP_Text-to-Speech/
├── data/
│   └── VCTK-Corpus-0.92/          # VCTK dataset
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── vctk_dataset.py        # Dataset implementation
│   ├── models/
│   │   ├── istft_vocoder.py       # Vocoder model
│   │   └── vocoder_utils.py       # Loss functions and utilities
├── scripts/
│   ├── train_vocoder.py           # Main training script
│   └── quick_train.py             # Quick start script
├── checkpoints/
│   └── istft_vocoder/             # Saved checkpoints
├── logs/
│   └── istft_vocoder/             # TensorBoard logs
└── results/                        # Evaluation results
```

## Training Arguments

### Data Arguments
- `--data_dir`: Path to VCTK dataset (default: `../data/VCTK-Corpus-0.92`)
- `--cache_dir`: Directory for caching preprocessed data (optional)

### Model Arguments
- `--mel_channels`: Number of mel channels (default: 80)
- `--hidden_channels`: Hidden channels in residual blocks (default: 256)
- `--num_blocks`: Number of residual blocks (default: 6)

### Training Arguments
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 2e-4)
- `--lr_decay`: Learning rate decay per epoch (default: 0.999)
- `--grad_clip`: Gradient clipping threshold (default: 1.0)

### Loss Arguments
- `--lambda_time`: Weight for time-domain loss (default: 1.0)
- `--lambda_mel`: Weight for mel-spectrogram loss (default: 45.0)
- `--lambda_stft`: Weight for STFT loss (default: 1.0)

### Logging Arguments
- `--checkpoint_dir`: Checkpoint directory (default: `../checkpoints/istft_vocoder`)
- `--log_dir`: TensorBoard log directory (default: `../logs/istft_vocoder`)
- `--log_interval`: Logging frequency in steps (default: 100)
- `--val_interval`: Validation frequency in steps (default: 1000)
- `--checkpoint_interval`: Checkpoint save frequency (default: 5000)

### Other Arguments
- `--num_workers`: DataLoader workers (default: 4)
- `--segment_length`: Audio segment length for training (default: 16000)
- `--device`: Device to use (`cuda` or `cpu`, auto-detected)
- `--resume`: Path to checkpoint to resume from

## Troubleshooting

### Out of Memory (OOM)

If you encounter CUDA out of memory errors:

1. Reduce batch size: `--batch_size 8`
2. Reduce segment length: `--segment_length 8000`
3. Reduce model size: `--hidden_channels 192 --num_blocks 4`

### Slow Training

1. Increase `--num_workers` (but don't exceed CPU cores)
2. Use data caching: `--cache_dir cache/vctk`
3. Ensure GPU is being used (check TensorBoard or logs)

### Poor Quality

If validation MCD is not improving:

1. Check that loss is decreasing (view TensorBoard)
2. Listen to audio samples in TensorBoard
3. Try adjusting loss weights
4. Increase training duration (more epochs)

## Expected Results

After 100 epochs (~500k steps), you should achieve:

- **MCD**: < 6 dB (typical: 4-5 dB)
- **RTF**: < 0.15 (3-5× faster than HiFi-GAN)
- **Parameters**: ~2.5M (80% reduction vs HiFi-GAN)
- **Audio Quality**: Clear, intelligible speech

## Next Steps

After training:

1. **Evaluate**: Run `experiments/vocoder_testing.ipynb` with trained checkpoint
2. **Compare**: Benchmark against HiFi-GAN baseline
3. **Integrate**: Replace HiFi-GAN in VITS pipeline
4. **Multi-band**: Train multi-band extension for further improvements

## Support

If you encounter issues:

1. Check the logs in `logs/istft_vocoder/`
2. Verify VCTK dataset structure
3. Ensure all dependencies are installed
4. Check GPU memory usage with `nvidia-smi`

## Training Example

```bash
# Full training command with all options
python scripts/train_vocoder.py \
    --data_dir data/VCTK-Corpus-0.92 \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 2e-4 \
    --lr_decay 0.999 \
    --lambda_time 1.0 \
    --lambda_mel 45.0 \
    --lambda_stft 1.0 \
    --checkpoint_dir checkpoints/istft_vocoder \
    --log_dir logs/istft_vocoder \
    --val_interval 1000 \
    --checkpoint_interval 5000 \
    --num_workers 4 \
    --device cuda
```

Good luck with training! 🚀
