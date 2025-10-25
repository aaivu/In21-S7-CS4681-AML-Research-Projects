# iSTFT Vocoder - Usage Instructions

**Project:** 210086E-NLP_Text-to-Speech  
**Last Updated:** October 20, 2025

---

## Overview

This guide provides instructions for using the trained iSTFT Vocoder models for inference. The vocoder converts mel-spectrograms to audio waveforms using frequency-domain synthesis.

## Prerequisites

1. **Trained Model Checkpoint**: Available in `checkpoints/istft_vocoder_v2/`
2. **Python Environment**: Python 3.10 with PyTorch, torchaudio
3. **Dependencies**: All packages from `requirements.txt` installed

## Quick Start

### 1. Load a Trained Model

```python
import torch
from src.models.istft_vocoder import iSTFTVocoder

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = iSTFTVocoder(
    mel_channels=80,
    hidden_channels=256,
    num_blocks=6,
    dilation_pattern=[1, 3, 9, 27, 1, 3],
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    dropout=0.1
)

# Load trained weights (Best MCD checkpoint recommended)
checkpoint_path = 'checkpoints/istft_vocoder_v2/best_mcd.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded from {checkpoint_path}")
print(f"Training step: {checkpoint.get('global_step', 'N/A')}")
print(f"Validation MCD: {checkpoint.get('val_mcd', 'N/A'):.2f} dB")
```

### 2. Generate Audio from Mel-Spectrogram

```python
import torch
import torchaudio
import numpy as np

# Example: Load a mel-spectrogram
# Mel-spec should be shape: (1, 80, T) where T is time steps
mel_spec = torch.randn(1, 80, 100).to(device)  # Random example

# Generate audio
with torch.no_grad():
    audio = model(mel_spec)  # Output shape: (1, samples)

# Convert to numpy
audio_np = audio.squeeze().cpu().numpy()

# Save audio
sample_rate = 22050
torchaudio.save('output.wav', audio.cpu(), sample_rate)
print(f"Generated audio: {audio_np.shape[0] / sample_rate:.2f} seconds")
```

### 3. Generate from Actual Audio File

```python
from src.models.vocoder_utils import mel_spectrogram

# Load audio file
audio_path = 'path/to/audio.wav'
waveform, sr = torchaudio.load(audio_path)

# Resample if needed
if sr != 22050:
    resampler = torchaudio.transforms.Resample(sr, 22050)
    waveform = resampler(waveform)

# Extract mel-spectrogram
mel = mel_spectrogram(
    waveform,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    sample_rate=22050,
    fmin=0,
    fmax=8000
)

# Move to device and add batch dimension
mel = mel.to(device)
if mel.dim() == 2:
    mel = mel.unsqueeze(0)  # (1, 80, T)

# Generate reconstructed audio
with torch.no_grad():
    audio_recon = model(mel)

# Save reconstructed audio
torchaudio.save('reconstructed.wav', audio_recon.cpu(), 22050)
print("Audio reconstructed and saved!")
```

## Advanced Usage

### Batch Processing

```python
# Process multiple mel-spectrograms at once
mel_batch = torch.randn(4, 80, 100).to(device)  # Batch of 4

with torch.no_grad():
    audio_batch = model(mel_batch)  # Shape: (4, samples)

# Save each audio
for i, audio in enumerate(audio_batch):
    torchaudio.save(f'output_{i}.wav', audio.unsqueeze(0).cpu(), 22050)
```

### Real-Time Synthesis

```python
import time

# Measure inference time
mel = torch.randn(1, 80, 100).to(device)

# Warmup
with torch.no_grad():
    _ = model(mel)

# Benchmark
num_runs = 100
start_time = time.time()

with torch.no_grad():
    for _ in range(num_runs):
        audio = model(mel)
        
end_time = time.time()

avg_time = (end_time - start_time) / num_runs
audio_duration = mel.shape[-1] * 256 / 22050  # hop_length * T / sr
rtf = avg_time / audio_duration

print(f"Average inference time: {avg_time*1000:.2f} ms")
print(f"Audio duration: {audio_duration:.2f} seconds")
print(f"Real-Time Factor (RTF): {rtf:.3f}")
print(f"Real-time capable: {'Yes ✓' if rtf < 1.0 else 'No ✗'}")
```

### Using Different Checkpoints

```python
# Best MCD (Recommended for quality)
checkpoint_mcd = 'checkpoints/istft_vocoder_v2/best_mcd.pt'

# Best Loss (Best reconstruction)
checkpoint_loss = 'checkpoints/istft_vocoder_v2/best_loss.pt'

# Specific epoch
checkpoint_epoch = 'checkpoints/istft_vocoder_v2/epoch_100.pt'

# Load any checkpoint
def load_vocoder(checkpoint_path, device='cuda'):
    model = iSTFTVocoder(
        mel_channels=80,
        hidden_channels=256,
        num_blocks=6,
        dilation_pattern=[1, 3, 9, 27, 1, 3]
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint

# Example
model, ckpt = load_vocoder(checkpoint_mcd, device='cuda')
print(f"Loaded checkpoint from step {ckpt['global_step']}")
```

## Evaluation

### Compute Quality Metrics

```python
from src.models.vocoder_utils import compute_mcd

# Load ground truth audio
audio_gt, _ = torchaudio.load('ground_truth.wav')

# Extract mel from ground truth
mel_gt = mel_spectrogram(audio_gt, ...)

# Generate audio
with torch.no_grad():
    audio_pred = model(mel_gt.unsqueeze(0).to(device))

# Compute MCD
mcd = compute_mcd(audio_pred.cpu(), audio_gt)
print(f"Mel-Cepstral Distortion: {mcd:.2f} dB")

# Compute SNR
def compute_snr(pred, target):
    noise = pred - target
    signal_power = torch.mean(target ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

snr = compute_snr(audio_pred.cpu().squeeze(), audio_gt.squeeze())
print(f"Signal-to-Noise Ratio: {snr:.2f} dB")
```

### Batch Evaluation on Test Set

```python
from torch.utils.data import DataLoader
from src.data.vctk_dataset import VCTKVocoderDataset
from tqdm import tqdm

# Load test dataset
test_dataset = VCTKVocoderDataset(
    data_dir='data/VCTK-Corpus-0.92',
    split='test',
    segment_length=None  # Use full utterances
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate
mcds = []
snrs = []

model.eval()
with torch.no_grad():
    for mel, audio_gt in tqdm(test_loader, desc="Evaluating"):
        mel = mel.to(device)
        audio_pred = model(mel)
        
        # Compute metrics
        mcd = compute_mcd(audio_pred.cpu(), audio_gt)
        snr = compute_snr(audio_pred.cpu().squeeze(), audio_gt.squeeze())
        
        mcds.append(mcd)
        snrs.append(snr)

# Print results
print(f"\nTest Set Results:")
print(f"MCD: {np.mean(mcds):.2f} ± {np.std(mcds):.2f} dB")
print(f"SNR: {np.mean(snrs):.2f} ± {np.std(snrs):.2f} dB")
```

## Complete Example Script

Here's a complete script that demonstrates end-to-end usage:

```python
"""
Complete example: Load model, generate audio, evaluate quality
"""
import torch
import torchaudio
from pathlib import Path
from src.models.istft_vocoder import iSTFTVocoder
from src.models.vocoder_utils import mel_spectrogram, compute_mcd

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/istft_vocoder_v2/best_mcd.pt'
    audio_path = 'data/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_001_mic2.flac'
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = iSTFTVocoder(
        mel_channels=80,
        hidden_channels=256,
        num_blocks=6,
        dilation_pattern=[1, 3, 9, 27, 1, 3]
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded (step {checkpoint['global_step']})")
    
    # Load audio
    print(f"Loading audio from {audio_path}...")
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample to 22050 Hz
    if sr != 22050:
        resampler = torchaudio.transforms.Resample(sr, 22050)
        waveform = resampler(waveform)
    
    # Extract mel-spectrogram
    print("Extracting mel-spectrogram...")
    mel = mel_spectrogram(
        waveform,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        sample_rate=22050,
        fmin=0,
        fmax=8000
    )
    
    # Generate audio
    print("Generating audio...")
    with torch.no_grad():
        mel_input = mel.unsqueeze(0).to(device)
        audio_recon = model(mel_input)
    
    # Save outputs
    output_audio = output_dir / 'reconstructed.wav'
    torchaudio.save(str(output_audio), audio_recon.cpu(), 22050)
    print(f"Saved reconstructed audio to {output_audio}")
    
    # Evaluate quality
    print("\nEvaluating quality...")
    mcd = compute_mcd(audio_recon.cpu(), waveform)
    print(f"Mel-Cepstral Distortion: {mcd:.2f} dB")
    
    # Compute SNR
    noise = audio_recon.cpu().squeeze() - waveform.squeeze()
    signal_power = torch.mean(waveform.squeeze() ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)
    print(f"Signal-to-Noise Ratio: {snr:.2f} dB")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
```

Save this as `scripts/inference_example.py` and run:
```bash
python scripts/inference_example.py
```

## Model Specifications

### iSTFT Vocoder V2

- **Parameters:** 2,500,000 (2.5M)
- **Model Size:** ~10 MB
- **Input:** Mel-spectrogram (80 channels)
- **Output:** Waveform (22050 Hz sample rate)
- **Inference Time:** ~7ms per utterance (GPU)
- **Real-Time Factor:** 0.22 (4-5× faster than real-time)
- **Quality:** MCD 5.21 dB, SNR 18.9 dB

### Checkpoints Available

| Checkpoint | Description | MCD (dB) | Notes |
|------------|-------------|----------|-------|
| `best_mcd.pt` | Best perceptual quality | 5.21 | **Recommended** |
| `best_loss.pt` | Best reconstruction loss | 5.34 | Alternative |
| `epoch_*.pt` | Epoch checkpoints | Varies | Training progress |
| `checkpoint_*.pt` | Step checkpoints | Varies | Regular saves |

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues:

```python
# Process in smaller batches
batch_size = 1  # Reduce batch size

# Or use CPU
device = torch.device('cpu')
```

### Audio Quality Issues

If generated audio has artifacts:

1. **Use Best MCD checkpoint** (`best_mcd.pt`)
2. **Check input mel-spectrogram** - Ensure proper normalization
3. **Verify sample rate** - Should be 22050 Hz
4. **Check mel extraction parameters** - Must match training config

### Slow Inference

To improve speed:

```python
# Enable inference optimizations
torch.backends.cudnn.benchmark = True

# Use half precision (if supported)
model = model.half()
mel = mel.half()
```

## Integration with VITS

To integrate this vocoder into a full TTS pipeline:

```python
# 1. Text-to-Mel model (e.g., VITS encoder)
text_input = "Hello world"
mel_spec = text_to_mel_model(text_input)

# 2. Mel-to-Audio (our vocoder)
audio = vocoder(mel_spec)

# 3. Save result
torchaudio.save('tts_output.wav', audio.cpu(), 22050)
```

## Performance Benchmarks

See `results/istft_vocoder_v2_evaluation_report.md` for comprehensive benchmarks including:
- Quality metrics (MCD, SNR, PESQ, STOI)
- Speed measurements (RTF, inference time)
- Comparison with baselines
- Detailed error analysis

## References

- **Architecture:** `docs/istft_vocoder_design.md`
- **Training:** `docs/training_guide.md`
- **Evaluation:** `results/istft_vocoder_v2_evaluation_report.md`
- **Source Code:** `src/models/istft_vocoder.py`

---

**Last Updated:** October 20, 2025  
**Author:** 210086E