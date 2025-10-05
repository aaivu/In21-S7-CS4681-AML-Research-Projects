# Quick Start Guide: iSTFT Vocoder

**Project:** 210086E-NLP_Text-to-Speech  
**Phase 3:** Vocoder Optimization

---

## Installation

```bash
# Navigate to project directory
cd projects/210086E-NLP_Text-to-Speech

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Additional dependencies for vocoder
pip install torch torchaudio soundfile
```

---

## Running the Notebooks

### 1. Analyze HiFi-GAN Bottleneck

```bash
# Open Jupyter Notebook
jupyter notebook experiments/vocoder_bottleneck_analysis.ipynb
```

**What it does:**
- Analyzes HiFi-GAN architecture
- Counts transposed convolutions
- Measures vocoder-only inference time
- Visualizes upsampling chain
- Calculates computational cost (GFLOPs)

**Expected runtime:** 5-10 minutes

---

### 2. Test iSTFT Vocoder

```bash
jupyter notebook experiments/vocoder_testing.ipynb
```

**What it does:**
- Unit tests for iSTFT vocoder
- Reconstruction quality evaluation
- Performance benchmarking
- Quality metrics (MCD, RTF)
- Audio playback and visualization

**Expected runtime:** 10-15 minutes

---

## Using the Vocoder in Python

### Single-Band Example

```python
import torch
from src.models.istft_vocoder import iSTFTVocoder

# Initialize vocoder
vocoder = iSTFTVocoder(
    mel_channels=80,
    hidden_channels=256,
    num_blocks=6,
    dilation_pattern=[1, 3, 9, 27, 1, 3]
)

# Load or create mel-spectrogram
mel_spec = torch.randn(1, 80, 100)  # (batch, mels, time)

# Generate audio
vocoder.eval()
with torch.no_grad():
    audio = vocoder(mel_spec)  # (batch, samples)

print(f"Generated audio shape: {audio.shape}")
```

### Multi-Band Example

```python
from src.models.multiband_istft_vocoder import MultiBandiSTFTVocoder

# Initialize multi-band vocoder
vocoder = MultiBandiSTFTVocoder(
    mel_channels=80,
    hidden_channels=128,
    num_blocks=4,
    combination_method='learned'
)

# Generate audio with band outputs
vocoder.eval()
with torch.no_grad():
    audio, band_audios = vocoder(mel_spec, return_bands=True)

print(f"Combined audio: {audio.shape}")
print(f"Number of bands: {len(band_audios)}")
```

---

## Testing the Implementation

### Run Unit Tests

```python
# Test single-band vocoder
python src/models/istft_vocoder.py

# Test multi-band vocoder
python src/models/multiband_istft_vocoder.py
```

### Expected Output

```
Testing iSTFT Vocoder...
======================================================================
Model parameters: 2,545,793
Model size: 9.71 MB (float32)

Input shape: torch.Size([2, 80, 100])
Output audio shape: torch.Size([2, 25600])
Magnitude spectrum shape: torch.Size([2, 513, 100])
Phase spectrum shape: torch.Size([2, 513, 100])

✅ All tests passed!
```

---

## Computing Quality Metrics

```python
from src.models.vocoder_utils import (
    mel_spectrogram,
    compute_mcd,
    compute_rtf
)

# Load ground truth audio
audio_gt = torch.randn(1, 22050)  # 1 second @ 22050 Hz

# Generate mel-spectrogram
mel = mel_spectrogram(audio_gt, n_fft=1024, hop_length=256, n_mels=80)

# Reconstruct audio
audio_pred = vocoder(mel)

# Compute MCD
mel_gt = mel_spectrogram(audio_gt, n_fft=1024, hop_length=256, n_mels=80)
mel_pred = mel_spectrogram(audio_pred, n_fft=1024, hop_length=256, n_mels=80)
mcd = compute_mcd(mel_pred, mel_gt)
print(f"MCD: {mcd:.3f} dB")

# Compute RTF
rtf = compute_rtf(vocoder, mel, sample_rate=22050, hop_length=256)
print(f"RTF: {rtf:.4f}")
```

---

## File Structure

```
210086E-NLP_Text-to-Speech/
├── src/
│   └── models/
│       ├── istft_vocoder.py              # Single-band vocoder
│       ├── multiband_istft_vocoder.py    # Multi-band vocoder
│       └── vocoder_utils.py              # Utilities
├── experiments/
│   ├── vocoder_bottleneck_analysis.ipynb # HiFi-GAN analysis
│   └── vocoder_testing.ipynb             # Testing & benchmarking
├── docs/
│   ├── istft_vocoder_design.md           # Architecture design
│   ├── phase3_summary.md                 # Implementation summary
│   └── vocoder_quickstart.md             # This file
└── results/
    ├── vocoder_test/                     # Test outputs
    └── Component-Specific Profiling/     # Analysis results
```

---

## Expected Results (After Training)

### Quality Metrics
- **MCD:** <6.0 dB (good quality)
- **Spectral Convergence:** <0.3
- **PESQ:** >4.0

### Performance Metrics
- **RTF (GPU):** <0.05 (very fast)
- **RTF (CPU):** <0.15 (real-time capable)
- **Speedup vs HiFi-GAN:** 3-5×

---

**Happy Experimenting! 🚀**
