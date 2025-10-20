# iSTFT Vocoder Architecture Design

**Date:** October 5, 2025  
**Project:** 210086E-NLP_Text-to-Speech  
**Phase:** 3 - Vocoder Optimization

---

## 1. Motivation

### 1.1 HiFi-GAN Bottleneck
Based on profiling analysis, the HiFi-GAN vocoder in VITS exhibits the following issues:
- **High computational cost**: Multiple transposed convolution layers
- **Sequential processing**: Limited parallelization opportunities
- **Memory intensive**: Large intermediate feature maps during upsampling
- **Performance bottleneck**: Accounts for 40-60% of total inference time

### 1.2 FLY-TTS Approach
The FLY-TTS paper demonstrates that frequency-domain synthesis can achieve:
- **4-8× speedup** over traditional neural vocoders
- **Reduced parameters** by eliminating upsampling layers
- **Maintained quality** with proper network design
- **Better parallelization** through frequency-domain operations

---

## 2. iSTFT Vocoder Architecture

iSTFT stands for Inverse Short-Time Fourier Transform. It's the reverse operation of the STFT (Short-Time Fourier Transform).

### 2.1 High-Level Overview

```
Input: Mel-Spectrogram (B, mel_dim, T)
    ↓
┌──────────────────────────┐
│  Feature Projection      │  Conv1d: mel_dim → hidden_dim
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│  Residual Conv Blocks    │  Stack of Conv1d blocks with residual connections
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│  Spectrum Prediction     │
│  ┌──────────────────┐   │
│  │ Magnitude Head   │───┼──→ Magnitude Spectrum (B, n_fft//2+1, T)
│  └──────────────────┘   │
│  ┌──────────────────┐   │
│  │ Phase Head       │───┼──→ Phase Spectrum (B, n_fft//2+1, T)
│  └──────────────────┘   │
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│  Complex Spectrum        │  Combine magnitude & phase
│  Mag * exp(j * Phase)    │
└──────────────────────────┘
    ↓
┌──────────────────────────┐
│  Inverse STFT (iSTFT)    │  torch.istft()
└──────────────────────────┘
    ↓
Output: Waveform (B, samples)
```

### 2.2 Component Details

#### 2.2.1 Feature Projection Layer
```python
# Project mel-spectrogram to hidden dimension
Conv1d(mel_channels, hidden_channels, kernel_size=7, padding=3)
BatchNorm1d / LayerNorm
Activation (LeakyReLU or GELU)
```

**Purpose:** Transform mel-spectrogram features to a rich representation space.

#### 2.2.2 Residual Convolution Blocks
```python
# Stack of N residual blocks
ResidualBlock:
    Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=d)
    Normalization
    Activation
    Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=d)
    Normalization
    + Residual Connection
    Activation
```

**Configuration:**
- Number of blocks: 4-8
- Dilation pattern: [1, 3, 9, 27] or [1, 2, 4, 8]
- Hidden channels: 256-512

**Purpose:** Extract temporal features with increasing receptive fields.

#### 2.2.3 Spectrum Prediction Heads

**Magnitude Head:**
```python
Conv1d(hidden_channels, hidden_channels, kernel_size=3)
Activation
Conv1d(hidden_channels, n_fft//2 + 1, kernel_size=1)
Softplus() or Exp()  # Ensure positive values
```

**Phase Head:**
```python
Conv1d(hidden_channels, hidden_channels, kernel_size=3)
Activation  
Conv1d(hidden_channels, n_fft//2 + 1, kernel_size=1)
Tanh() * π  # Constrain to [-π, π]
```

**Purpose:** Predict magnitude and phase spectra for iSTFT reconstruction.

#### 2.2.4 STFT Parameters

```python
n_fft = 1024          # FFT size
hop_length = 256      # Hop size (matches mel-spectrogram)
win_length = 1024     # Window length
window = 'hann'       # Window function
```

**Upsampling Factor:** hop_length = 256 → each time step generates 256 audio samples

---

## 3. Single-Band Implementation (Phase 1)

### 3.1 Network Architecture

```python
class iSTFTVocoder(nn.Module):
    def __init__(
        self,
        mel_channels=80,
        hidden_channels=256,
        num_blocks=6,
        kernel_size=3,
        dilation_pattern=[1, 3, 9, 27, 1, 3],
        n_fft=1024,
        hop_length=256,
        win_length=1024
    ):
        # Architecture definition
```

### 3.2 Forward Pass

```python
def forward(self, mel_spec):
    """
    Args:
        mel_spec: (B, mel_channels, T) mel-spectrogram
    
    Returns:
        audio: (B, samples) waveform
    """
    # 1. Feature projection
    x = self.input_conv(mel_spec)
    
    # 2. Residual blocks
    for block in self.res_blocks:
        x = block(x)
    
    # 3. Predict spectra
    magnitude = self.magnitude_head(x)  # (B, n_fft//2+1, T)
    phase = self.phase_head(x)          # (B, n_fft//2+1, T)
    
    # 4. Construct complex spectrum
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    complex_spec = torch.complex(real, imag)
    
    # 5. Inverse STFT
    audio = torch.istft(
        complex_spec,
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        win_length=self.win_length,
        window=self.window,
        return_complex=False
    )
    
    return audio
```

### 3.3 Loss Functions

#### Reconstruction Losses
```python
# 1. Time-domain loss
L_time = L1(audio_pred, audio_gt)

# 2. Frequency-domain loss
mel_pred = mel_spectrogram(audio_pred)
mel_gt = mel_spectrogram(audio_gt)
L_mel = L1(mel_pred, mel_gt)

# 3. Multi-resolution STFT loss
L_stft = sum(L1(STFT_i(audio_pred), STFT_i(audio_gt)) for i in scales)
```

#### Total Loss
```python
L_total = λ_time * L_time + λ_mel * L_mel + λ_stft * L_stft
```

**Suggested weights:** λ_time=1.0, λ_mel=45.0, λ_stft=1.0

---

## 4. Multi-Band Extension (Phase 2)

### 4.1 Motivation
- **Parallel processing**: Process different frequency bands independently
- **Specialized learning**: Different networks for low/mid/high frequencies
- **Reduced complexity**: Each sub-band has fewer frequencies

### 4.2 Architecture

```
Input: Mel-Spectrogram (B, 80, T)
    ↓
┌──────────────────────────────────────────┐
│         Frequency Band Splitting          │
├──────────────┬──────────────┬────────────┤
│   Low Band   │   Mid Band   │  High Band │
│   0-4 kHz    │   4-8 kHz    │  8-11 kHz  │
└──────┬───────┴──────┬───────┴──────┬─────┘
       ↓              ↓              ↓
  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ iSTFT-L │   │ iSTFT-M │   │ iSTFT-H │
  │ Vocoder │   │ Vocoder │   │ Vocoder │
  └────┬────┘   └────┬────┘   └────┬────┘
       ↓              ↓              ↓
     Low           Mid           High
    Audio         Audio          Audio
       └──────────┬──────────┘
                  ↓
           ┌─────────────┐
           │   Combine   │
           └─────────────┘
                  ↓
           Final Audio
```

### 4.3 Band Configuration

```python
# Example: 3-band split
bands = [
    {
        'name': 'low',
        'freq_range': (0, 4000),      # Hz
        'n_fft': 512,
        'hop_length': 256,
        'mel_range': (0, 27)          # Mel channels 0-26
    },
    {
        'name': 'mid',
        'freq_range': (4000, 8000),
        'n_fft': 512,
        'hop_length': 256,
        'mel_range': (27, 54)         # Mel channels 27-53
    },
    {
        'name': 'high',
        'freq_range': (8000, 11025),
        'n_fft': 512,
        'hop_length': 256,
        'mel_range': (54, 80)         # Mel channels 54-79
    }
]
```

### 4.4 Multi-Band Forward Pass

```python
class MultiBandiSTFTVocoder(nn.Module):
    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (B, 80, T) full mel-spectrogram
        
        Returns:
            audio: (B, samples) combined waveform
        """
        band_audios = []
        
        # Process each band in parallel
        for band_config, vocoder in zip(self.bands, self.vocoders):
            # Extract band-specific mels
            mel_band = mel_spec[:, band_config['mel_range'][0]:band_config['mel_range'][1], :]
            
            # Generate audio for this band
            audio_band = vocoder(mel_band)
            band_audios.append(audio_band)
        
        # Combine bands
        audio = self.combine_bands(band_audios)
        
        return audio
```

### 4.5 Band Combination Methods

**Method 1: Simple Addition**
```python
audio = sum(band_audios)
```

**Method 2: Weighted Sum**
```python
audio = sum(w * band for w, band in zip(weights, band_audios))
```

**Method 3: Learned Combination**
```python
# Concatenate and use 1x1 conv to mix
stacked = torch.stack(band_audios, dim=1)  # (B, n_bands, samples)
audio = conv1d(stacked).squeeze(1)         # (B, samples)
```

---

## 5. Training Strategy

### 5.1 Single-Band Training

```python
# Phase 1: Train from scratch
optimizer = AdamW(model.parameters(), lr=2e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in epochs:
    for mel, audio_gt in dataloader:
        # Forward
        audio_pred = model(mel)
        
        # Loss
        loss = compute_loss(audio_pred, audio_gt)
        
        # Backward
        loss.backward()
        optimizer.step()
```

### 5.2 Multi-Band Training

```python
# Phase 2a: Initialize sub-vocoders from single-band weights
for band_vocoder in multi_band_model.vocoders:
    band_vocoder.load_state_dict(single_band_model.state_dict(), strict=False)

# Phase 2b: Fine-tune with frozen band vocoders, train combiner
for param in multi_band_model.vocoders.parameters():
    param.requires_grad = False

# Phase 2c: Full fine-tuning
for param in multi_band_model.parameters():
    param.requires_grad = True
```

### 5.3 Data Requirements

**Option 1: Train on VCTK directly**
- Use ground truth mel-spectrograms + audio pairs
- Requires full training (~50-100k iterations)

**Option 2: Knowledge distillation from HiFi-GAN**
- Use HiFi-GAN to generate pseudo-ground-truth
- Faster training, maintains quality

---

## 6. Evaluation Metrics

### 6.1 Quality Metrics
- **Mel Cepstral Distortion (MCD)**: Spectral similarity
- **PESQ**: Perceptual quality
- **STOI**: Intelligibility
- **MOS**: Subjective quality (if feasible)

### 6.2 Efficiency Metrics
- **RTF (Real-Time Factor)**: On CPU and GPU
- **Latency**: Per-utterance synthesis time
- **FLOPs**: Theoretical computational cost
- **Memory Usage**: Peak GPU/RAM consumption

### 6.3 Comparison Baseline
- **HiFi-GAN**: Original vocoder in VITS
- **Target**: 3-5× speedup with <5% quality degradation

---

## 7. Implementation Phases

### Phase 1: Single-Band iSTFT Vocoder (Week 1-2)
✅ Design architecture  
⏳ Implement `iSTFTVocoder` class  
⏳ Create training script  
⏳ Train and evaluate on VCTK  
⏳ Benchmark vs HiFi-GAN  

### Phase 2: Multi-Band Extension (Week 3)
⏳ Design band configuration  
⏳ Implement `MultiBandiSTFTVocoder` class  
⏳ Train multi-band model  
⏳ Optimize band combination  

### Phase 3: Integration & Optimization (Week 4)
⏳ Integrate into VITS pipeline  
⏳ End-to-end testing  
⏳ Performance optimization  
⏳ Documentation & reporting  

---

## 8. Expected Outcomes

### 8.1 Performance Improvements
- **RTF Target**: <0.15 on CPU (vs 0.4-0.6 for HiFi-GAN)
- **Speedup**: 3-5× faster inference
- **Parameters**: 30-50% reduction

### 8.2 Quality Preservation
- **MCD**: Within 0.3 dB of baseline
- **PESQ**: >4.0 (similar to HiFi-GAN)
- **STOI**: >0.95

### 8.3 Trade-offs
- May require more training iterations
- Initial quality might be lower than HiFi-GAN
- Phase prediction can be challenging

---

## 9. References

1. **FLY-TTS** - Fast Lightweight TTS using frequency domain synthesis
2. **HiFi-GAN** - Original vocoder architecture in VITS
3. **VITS** - End-to-end TTS with VAE and adversarial training
4. **Multi-band MelGAN** - Multi-band processing for neural vocoders

---

## 10. Risk Mitigation

### Risk 1: Poor phase prediction
**Mitigation:**
- Start with magnitude-only prediction (Griffin-Lim phase)
- Gradually introduce phase prediction
- Use perceptual loss functions

### Risk 2: Training instability
**Mitigation:**
- Careful learning rate tuning
- Gradient clipping
- Multi-scale discriminators

### Risk 3: Quality degradation
**Mitigation:**
- Knowledge distillation from HiFi-GAN
- Multi-objective loss functions
- Careful hyperparameter tuning

---

**Status:** Design Complete ✅  
**Next:** Implementation Phase →
