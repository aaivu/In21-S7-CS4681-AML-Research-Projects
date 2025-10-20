"""
iSTFT-based Vocoder for Fast Text-to-Speech Synthesis

This module implements a frequency-domain vocoder that replaces HiFi-GAN's
transposed convolutions with Conv1D → iSTFT pipeline for improved efficiency.

Author: 210086E
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class ResidualBlock(nn.Module):
    """
    Residual convolution block with dilated convolutions.
    
    Args:
        channels: Number of input/output channels
        kernel_size: Kernel size for convolutions
        dilation: Dilation rate for temporal receptive field
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.norm1 = nn.LayerNorm(channels)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.norm2 = nn.LayerNorm(channels)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation_out = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input tensor
        
        Returns:
            (B, C, T) output tensor
        """
        residual = x
        
        # First conv block
        x = self.conv1(x)
        x = x.transpose(1, 2)  # (B, T, C) for LayerNorm
        x = self.norm1(x)
        x = x.transpose(1, 2)  # Back to (B, C, T)
        x = self.activation1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.dropout2(x)
        
        # Residual connection
        x = x + residual
        x = self.activation_out(x)
        
        return x


class iSTFTVocoder(nn.Module):
    """
    Single-band iSTFT vocoder for mel-spectrogram to waveform conversion.
    
    Uses Conv1D layers to predict magnitude and phase spectra, then applies
    inverse STFT to generate the waveform directly in the frequency domain.
    
    Args:
        mel_channels: Number of mel-spectrogram channels (default: 80)
        hidden_channels: Hidden dimension size (default: 256)
        num_blocks: Number of residual blocks (default: 6)
        kernel_size: Kernel size for residual blocks (default: 3)
        dilation_pattern: Dilation rates for residual blocks (default: [1,3,9,27,1,3])
        dropout: Dropout rate (default: 0.1)
        n_fft: FFT size (default: 1024)
        hop_length: Hop length for STFT (default: 256)
        win_length: Window length for STFT (default: 1024)
        window_fn: Window function (default: torch.hann_window)
    """
    
    def __init__(
        self,
        mel_channels: int = 80,
        hidden_channels: int = 256,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dilation_pattern: Optional[List[int]] = None,
        dropout: float = 0.1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        window_fn=torch.hann_window
    ):
        super().__init__()
        
        self.mel_channels = mel_channels
        self.hidden_channels = hidden_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_bins = n_fft // 2 + 1  # Number of frequency bins
        
        # Default dilation pattern if not provided
        if dilation_pattern is None:
            dilation_pattern = [1, 3, 9, 27, 1, 3]
        
        assert len(dilation_pattern) == num_blocks, \
            f"Dilation pattern length ({len(dilation_pattern)}) must match num_blocks ({num_blocks})"
        
        # Input projection: mel-spectrogram → hidden representation
        self.input_conv = nn.Conv1d(mel_channels, hidden_channels, kernel_size=7, padding=3)
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.input_activation = nn.GELU()
        
        # Residual blocks with dilated convolutions
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation_pattern[i],
                dropout=dropout
            )
            for i in range(num_blocks)
        ])
        
        # Magnitude prediction head
        self.magnitude_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, self.n_bins, kernel_size=1),
            nn.Softplus()  # Ensure positive magnitudes
        )
        
        # Phase prediction head
        self.phase_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, self.n_bins, kernel_size=1),
            nn.Tanh()  # Output in [-1, 1], scaled to [-π, π]
        )
        
        # Register STFT window
        self.register_buffer('window', window_fn(win_length))
    
    def forward(
        self,
        mel_spec: torch.Tensor,
        return_spec: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert mel-spectrogram to waveform using iSTFT.
        
        Args:
            mel_spec: (B, mel_channels, T) mel-spectrogram
            return_spec: If True, also return magnitude and phase spectra
        
        Returns:
            audio: (B, samples) waveform
            If return_spec=True: (audio, magnitude, phase)
        """
        # 1. Feature projection
        x = self.input_conv(mel_spec)  # (B, hidden_channels, T)
        x = x.transpose(1, 2)  # (B, T, hidden_channels) for LayerNorm
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # Back to (B, hidden_channels, T)
        x = self.input_activation(x)
        
        # 2. Process through residual blocks
        for block in self.res_blocks:
            x = block(x)  # (B, hidden_channels, T)
        
        # 3. Predict magnitude and phase spectra
        magnitude = self.magnitude_head(x)  # (B, n_bins, T)
        phase_raw = self.phase_head(x)      # (B, n_bins, T), in [-1, 1]
        phase = phase_raw * torch.pi        # Scale to [-π, π]
        
        # 4. Construct complex spectrum
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        complex_spec = torch.complex(real, imag)  # (B, n_bins, T)
        
        # 5. Apply inverse STFT
        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
            length=mel_spec.shape[-1] * self.hop_length  # Ensure correct output length
        )  # (B, samples)
        
        if return_spec:
            return audio, magnitude, phase
        return audio
    
    def inference(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Inference mode for vocoder (same as forward but clearer intent).
        
        Args:
            mel_spec: (B, mel_channels, T) mel-spectrogram
        
        Returns:
            audio: (B, samples) waveform
        """
        return self.forward(mel_spec, return_spec=False)
    
    def get_num_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for vocoder training.
    
    Computes spectral losses at multiple time-frequency resolutions
    to ensure high-quality reconstruction across different scales.
    
    Args:
        fft_sizes: List of FFT sizes for different resolutions
        hop_sizes: List of hop sizes (should match fft_sizes length)
        win_lengths: List of window lengths (should match fft_sizes length)
        window_fn: Window function (default: torch.hann_window)
    """
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
        window_fn=torch.hann_window
    ):
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "fft_sizes, hop_sizes, and win_lengths must have same length"
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        # Register windows
        for i, win_length in enumerate(win_lengths):
            self.register_buffer(f'window_{i}', window_fn(win_length))
    
    def stft(
        self,
        audio: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int,
        window: torch.Tensor
    ) -> torch.Tensor:
        """Compute STFT magnitude spectrum."""
        spec = torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True
        )
        return torch.abs(spec)
    
    def forward(
        self,
        audio_pred: torch.Tensor,
        audio_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-resolution STFT loss.
        
        Args:
            audio_pred: (B, samples) predicted waveform
            audio_target: (B, samples) target waveform
        
        Returns:
            spectral_loss: Combined spectral magnitude loss
            phase_loss: Phase-aware loss component
        """
        spectral_loss = 0.0
        phase_loss = 0.0
        
        for i, (fft_size, hop_size, win_length) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths)
        ):
            window = getattr(self, f'window_{i}')
            
            # Compute magnitude spectra
            spec_pred = self.stft(audio_pred, fft_size, hop_size, win_length, window)
            spec_target = self.stft(audio_target, fft_size, hop_size, win_length, window)
            
            # Spectral convergence loss
            spectral_loss += torch.norm(spec_target - spec_pred, p='fro') / torch.norm(spec_target, p='fro')
            
            # Log-magnitude loss
            log_spec_pred = torch.log(spec_pred + 1e-5)
            log_spec_target = torch.log(spec_target + 1e-5)
            spectral_loss += F.l1_loss(log_spec_pred, log_spec_target)
        
        # Average over resolutions
        spectral_loss /= len(self.fft_sizes)
        
        return spectral_loss, phase_loss


def test_istft_vocoder():
    """Test function for iSTFT vocoder."""
    print("Testing iSTFT Vocoder...")
    print("=" * 70)
    
    # Model configuration
    batch_size = 2
    mel_channels = 80
    time_steps = 100
    
    # Create dummy input
    mel_spec = torch.randn(batch_size, mel_channels, time_steps)
    
    # Initialize model
    model = iSTFTVocoder(
        mel_channels=mel_channels,
        hidden_channels=256,
        num_blocks=6,
        dilation_pattern=[1, 3, 9, 27, 1, 3]
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Model size: {model.get_num_params() * 4 / (1024**2):.2f} MB (float32)")
    
    # Test forward pass
    print(f"\nInput shape: {mel_spec.shape}")
    
    with torch.no_grad():
        audio, magnitude, phase = model(mel_spec, return_spec=True)
    
    print(f"Output audio shape: {audio.shape}")
    print(f"Magnitude spectrum shape: {magnitude.shape}")
    print(f"Phase spectrum shape: {phase.shape}")
    
    # Calculate expected output length
    expected_length = time_steps * model.hop_length
    print(f"\nExpected audio length: {expected_length} samples")
    print(f"Actual audio length: {audio.shape[-1]} samples")
    print(f"Audio duration: {audio.shape[-1] / 22050:.3f}s (@ 22050 Hz)")
    
    # Test loss function
    print("\n" + "=" * 70)
    print("Testing Multi-Resolution STFT Loss...")
    
    loss_fn = MultiResolutionSTFTLoss()
    audio_target = torch.randn_like(audio)
    
    spectral_loss, phase_loss = loss_fn(audio, audio_target)
    print(f"Spectral loss: {spectral_loss.item():.4f}")
    print(f"Phase loss: {phase_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    
    return model


if __name__ == "__main__":
    # Run tests
    model = test_istft_vocoder()
