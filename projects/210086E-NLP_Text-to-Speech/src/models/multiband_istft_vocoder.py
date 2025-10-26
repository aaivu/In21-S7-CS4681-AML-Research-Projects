"""
Multi-Band iSTFT Vocoder for Fast Text-to-Speech Synthesis

This module extends the single-band iSTFT vocoder to process multiple
frequency bands in parallel for improved efficiency and specialized learning.

Author: 210086E
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .istft_vocoder import iSTFTVocoder, ResidualBlock


@dataclass
class BandConfig:
    """Configuration for a frequency band."""
    name: str
    freq_range: Tuple[int, int]  # (f_min, f_max) in Hz
    mel_range: Tuple[int, int]   # (start_idx, end_idx) for mel channels
    n_fft: int = 512
    hop_length: int = 256
    win_length: int = 512


class MultiBandiSTFTVocoder(nn.Module):
    """
    Multi-band iSTFT vocoder that processes different frequency bands in parallel.
    
    Architecture:
        Input Mel-Spectrogram (80 channels)
          ↓
        Split into frequency bands (low, mid, high)
          ↓
        Process each band with separate iSTFT vocoders
          ↓
        Combine band outputs to generate final waveform
    
    Args:
        mel_channels: Total number of mel-spectrogram channels (default: 80)
        bands: List of BandConfig objects defining frequency bands
        hidden_channels: Hidden dimension for each band vocoder (default: 128)
        num_blocks: Number of residual blocks per vocoder (default: 4)
        combination_method: How to combine bands ('learned', 'weighted', 'simple')
        share_parameters: Whether to share parameters across band vocoders
    """
    
    def __init__(
        self,
        mel_channels: int = 80,
        bands: Optional[List[BandConfig]] = None,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        combination_method: str = 'learned',
        share_parameters: bool = False
    ):
        super().__init__()
        
        self.mel_channels = mel_channels
        self.combination_method = combination_method
        self.share_parameters = share_parameters
        
        # Default 3-band configuration if not provided
        if bands is None:
            bands = self._create_default_bands()
        
        self.bands = bands
        self.num_bands = len(bands)
        
        # Validate band configuration
        self._validate_bands()
        
        # Create vocoder for each band
        if share_parameters:
            # All bands share the same vocoder architecture
            base_vocoder = iSTFTVocoder(
                mel_channels=max(b.mel_range[1] - b.mel_range[0] for b in bands),
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                n_fft=bands[0].n_fft,
                hop_length=bands[0].hop_length,
                win_length=bands[0].win_length
            )
            self.vocoders = nn.ModuleList([base_vocoder for _ in bands])
        else:
            # Each band has its own vocoder
            self.vocoders = nn.ModuleList([
                iSTFTVocoder(
                    mel_channels=band.mel_range[1] - band.mel_range[0],
                    hidden_channels=hidden_channels,
                    num_blocks=num_blocks,
                    n_fft=band.n_fft,
                    hop_length=band.hop_length,
                    win_length=band.win_length
                )
                for band in bands
            ])
        
        # Band combination module
        if combination_method == 'learned':
            self.combiner = LearnedCombiner(num_bands=self.num_bands)
        elif combination_method == 'weighted':
            # Learnable weights for each band
            self.band_weights = nn.Parameter(torch.ones(self.num_bands))
        # 'simple' method doesn't need parameters (just sum)
    
    def _create_default_bands(self) -> List[BandConfig]:
        """Create default 3-band configuration."""
        return [
            BandConfig(
                name='low',
                freq_range=(0, 4000),
                mel_range=(0, 27),
                n_fft=512,
                hop_length=256,
                win_length=512
            ),
            BandConfig(
                name='mid',
                freq_range=(4000, 8000),
                mel_range=(27, 54),
                n_fft=512,
                hop_length=256,
                win_length=512
            ),
            BandConfig(
                name='high',
                freq_range=(8000, 11025),
                mel_range=(54, 80),
                n_fft=512,
                hop_length=256,
                win_length=512
            )
        ]
    
    def _validate_bands(self):
        """Validate band configuration."""
        # Check mel range coverage
        all_mels = set()
        for band in self.bands:
            start, end = band.mel_range
            assert start < end, f"Invalid mel range for band {band.name}: {band.mel_range}"
            all_mels.update(range(start, end))
        
        # Check for overlaps or gaps (optional - can allow overlaps)
        expected_mels = set(range(self.mel_channels))
        if all_mels != expected_mels:
            print(f"⚠️ Warning: Mel channel coverage mismatch. "
                  f"Expected: 0-{self.mel_channels}, Got: {sorted(all_mels)}")
        
        # Check hop lengths are consistent
        hop_lengths = [b.hop_length for b in self.bands]
        if len(set(hop_lengths)) > 1:
            print(f"⚠️ Warning: Different hop lengths across bands: {hop_lengths}")
    
    def forward(
        self,
        mel_spec: torch.Tensor,
        return_bands: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Convert mel-spectrogram to waveform using multi-band processing.
        
        Args:
            mel_spec: (B, mel_channels, T) full mel-spectrogram
            return_bands: If True, also return individual band outputs
        
        Returns:
            audio: (B, samples) combined waveform
            If return_bands=True: (audio, band_audios)
        """
        batch_size = mel_spec.shape[0]
        band_audios = []
        
        # Process each frequency band
        for band_config, vocoder in zip(self.bands, self.vocoders):
            # Extract mel channels for this band
            start_mel, end_mel = band_config.mel_range
            mel_band = mel_spec[:, start_mel:end_mel, :]
            
            # Generate audio for this band
            with torch.no_grad() if self.training else torch.no_grad():
                audio_band = vocoder(mel_band)
            
            band_audios.append(audio_band)
        
        # Combine band outputs
        audio = self._combine_bands(band_audios)
        
        if return_bands:
            return audio, band_audios
        return audio
    
    def _combine_bands(self, band_audios: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine audio from different frequency bands.
        
        Args:
            band_audios: List of (B, samples) tensors for each band
        
        Returns:
            audio: (B, samples) combined waveform
        """
        # Ensure all bands have the same length (pad if necessary)
        max_length = max(audio.shape[-1] for audio in band_audios)
        
        padded_bands = []
        for audio in band_audios:
            if audio.shape[-1] < max_length:
                padding = max_length - audio.shape[-1]
                audio = F.pad(audio, (0, padding))
            padded_bands.append(audio)
        
        # Combine based on selected method
        if self.combination_method == 'simple':
            # Simple addition
            audio = sum(padded_bands)
        
        elif self.combination_method == 'weighted':
            # Weighted sum with learnable weights
            weights = F.softmax(self.band_weights, dim=0)
            audio = sum(w * band for w, band in zip(weights, padded_bands))
        
        elif self.combination_method == 'learned':
            # Use learned combination module
            stacked = torch.stack(padded_bands, dim=1)  # (B, n_bands, samples)
            audio = self.combiner(stacked)
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return audio
    
    def inference(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Inference mode (same as forward but clearer intent)."""
        return self.forward(mel_spec, return_bands=False)
    
    def get_num_params(self) -> Dict[str, int]:
        """Return parameter counts for each component."""
        params = {
            'total': sum(p.numel() for p in self.parameters()),
            'vocoders': sum(p.numel() for vocoder in self.vocoders for p in vocoder.parameters())
        }
        
        if hasattr(self, 'combiner'):
            params['combiner'] = sum(p.numel() for p in self.combiner.parameters())
        elif hasattr(self, 'band_weights'):
            params['band_weights'] = self.band_weights.numel()
        
        return params


class LearnedCombiner(nn.Module):
    """
    Learned module for combining multiple frequency bands.
    
    Uses 1D convolutions to mix band outputs with temporal context.
    
    Args:
        num_bands: Number of frequency bands to combine
        hidden_channels: Hidden dimension for processing
        kernel_size: Kernel size for convolutions
    """
    
    def __init__(
        self,
        num_bands: int,
        hidden_channels: int = 64,
        kernel_size: int = 15
    ):
        super().__init__()
        
        self.num_bands = num_bands
        padding = kernel_size // 2
        
        # Process each band
        self.band_conv = nn.Conv1d(
            num_bands,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Refine combination
        self.refine = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_channels, 1, kernel_size=1)
        )
    
    def forward(self, band_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_stack: (B, num_bands, samples) stacked band outputs
        
        Returns:
            audio: (B, samples) combined waveform
        """
        # Process bands
        x = self.band_conv(band_stack)  # (B, hidden, samples)
        
        # Refine and combine
        audio = self.refine(x).squeeze(1)  # (B, samples)
        
        return audio


class ParallelBandProcessor(nn.Module):
    """
    Parallel processing wrapper for multi-band vocoder.
    
    Enables true parallel execution of band vocoders using torch.jit or
    multi-threading for maximum efficiency.
    """
    
    def __init__(self, multiband_vocoder: MultiBandiSTFTVocoder):
        super().__init__()
        self.vocoder = multiband_vocoder
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Process bands in parallel (if supported by backend)."""
        # This is a placeholder for future optimization
        # Actual parallel execution would require torch.jit.fork or similar
        return self.vocoder(mel_spec)


def test_multiband_vocoder():
    """Test function for multi-band iSTFT vocoder."""
    print("Testing Multi-Band iSTFT Vocoder...")
    print("=" * 70)
    
    # Model configuration
    batch_size = 2
    mel_channels = 80
    time_steps = 100
    
    # Create dummy input
    mel_spec = torch.randn(batch_size, mel_channels, time_steps)
    
    # Test different combination methods
    for method in ['simple', 'weighted', 'learned']:
        print(f"\n{'='*70}")
        print(f"Testing combination method: {method}")
        print(f"{'='*70}")
        
        # Initialize model
        model = MultiBandiSTFTVocoder(
            mel_channels=mel_channels,
            hidden_channels=128,
            num_blocks=4,
            combination_method=method,
            share_parameters=False
        )
        
        params = model.get_num_params()
        print(f"\nModel parameters:")
        for name, count in params.items():
            print(f"   {name}: {count:,}")
        print(f"   Total size: {params['total'] * 4 / (1024**2):.2f} MB")
        
        # Test forward pass
        print(f"\nInput shape: {mel_spec.shape}")
        
        with torch.no_grad():
            audio, band_audios = model(mel_spec, return_bands=True)
        
        print(f"Output audio shape: {audio.shape}")
        print(f"Number of bands: {len(band_audios)}")
        for i, band_audio in enumerate(band_audios):
            print(f"   Band {i+1} shape: {band_audio.shape}")
        
        # Verify output
        assert audio.shape[0] == batch_size, "Batch size mismatch"
        assert audio.dim() == 2, "Audio should be 2D"
        print(f"\n✅ Test passed for {method} method")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    
    return model


def compare_single_vs_multiband():
    """Compare single-band vs multi-band vocoder."""
    print("\n" + "=" * 70)
    print("COMPARISON: Single-Band vs Multi-Band")
    print("=" * 70)
    
    mel_channels = 80
    
    # Single-band vocoder
    single_band = iSTFTVocoder(
        mel_channels=mel_channels,
        hidden_channels=256,
        num_blocks=6
    )
    
    # Multi-band vocoder
    multi_band = MultiBandiSTFTVocoder(
        mel_channels=mel_channels,
        hidden_channels=128,
        num_blocks=4,
        combination_method='learned'
    )
    
    single_params = sum(p.numel() for p in single_band.parameters())
    multi_params = sum(p.numel() for p in multi_band.parameters())
    
    print(f"\nSingle-Band Vocoder:")
    print(f"   Parameters: {single_params:,}")
    print(f"   Size: {single_params * 4 / (1024**2):.2f} MB")
    
    print(f"\nMulti-Band Vocoder (3 bands):")
    print(f"   Parameters: {multi_params:,}")
    print(f"   Size: {multi_params * 4 / (1024**2):.2f} MB")
    
    print(f"\nParameter Difference: {abs(multi_params - single_params):,}")
    print(f"Ratio: {multi_params / single_params:.2f}x")
    
    # Test inference time (rough estimate)
    mel_input = torch.randn(1, mel_channels, 200)
    
    import time
    
    # Single-band
    single_band.eval()
    with torch.no_grad():
        _ = single_band(mel_input)  # Warmup
        start = time.perf_counter()
        _ = single_band(mel_input)
        single_time = time.perf_counter() - start
    
    # Multi-band
    multi_band.eval()
    with torch.no_grad():
        _ = multi_band(mel_input)  # Warmup
        start = time.perf_counter()
        _ = multi_band(mel_input)
        multi_time = time.perf_counter() - start
    
    print(f"\nInference Time (200 time steps):")
    print(f"   Single-Band: {single_time*1000:.2f} ms")
    print(f"   Multi-Band: {multi_time*1000:.2f} ms")
    print(f"   Speedup: {single_time / multi_time:.2f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run tests
    model = test_multiband_vocoder()
    
    # Run comparison
    compare_single_vs_multiband()
