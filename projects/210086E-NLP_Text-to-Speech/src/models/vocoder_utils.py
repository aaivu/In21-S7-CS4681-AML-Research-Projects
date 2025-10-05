"""
Utility functions for iSTFT vocoder training and evaluation.

Author: 210086E
Date: October 2025
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Optional


def mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    sample_rate: int = 22050,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    power: float = 1.0,
    center: bool = True,
    norm: Optional[str] = "slaney",
    mel_scale: str = "slaney"
) -> torch.Tensor:
    """
    Compute mel-spectrogram from audio waveform.
    
    Args:
        audio: (B, samples) or (samples,) audio waveform
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel filterbanks
        sample_rate: Audio sample rate
        f_min: Minimum frequency
        f_max: Maximum frequency (None = sample_rate/2)
        power: Power for magnitude (1.0 = magnitude, 2.0 = power)
        center: Whether to center frames
        norm: Mel filterbank normalization
        mel_scale: Mel scale type
    
    Returns:
        mel: (B, n_mels, T) mel-spectrogram
    """
    if f_max is None:
        f_max = sample_rate / 2.0
    
    # Ensure audio is 2D (B, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Create mel filterbank
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm=norm,
        mel_scale=mel_scale
    ).to(audio.device)
    
    # Create window
    window = torch.hann_window(win_length).to(audio.device)
    
    # Compute STFT
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Compute magnitude
    spec_mag = torch.abs(spec) ** power
    
    # Apply mel filterbank
    mel = torch.matmul(mel_fb.T, spec_mag)
    
    # Convert to log scale
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    return mel


def dynamic_range_compression(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    """Apply dynamic range compression (log)."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    """Reverse dynamic range compression (exp)."""
    return torch.exp(x) / C


def normalize_audio(audio: torch.Tensor, target_level: float = -25.0) -> torch.Tensor:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Input audio waveform
        target_level: Target RMS level in dB
    
    Returns:
        Normalized audio
    """
    rms = torch.sqrt(torch.mean(audio ** 2))
    target_rms = 10 ** (target_level / 20)
    audio = audio * (target_rms / (rms + 1e-8))
    return audio


def compute_mcd(
    mel_pred: torch.Tensor,
    mel_target: torch.Tensor
) -> float:
    """
    Compute Mel Cepstral Distortion (MCD).
    
    Args:
        mel_pred: Predicted mel-spectrogram (B, n_mels, T)
        mel_target: Target mel-spectrogram (B, n_mels, T)
    
    Returns:
        MCD value in dB
    """
    # Convert to numpy if needed
    if isinstance(mel_pred, torch.Tensor):
        mel_pred = mel_pred.cpu().numpy()
    if isinstance(mel_target, torch.Tensor):
        mel_target = mel_target.cpu().numpy()
    
    # Compute MCD
    diff = mel_pred - mel_target
    mcd = np.sqrt(np.sum(diff ** 2, axis=1)).mean()
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2) * mcd
    
    return float(mcd)


class VocoderLoss(nn.Module):
    """
    Combined loss function for vocoder training.
    
    Combines time-domain, frequency-domain, and perceptual losses.
    
    Args:
        lambda_time: Weight for time-domain L1 loss
        lambda_mel: Weight for mel-spectrogram loss
        lambda_stft: Weight for multi-resolution STFT loss
    """
    
    def __init__(
        self,
        lambda_time: float = 1.0,
        lambda_mel: float = 45.0,
        lambda_stft: float = 1.0,
        mel_config: Optional[dict] = None
    ):
        super().__init__()
        
        self.lambda_time = lambda_time
        self.lambda_mel = lambda_mel
        self.lambda_stft = lambda_stft
        
        # Mel-spectrogram configuration
        self.mel_config = mel_config or {
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'n_mels': 80,
            'sample_rate': 22050
        }
        
        # Multi-resolution STFT loss
        from .istft_vocoder import MultiResolutionSTFTLoss
        self.stft_loss = MultiResolutionSTFTLoss()
    
    def forward(
        self,
        audio_pred: torch.Tensor,
        audio_target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined vocoder loss.
        
        Args:
            audio_pred: Predicted audio (B, samples)
            audio_target: Target audio (B, samples)
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        losses = {}
        
        # 1. Time-domain L1 loss
        if self.lambda_time > 0:
            time_loss = nn.functional.l1_loss(audio_pred, audio_target)
            losses['time'] = time_loss
        else:
            time_loss = 0.0
        
        # 2. Mel-spectrogram loss
        if self.lambda_mel > 0:
            mel_pred = mel_spectrogram(audio_pred, **self.mel_config)
            mel_target = mel_spectrogram(audio_target, **self.mel_config)
            mel_loss = nn.functional.l1_loss(mel_pred, mel_target)
            losses['mel'] = mel_loss
        else:
            mel_loss = 0.0
        
        # 3. Multi-resolution STFT loss
        if self.lambda_stft > 0:
            stft_loss, phase_loss = self.stft_loss(audio_pred, audio_target)
            losses['stft'] = stft_loss
            losses['phase'] = phase_loss
        else:
            stft_loss = 0.0
        
        # Combined loss
        total_loss = (
            self.lambda_time * time_loss +
            self.lambda_mel * mel_loss +
            self.lambda_stft * stft_loss
        )
        
        losses['total'] = total_loss
        
        return total_loss, losses


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in a model.
    
    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def measure_inference_time(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Measure inference time for a model.
    
    Args:
        model: Model to benchmark
        input_tensor: Sample input
        num_runs: Number of inference runs
        warmup: Number of warmup runs
        device: Device to run on
    
    Returns:
        (mean_time, std_time) in seconds
    """
    import time
    
    model = model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(input_tensor)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
    
    return np.mean(times), np.std(times)


def compute_rtf(
    model: nn.Module,
    mel_spec: torch.Tensor,
    sample_rate: int = 22050,
    hop_length: int = 256,
    device: str = "cuda"
) -> float:
    """
    Compute Real-Time Factor (RTF) for vocoder.
    
    Args:
        model: Vocoder model
        mel_spec: Input mel-spectrogram (B, n_mels, T)
        sample_rate: Audio sample rate
        hop_length: Hop length used in mel-spectrogram
        device: Device to run on
    
    Returns:
        RTF value (lower is better, <1.0 means real-time)
    """
    import time
    
    model = model.to(device)
    model.eval()
    mel_spec = mel_spec.to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(mel_spec)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Measure inference time
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        audio = model(mel_spec)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    inference_time = time.perf_counter() - start
    
    # Calculate audio duration
    audio_samples = audio.shape[-1]
    audio_duration = audio_samples / sample_rate
    
    # RTF = inference_time / audio_duration
    rtf = inference_time / audio_duration
    
    return rtf


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    filepath: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, int, int, float]:
    """
    Load training checkpoint.
    
    Returns:
        (model, epoch, step, loss)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return model, epoch, step, loss
