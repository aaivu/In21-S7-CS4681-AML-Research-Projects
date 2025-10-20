import torch

def stft(x, n_fft=1200, hop_length=600, win_length=None, window=None):
    """Compute STFT for 1D, 2D, or 3D audio tensors with Hann window."""
    # Accept shapes: [T], [B,T], [B,C,T]
    if x.dim() == 3:
        x = x.view(-1, x.size(-1))
    elif x.dim() == 1:
        x = x.unsqueeze(0)
    if window is None:
        window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

def istft(spec, n_fft=1200, hop_length=600, win_length=None, window=None):
    """Inverse STFT for 1D, 2D, or 3D complex specs."""
    if spec.dim() == 3:
        spec = spec.view(-1, spec.size(-2), spec.size(-1))
    if window is None:
        window = torch.hann_window(n_fft, device=spec.device)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length or n_fft,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
    )
