import torch
import torch.nn as nn

class SpectralCompression(nn.Module):
    """
    Compress along the frequency axis.
    Input:  x  [B, F, T]
    Output: y  [B, Fc, T]
    """
    def __init__(self, n_bins=601, compressed_bins=256, fixed_bins=64):
        super().__init__()
        assert compressed_bins >= fixed_bins, "compressed_bins must be >= fixed_bins"
        self.fixed_bins = fixed_bins
        self.in_bins = n_bins
        self.out_bins = compressed_bins
        self.learnable = nn.Linear(n_bins - fixed_bins, compressed_bins - fixed_bins)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B,F,T], got {x.shape}")
        B, F, T = x.shape
        if F != self.in_bins:
            raise ValueError(f"Expected F={self.in_bins} bins, got {F}")

        low  = x[:, :self.fixed_bins, :]                 # [B, fixed, T]
        high = x[:, self.fixed_bins:, :]                 # [B, F-fixed, T]
        high = high.transpose(1, 2)                      # [B, T, F-fixed]
        high = self.learnable(high)                      # [B, T, Fc-fixed]
        high = high.transpose(1, 2)                      # [B, Fc-fixed, T]
        y = torch.cat([low, high], dim=1)                # [B, Fc, T]
        return y
