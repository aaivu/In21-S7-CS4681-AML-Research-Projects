# omniq/models/fusion_mamba.py
from typing import Optional
import torch
import torch.nn as nn

def exists(x): return x is not None

class GatedConvMixer(nn.Module):
    """
    Lightweight fallback if mamba-ssm isn't available.
    Sequence mixer in O(L) using depthwise 1D conv + GLU gating.
    x: (B, L, D) -> (B, L, D)
    """
    def __init__(self, d_model: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(d_model, d_model, kernel_size,
                            padding=pad, dilation=dilation, groups=d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        u = self.pw1(x)                               # (B,L,2D)
        a, g = u.chunk(2, dim=-1)                     # gates
        v = self.dw(x.transpose(1, 2)).transpose(1, 2)  # depthwise conv over L
        y = torch.sigmoid(g) * (a + v)
        y = self.pw2(y)
        return self.dropout(y)

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba (if available) + FFN, with residuals.
    Falls back to GatedConvMixer if mamba-ssm is not installed.
    """
    def __init__(self, d_model: int, d_state: int = 128, expand: int = 2,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional

        self.use_mamba = False
        try:
            from mamba_ssm import Mamba
            self.fwd = Mamba(d_model=d_model, d_state=d_state, expand=expand)
            self.bwd = Mamba(d_model=d_model, d_state=d_state, expand=expand) if bidirectional else None
            self.use_mamba = True
        except Exception:
            self.fwd = GatedConvMixer(d_model, kernel_size=7, dilation=1, dropout=dropout)
            self.bwd = GatedConvMixer(d_model, kernel_size=7, dilation=2, dropout=dropout) if bidirectional else None

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        y_f = self.fwd(x)
        if self.bwd is not None:
            y_b = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
            y = 0.5 * (y_f + y_b)
        else:
            y = y_f
        x = x + self.dropout(y)
        x = x + self.ff(x)
        return x

class FusionMamba(nn.Module):
    """
    Stacks N BiMamba blocks.
    """
    def __init__(self, d_model: int, depth: int = 2, d_state: int = 128,
                 expand: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model, d_state, expand, dropout, bidirectional)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
