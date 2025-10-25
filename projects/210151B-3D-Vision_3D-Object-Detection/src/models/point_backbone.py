import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePointNet(nn.Module):
    """
    Simple point encoder: per-point MLP then global pooling to produce a point feature set.
    """
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )
 
    def forward(self, points):
        # points: (B, N, C)
        B, N, C = points.shape
        x = self.mlp(points)   # (B, N, out)
        x_max = torch.max(x, dim=1).values  # global
        # Also return per-point features if needed
        return x, x_max
