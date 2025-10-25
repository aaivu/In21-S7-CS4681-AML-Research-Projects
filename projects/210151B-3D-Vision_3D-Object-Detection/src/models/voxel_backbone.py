import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVoxelEncoder(nn.Module):
    """
    Very small voxel encoder:
    - Convert points to a fixed-size voxel grid by pooling
    - Apply small 3D conv stack (dense)
    This is inefficient for huge scenes but keeps installation simple.
    """
    def __init__(self, grid_size=(32,32,8), in_channels=4, out_channels=64):
        super().__init__()
        gx, gy, gz = grid_size
        self.grid_size = grid_size
        # we will flatten voxel occupancy / features into N_vox x C then use conv3d
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool3d((4,4,2))  # reduce spatial dims
        )

    def forward(self, voxel_features):
        # voxel_features expected as tensor shape (B, C, X, Y, Z)
        return self.net(voxel_features)  # -> (B, out_channels, 4,4,2)
