import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Voxel Backbone
# ------------------------
class VoxelBackbone(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4,4,4))  # reduce spatial dims
        )

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        return self.net(x).flatten(start_dim=1)  # -> [B, out_channels*4*4*4]

# ------------------------
# Point Backbone
# ------------------------
class PointBackbone(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x):
        # x shape: [B, N, C]
        point_feats = self.mlp(x)
        global_feats = torch.max(point_feats, dim=1).values
        return global_feats  # [B, out_channels]

# ------------------------
# ROI Head
# ------------------------
class RoIHead(nn.Module):
    def __init__(self, in_channels, hidden=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.cls = nn.Linear(hidden, num_classes)
        self.reg = nn.Linear(hidden, 7)  # x,y,z,w,l,h,theta

    def forward(self, x):
        feat = self.net(x)
        return self.cls(feat), self.reg(feat)

# ------------------------
# PV-RCNN++ Model
# ------------------------
class PVRCNNPP(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.voxel_backbone = VoxelBackbone(in_channels=4, out_channels=64)
        self.point_backbone = PointBackbone(in_channels=4, out_channels=64)
        self.roi_head = RoIHead(in_channels=64*4*4*4 + 64, hidden=128, num_classes=num_classes)

    def forward(self, voxel_input, point_input=None):
        """
        voxel_input: [B, C, D, H, W]
        point_input: [B, N, 4] optional
        """
        voxel_feats = self.voxel_backbone(voxel_input)  # [B, voxel_feat_dim]
        if point_input is not None:
            point_feats = self.point_backbone(point_input)
            combined_feats = torch.cat([voxel_feats, point_feats], dim=1)
        else:
            combined_feats = voxel_feats

        cls_logits, box_reg = self.roi_head(combined_feats)
        return cls_logits, box_reg
