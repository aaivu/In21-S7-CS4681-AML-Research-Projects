from typing import Optional
import torch
import torch.nn as nn
import timm

class Swin2DTemporalAvg(nn.Module):
    """
    Wraps a 2D Swin model to handle video by per-frame encode + temporal average.
    Input: x (B, C, T, H, W)
    """
    def __init__(self, backbone_name: str = "swin_tiny_patch4_window7_224",
                 num_classes: int = 101, pretrained: bool = True):
        super().__init__()
        # Make Swin output features (no classifier), global pooled
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                          num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T, H, W) -> (B*T, C, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        feats = self.backbone(x)          # (B*T, D)
        feats = feats.reshape(B, T, -1).mean(dim=1)  # temporal avg
        logits = self.head(feats)         # (B, num_classes)
        return logits
