import torch
import torch.nn as nn

class SimpleRoIHead(nn.Module):
    def __init__(self, in_channels=64, hidden=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.cls = nn.Linear(hidden, num_classes)  # classification logits
        self.reg = nn.Linear(hidden, 7)  # 3D box reg: x,y,z,w,l,h,theta

    def forward(self, x):
        feat = self.net(x)
        return self.cls(feat), self.reg(feat)
