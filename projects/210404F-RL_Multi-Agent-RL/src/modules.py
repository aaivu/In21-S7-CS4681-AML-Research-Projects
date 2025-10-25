# modules.py
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, out_act=None):
    """
    Build a simple MLP: sizes = [in, h1, h2, ..., out]
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else (out_act or nn.Identity)
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)


def ortho_init(model: nn.Module, gain: float = 1.0):
    """
    Orthogonal init for all Linear layers; zero biases.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
