# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(activation())
            else:
                layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(400,300)):
        super().__init__()
        sizes = [obs_dim, *hidden, act_dim]
        self.net = MLP(sizes, activation=nn.ReLU, output_activation=nn.Tanh)
    def forward(self, obs):
        # output in [-1,1]
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(400,300)):
        super().__init__()
        # Q(s,a) - concatenate
        self.net = MLP([obs_dim + act_dim, *hidden, 1])
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)

class RewardModel(nn.Module):
    """
    Learned reward function r_hat(s,a,s')
    """
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        sizes = [obs_dim*2 + act_dim, *hidden, 1]
        self.net = MLP(sizes, activation=nn.ReLU, output_activation=nn.Identity)
    def forward(self, s, a, sp):
        x = torch.cat([s, a, sp], dim=-1)
        return self.net(x).squeeze(-1)
