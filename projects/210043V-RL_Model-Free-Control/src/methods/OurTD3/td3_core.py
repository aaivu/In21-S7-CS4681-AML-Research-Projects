from dataclasses import dataclass
import torch
import torch.nn as nn

def fanin_init(m):
    if isinstance(m, nn.Linear):
        bound = 1.0 / (m.weight.size(0) ** 0.5)
        nn.init.uniform_(m.weight, -bound, +bound)
        nn.init.uniform_(m.bias, -bound, +bound)

class MLP(nn.Module):
    def _init_(self, in_dim, out_dim, hidden=(256, 256), activation=nn.ReLU):
        super()._init_()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(fanin_init)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def _init_(self, obs_dim, act_dim, act_limit, hidden=(256, 256)):
        super()._init_()
        self.mu = MLP(obs_dim, act_dim, hidden)
        self.act_limit = act_limit

    def forward(self, obs):
        return torch.tanh(self.mu(obs)) * self.act_limit

class Critic(nn.Module):
    def _init_(self, obs_dim, act_dim, hidden=(256, 256)):
        super()._init_()
        self.q = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        import torch
        return self.q(torch.cat([obs, act], dim=-1))