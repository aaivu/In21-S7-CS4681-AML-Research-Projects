# model.py
import torch
import torch.nn as nn
from modules import mlp, ortho_init


class Actor(nn.Module):
    """Shared policy πθ(a|o): local observation -> action logits."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim], activation=nn.ReLU)
        ortho_init(self.net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralCritic(nn.Module):
    """Centralized Vφ(s_i): concat(all obs) + 1-hot(agent_id) -> value."""
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = mlp([state_dim, hidden, hidden, 1], activation=nn.ReLU)
        ortho_init(self.net)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ---------------- Recurrent (GRU) variants ----------------

class RecurrentActorGRU(nn.Module):
    """Recurrent policy πθ(a|o_t, h_{t-1}). Expects [B, T, obs_dim]."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, act_dim)
        ortho_init(self.fc); ortho_init(self.head)

    def forward(self, obs_seq, h0=None):
        # obs_seq: [B, T, obs_dim]
        x = self.fc(obs_seq)
        x, h = self.rnn(x, h0)        # [B, T, hid], [1, B, hid]
        logits = self.head(x)         # [B, T, act_dim]
        return logits, h


class RecurrentCentralCriticGRU(nn.Module):
    """Recurrent critic Vφ(s_t, h_{t-1}). Expects [B, T, state_dim]."""
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.v = nn.Linear(hidden, 1)
        ortho_init(self.fc); ortho_init(self.v)

    def forward(self, state_seq, h0=None):
        x = self.fc(state_seq)
        x, h = self.rnn(x, h0)        # [B, T, hid]
        v = self.v(x).squeeze(-1)     # [B, T]
        return v, h
