# append to your models.py (or replace contents with these additions)
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

class TwinCritic(nn.Module):
    """
    Two independent critics Q1 and Q2. Each outputs a scalar.
    """
    def __init__(self, obs_dim, act_dim, hidden=(400,300)):
        super().__init__()
        self.q1_net = MLP([obs_dim + act_dim, *hidden, 1])
        self.q2_net = MLP([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1_net(x).squeeze(-1)
        q2 = self.q2_net(x).squeeze(-1)
        return q1, q2

    def q1(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_net(x).squeeze(-1)

class RNDModel(nn.Module):
    """
    RND: fixed target network + predictor network.
    predictor tries to mimic output of target; prediction error = intrinsic reward.
    """
    def __init__(self, obs_dim, hidden=(128,128)):
        super().__init__()
        # target network — randomly initialized and fixed
        self.target = MLP([obs_dim, *hidden, hidden[-1]])
        for p in self.target.parameters():
            p.requires_grad = False

        # predictor network — trainable
        self.predictor = MLP([obs_dim, *hidden, hidden[-1]])

    def forward(self, obs):
        # obs: (batch, obs_dim)
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        return p, t

class RewardEnsemble(nn.Module):
    """
    Ensemble of small reward predictors. Each predicts scalar r_hat(s,a,sp).
    We use ensemble mean as learned reward and ensemble variance as uncertainty bonus.
    """
    def __init__(self, obs_dim, act_dim, ensemble_size=3, hidden=(128,128)):
        super().__init__()
        self.models = nn.ModuleList([
            MLP([obs_dim*2 + act_dim, *hidden, 1]) for _ in range(ensemble_size)
        ])

    def forward(self, s, a, sp):
        # s, a, sp: tensors
        x = torch.cat([s, a, sp], dim=-1)
        outs = [m(x).squeeze(-1) for m in self.models]
        # (ensemble_size, batch)
        stacked = torch.stack(outs, dim=0)
        mean = torch.mean(stacked, dim=0)
        var = torch.var(stacked, dim=0, unbiased=False)
        return mean, var, stacked
