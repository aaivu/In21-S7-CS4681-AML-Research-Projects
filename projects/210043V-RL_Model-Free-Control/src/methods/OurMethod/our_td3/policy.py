from dataclasses import dataclass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .td3_core import Actor, Critic
from .replay_per import PrioritizedReplayBuffer
from .replay_vmfer import AgreementReplayBuffer
from . import vi_td3

@dataclass
class OurTD3Config:
    obs_dim: int
    act_dim: int
    act_limit: float
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    actor_delay: int = 2
    batch_size: int = 256
    replay_size: int = 1000000
    replay: str = "uniform"            # "uniform" | "per" | "vmfer"
    use_vi: bool = False
    vi_coef: float = 0.05
    grad_clip: float = 10.0

class OurTD3:
    def __init__(self, cfg: OurTD3Config, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

        self.actor = Actor(cfg.obs_dim, cfg.act_dim, cfg.act_limit).to(self.device)
        self.actor_targ = Actor(cfg.obs_dim, cfg.act_dim, cfg.act_limit).to(self.device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.q1 = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q2 = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q1_targ = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q2_targ = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q1_targ.load_state_dict(self.q1.state_dict())
        self.q2_targ.load_state_dict(self.q2.state_dict())

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        if cfg.replay == "per":
            self.replay = PrioritizedReplayBuffer(cfg.replay_size)
        else:
            self.replay = AgreementReplayBuffer(cfg.replay_size)

        self.total_it = 0

    def act(self, obs, noise_scale=0.1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor(obs_t).cpu().numpy()
        if noise_scale > 0:
            a += noise_scale * np.random.randn(*a.shape)
        return np.clip(a, -self.cfg.act_limit, self.cfg.act_limit)

    def push(self, s, a, r, s2, d):
        self.replay.push((s, a, r, s2, d))

    def _can_sample(self):
        if isinstance(self.replay, AgreementReplayBuffer):
            return self.replay.size >= self.cfg.batch_size
        else:
            return self.replay.tree.size >= self.cfg.batch_size

    def update(self):
        if not self._can_sample():
            return {}

        self.total_it += 1
        # Sample
        if self.cfg.replay == "per":
            batch, idxs, weights = self.replay.sample(self.cfg.batch_size)
            w = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)
        else:
            batch, idxs = self.replay.sample(self.cfg.batch_size)
            w = None

        s, a, r, s2, d = zip(*batch)
        s = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array(a), dtype=torch.float32, device=self.device)
        r = torch.as_tensor(np.array(r), dtype=torch.float32, device=self.device).unsqueeze(-1)
        s2 = torch.as_tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(np.array(d), dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Targets with smoothing
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            a2 = (self.actor_targ(s2) + noise).clamp(-self.cfg.act_limit, self.cfg.act_limit)
            q1_t = self.q1_targ(s2, a2)
            q2_t = self.q2_targ(s2, a2)
            q_targ = torch.min(q1_t, q2_t)
            backup = r + self.cfg.gamma * (1 - d) * q_targ

        # Current Q
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        td_err1 = q1 - backup
        td_err2 = q2 - backup

        # Base critic loss
        base = (td_err1.pow(2) + td_err2.pow(2)) / 2.0

        # Replay weighting
        if self.cfg.replay == "per":
            critic_loss = (w * base).mean()
            with torch.no_grad():
                prios = (td_err1.abs() + td_err2.abs()).mean(dim=1).cpu().numpy()
            self.replay.update_priorities(idxs, prios)
        elif self.cfg.replay == "vmfer":
            with torch.no_grad():
                eps = 1e-8
                num = (td_err1 * td_err2).sum(dim=1, keepdim=True)
                den = td_err1.norm(dim=1, keepdim=True) * td_err2.norm(dim=1, keepdim=True) + eps
                cos = (num / den).clamp(-1.0, 1.0)
                kappa = 5.0
                w_agree = torch.exp(kappa * cos)
                w_agree = (w_agree / (w_agree.mean() + 1e-8)).clamp(0.1, 10.0)
            critic_loss = (w_agree * base).mean()
        else:
            critic_loss = base.mean()

        # Optional VI term
        if self.cfg.use_vi:
            vi = vi_td3.vi_loss(q1, q1_t, q2_t, coefficient=self.cfg.vi_coef) + \
                 vi_td3.vi_loss(q2, q1_t, q2_t, coefficient=self.cfg.vi_coef)
            critic_loss = critic_loss + vi

        self.q1_opt.zero_grad(set_to_none=True)
        self.q2_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.cfg.grad_clip)
        self.q1_opt.step()
        self.q2_opt.step()

        info = {"critic_loss": float(critic_loss.detach().cpu().item())}

        # Delayed actor update
        if self.total_it % self.cfg.actor_delay == 0:
            a_pi = self.actor(s)
            actor_loss = -self.q1(s, a_pi).mean()
            self.pi_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
            self.pi_opt.step()

            # Polyak
            with torch.no_grad():
                for p, p_t in zip(self.actor.parameters(), self.actor_targ.parameters()):
                    p_t.data.mul_(1 - self.cfg.tau); p_t.data.add_(self.cfg.tau * p.data)
                for p, p_t in zip(self.q1.parameters(), self.q1_targ.parameters()):
                    p_t.data.mul_(1 - self.cfg.tau); p_t.data.add_(self.cfg.tau * p.data)
                for p, p_t in zip(self.q2.parameters(), self.q2_targ.parameters()):
                    p_t.data.mul_(1 - self.cfg.tau); p_t.data.add_(self.cfg.tau * p.data)
            info.update({"actor_loss": float(actor_loss.detach().cpu().item())})

        return info