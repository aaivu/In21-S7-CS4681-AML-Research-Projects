# util.py
from typing import Dict, Tuple
import random, math
import numpy as np
import torch

# ---------- Reproducibility ----------
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- Small helpers ----------
def one_hot(i: int, n: int):
    v = np.zeros(n, dtype=np.float32); v[i] = 1.0; return v

def build_agent_specific_states(obs_dict: Dict[str, np.ndarray], agent_order: list) -> Dict[str, np.ndarray]:
    all_obs = np.concatenate([obs_dict[a] for a in agent_order], axis=-1)
    n_agents = len(agent_order)
    return {a: np.concatenate([all_obs, one_hot(i, n_agents)], axis=-1) for i, a in enumerate(agent_order)}

# ---------- Value Normalization ----------
class ValueNorm:
    def __init__(self, eps: float = 1e-5):
        self.mean = 0.0; self.var = 1.0; self.count = eps
    def update(self, x: torch.Tensor):
        x = x.detach().cpu().numpy()
        b_mean, b_var, b_count = x.mean(), x.var() + 1e-8, x.shape[0]
        delta = b_mean - self.mean; tot = self.count + b_count
        new_mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count; m_b = b_var * b_count
        M2 = m_a + m_b + delta * delta * (self.count * b_count) / tot
        self.mean, self.var, self.count = float(new_mean), float(M2 / tot), float(tot)
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (math.sqrt(self.var) + 1e-8)
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (math.sqrt(self.var) + 1e-8) + self.mean

# ---------- GAE(λ) & returns ----------
def compute_gae_returns(rew: torch.Tensor, val: torch.Tensor, done: torch.Tensor, gamma: float, lam: float, n_agents: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rew.shape[0] // n_agents
    adv = torch.zeros_like(rew); ret = torch.zeros_like(rew)
    for k in range(n_agents):
        s = slice(k * T, (k + 1) * T)
        r, v, d = rew[s], val[s], done[s]
        v_next = torch.cat([v[1:], torch.zeros(1, device=v.device)], dim=0)
        delta = r + (1.0 - d) * gamma * v_next - v
        gae = 0.0; A = torch.zeros_like(delta)
        for t in reversed(range(T)):
            gae = delta[t] + (1.0 - d[t]) * gamma * lam * gae
            A[t] = gae
        adv[s] = A; ret[s] = A + v
    return adv, ret

def whiten(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + eps)

def iter_chunks(T: int, chunk_len: int):
    """Yield (start, end) indices for truncated BPTT over time dimension."""
    t = 0
    while t < T:
        yield t, min(t + chunk_len, T)
        t += chunk_len
