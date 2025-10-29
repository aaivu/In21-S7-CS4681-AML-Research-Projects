# utils.py
import numpy as np
import torch
import random
import os

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    if env is not None:
        try:
            env.reset(seed=seed)
        except Exception:
            pass

def scale_action(a, low, high):
    return low + (a + 1.) * 0.5 * (high - low)  # map from [-1,1] to [low,high]
