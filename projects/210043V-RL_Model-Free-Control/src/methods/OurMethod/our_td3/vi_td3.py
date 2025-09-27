# VI-TD3: critic-side value-improvement regularizer (lightweight)
# Gently nudges Q toward a greedier target while keeping TD3 stabilizers intact.

import torch
import torch.nn.functional as F

@torch.no_grad()
def _greedy_target(q1_targ, q2_targ):
    # Greedier than TD3's min backup: take max of twin target critics.
    return torch.max(q1_targ, q2_targ)

def vi_loss(q_pred, q1_targ, q2_targ, coefficient=0.05):
    '''
    q_pred: current critic estimate Q_theta(s,a)  [B,1]
    q*_targ: twin target critics evaluated on next state + target action [B,1]
    coefficient: small scalar weight (e.g., 0.01-0.1)
    '''
    with torch.no_grad():
        q_greedy = _greedy_target(q1_targ, q2_targ)
    return coefficient * F.mse_loss(q_pred, q_greedy)