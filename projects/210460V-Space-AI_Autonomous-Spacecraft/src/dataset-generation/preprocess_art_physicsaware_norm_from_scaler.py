"""
preprocess_art_physicsaware_norm_from_scaler.py
-----------------------------------------------
Physics-aware ART dataset preprocessing that
**uses existing training normalization statistics**
(from scaler_stats.npz) instead of refitting StandardScaler.

Author : Tithira Perera
Date   : 2025-10-13
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------
# Path setup
# -----------------------------------------------------------
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *

# -----------------------------------------------------------
# Argument parser
parser = argparse.ArgumentParser(description='physics-aware ART preprocessing (using existing scaler)')
parser.add_argument('--data_dir', type=str, required=True, default='dataset-seed-2030',
                    help='directory containing dataset-rpod-cvx2000 files')
parser.add_argument('--scaler_path', type=str, required=True, default='dataset-seed-2030/scaler_stats.npz',
                    help='path to training scaler_stats.npz (from training dataset)')
args = parser.parse_args()
args.data_dir = os.path.join(root_folder, args.data_dir)

print("Loading data...", end="")

# -----------------------------------------------------------
# Load dataset (CVX-only)
data_cvx = np.load(os.path.join(args.data_dir, 'dataset-rpod-cvx.npz'))
data_param = np.load(os.path.join(args.data_dir, 'dataset-rpod-cvx-param.npz'))

time_discr = data_param['dtime']
oe = data_param['oe']
n_data = oe.shape[0]
n_time = oe.shape[1]

print(f" done. Found {n_data} samples, {n_time} timesteps each.")

# -----------------------------------------------------------
# Load states, actions, and auxiliary data
states_roe_cvx = data_cvx['states_roe_cvx']
states_rtn_cvx = data_cvx['states_rtn_cvx']
actions_cvx    = data_cvx['actions_cvx']
oe             = data_param['oe']
time_steps     = data_param['time']
horizons       = data_param['horizons']

if 'rtg' in data_param and 'ctg' in data_param:
    rtg = data_param['rtg']
    ctg = data_param['ctg']
    print("Using precomputed RTG/CTG.")
else:
    print("Computing RTG/CTG from scratch (older dataset format).")
    rtg = compute_reward_to_go(actions_cvx, n_data, n_time)
    ctg = compute_constraint_to_go(states_rtn_cvx, n_data, n_time)

# -----------------------------------------------------------
# Prepare full feature tensor (state, action, rtg, ctg, oe)
tokens = np.concatenate([
    states_rtn_cvx,
    actions_cvx,
    rtg[..., None],
    ctg[..., None],
    oe
], axis=-1)  # shape (N, T, 17)

flat_tokens = tokens.reshape(-1, tokens.shape[-1])

# -----------------------------------------------------------
# Load existing training scaler stats
scaler_stats = np.load(args.scaler_path)
mean = scaler_stats['mean']
scale = scaler_stats['scale']

if len(mean) != flat_tokens.shape[-1]:
    raise ValueError(f"Feature dimension mismatch: scaler mean has {len(mean)}, but dataset has {flat_tokens.shape[-1]}")

print("Using existing training scaler for normalization:", args.scaler_path)

# -----------------------------------------------------------
# Normalize using training statistics
flat_tokens_norm = (flat_tokens - mean) / scale
tokens_norm = flat_tokens_norm.reshape(n_data, n_time, -1)

# -----------------------------------------------------------
# Split back into components
states_rtn_norm = tokens_norm[..., 0:6]
actions_norm    = tokens_norm[..., 6:9]
rtg_norm        = tokens_norm[..., 9]
ctg_norm        = tokens_norm[..., 10]
oe_norm         = tokens_norm[..., 11:17]

# -----------------------------------------------------------
# Convert to torch tensors
torch_states_rtn_cvx = torch.from_numpy(states_rtn_norm).float()
torch_actions_cvx    = torch.from_numpy(actions_norm).float()
torch_rtgs_cvx       = torch.from_numpy(rtg_norm).float()
torch_ctgs_cvx       = torch.from_numpy(ctg_norm).float()
torch_oe_cvx         = torch.from_numpy(oe_norm).float()

# -----------------------------------------------------------
# Save normalized data
torch.save(torch_states_rtn_cvx, os.path.join(args.data_dir, 'torch_states_rtn_cvx_norm.pth'))
torch.save(torch_actions_cvx,    os.path.join(args.data_dir, 'torch_actions_cvx_norm.pth'))
torch.save(torch_rtgs_cvx,       os.path.join(args.data_dir, 'torch_rtgs_cvx_norm.pth'))
torch.save(torch_ctgs_cvx,       os.path.join(args.data_dir, 'torch_ctgs_cvx_norm.pth'))
torch.save(torch_oe_cvx,         os.path.join(args.data_dir, 'torch_oe_cvx_norm.pth'))

# -----------------------------------------------------------
# Save also non-normalized (raw) for reference
torch.save(torch.from_numpy(states_rtn_cvx).float(), os.path.join(args.data_dir, 'torch_states_rtn_cvx_raw.pth'))
torch.save(torch.from_numpy(actions_cvx).float(),    os.path.join(args.data_dir, 'torch_actions_cvx_raw.pth'))
torch.save(torch.from_numpy(rtg).float(),            os.path.join(args.data_dir, 'torch_rtgs_cvx_raw.pth'))
torch.save(torch.from_numpy(ctg).float(),            os.path.join(args.data_dir, 'torch_ctgs_cvx_raw.pth'))
torch.save(torch.from_numpy(oe).float(),             os.path.join(args.data_dir, 'torch_oe_cvx_raw.pth'))

# -----------------------------------------------------------
# Metadata
torch.save(torch.from_numpy(time_steps).float(), os.path.join(args.data_dir, 'torch_time_cvx.pth'))
torch.save(torch.from_numpy(time_discr).float(), os.path.join(args.data_dir, 'torch_dtime_cvx.pth'))
torch.save(torch.from_numpy(horizons).float(),   os.path.join(args.data_dir, 'torch_horizon_cvx.pth'))

# -----------------------------------------------------------
# Summary
print("\n Preprocessing completed using existing training scaler.")
print(f"  Normalized using: {args.scaler_path}")
print(f"  Saved to: {args.data_dir}")
print("\n  Files created:")
print("   - torch_states_rtn_cvx_norm.pth")
print("   - torch_actions_cvx_norm.pth")
print("   - torch_rtgs_cvx_norm.pth")
print("   - torch_ctgs_cvx_norm.pth")
print("   - torch_oe_cvx_norm.pth")
