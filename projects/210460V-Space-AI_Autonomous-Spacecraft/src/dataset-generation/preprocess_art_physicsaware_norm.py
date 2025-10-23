import os
import sys
import argparse

# Path setup
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from dynamics.orbit_dynamics import *
from optimization.rpod_scenario import *
from optimization.ocp import *


# Argument parser
parser = argparse.ArgumentParser(description='physics-aware ART preprocessing (with normalization)')
parser.add_argument('--data_dir', type=str, required=True, default='dataset',
                    help='directory where dataset-rpod-cvx files are located')
args = parser.parse_args()
args.data_dir = os.path.join(root_folder, args.data_dir)

print('Loading data...', end='')

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
states_roe_cvx = data_cvx['states_roe_cvx']  # (N, T, 6)
states_rtn_cvx = data_cvx['states_rtn_cvx']  # (N, T, 6)
actions_cvx    = data_cvx['actions_cvx']     # (N, T, 3)
oe             = data_param['oe']            # (N, T, 6)
time_steps     = data_param['time']          # (N, T)
horizons       = data_param['horizons']      # (N,)

# Load RTG/CTG (or compute)
if 'rtg' in data_param and 'ctg' in data_param:
    rtg = data_param['rtg']  # (N, T)
    ctg = data_param['ctg']  # (N, T)
    print("Using precomputed RTG/CTG.")
else:
    print("Computing RTG/CTG from scratch (older dataset format).")
    rtg = compute_reward_to_go(actions_cvx, n_data, n_time)
    ctg = compute_constraint_to_go(states_rtn_cvx, n_data, n_time)

# -----------------------------------------------------------
# Prepare full feature tensor for normalization

# Feature vector per timestep: [state(6), action(3), rtg(1), ctg(1), oe(6)]
tokens = np.concatenate([
    states_rtn_cvx,
    actions_cvx,
    rtg[..., None],
    ctg[..., None],
    oe
], axis=-1)  # (N, T, 17)

flat_tokens = tokens.reshape(-1, tokens.shape[-1])

# Normalize features using StandardScaler
scaler = StandardScaler()
flat_tokens_norm = scaler.fit_transform(flat_tokens)
tokens_norm = flat_tokens_norm.reshape(n_data, n_time, -1)

# -----------------------------------------------------------
# Save normalization statistics
scaler_stats = {
    'mean': scaler.mean_,
    'scale': scaler.scale_,
    'n_features': tokens.shape[-1]
}
np.savez(os.path.join(args.data_dir, 'scaler_stats.npz'), **scaler_stats)
print("Saved normalization statistics: scaler_stats.npz")

# -----------------------------------------------------------
# Split back normalized data

# Unpack normalized components
states_rtn_norm = tokens_norm[..., 0:6]
actions_norm    = tokens_norm[..., 6:9]
rtg_norm        = tokens_norm[..., 9]
ctg_norm        = tokens_norm[..., 10]
oe_norm         = tokens_norm[..., 11:17]

# Convert to torch
torch_states_rtn_cvx = torch.from_numpy(states_rtn_norm).float()
torch_actions_cvx    = torch.from_numpy(actions_norm).float()
torch_rtgs_cvx       = torch.from_numpy(rtg_norm).float()
torch_ctgs_cvx       = torch.from_numpy(ctg_norm).float()
torch_oe_cvx         = torch.from_numpy(oe_norm).float()

# -----------------------------------------------------------
# Save PyTorch tensors
torch.save(torch_states_rtn_cvx, os.path.join(args.data_dir, 'torch_states_rtn_cvx_norm.pth'))
torch.save(torch_actions_cvx,    os.path.join(args.data_dir, 'torch_actions_cvx_norm.pth'))
torch.save(torch_rtgs_cvx,       os.path.join(args.data_dir, 'torch_rtgs_cvx_norm.pth'))
torch.save(torch_ctgs_cvx,       os.path.join(args.data_dir, 'torch_ctgs_cvx_norm.pth'))
torch.save(torch_oe_cvx,         os.path.join(args.data_dir, 'torch_oe_cvx_norm.pth'))

# -----------------------------------------------------------
# Save also non-normalized versions for reference (optional)
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
print("\n Physics-aware ART preprocessing completed successfully.")
print(f"  Normalized features: {tokens.shape[-1]} per timestep")
print("  Saved to directory:", args.data_dir)
print("\n  Files created:")
print("   - torch_states_rtn_cvx_norm.pth")
print("   - torch_actions_cvx_norm.pth")
print("   - torch_rtgs_cvx_norm.pth")
print("   - torch_ctgs_cvx_norm.pth")
print("   - torch_oe_cvx_norm.pth")
print("   - scaler_stats.npz (mean/std for normalization)")
