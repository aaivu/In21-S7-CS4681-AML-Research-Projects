import os, sys, argparse, time, numpy as np, torch

# Path setup
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)


import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DecisionTransformerConfig
from accelerate import Accelerator
from contextlib import contextmanager

from transformer.art import AutonomousRendezvousTransformer
from dynamics.orbit_dynamics import dynamics, map_rtn_to_roe, map_roe_to_rtn

# -------------------------------------------------------
# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data_dir", type=str,
                    default="dataset")
parser.add_argument("--save_dir", type=str,
                    default="saved_files/checkpoints/physicsaware_run2")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-5)
args = parser.parse_args()

# -------------------------------------------------------
# Reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# -------------------------------------------------------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on {device}\n")

# -------------------------------------------------------
# LOAD DATA
print("Loading dataset...", end="")

states = torch.load(os.path.join(args.data_dir, "torch_states_rtn_cvx_norm.pth"))
actions = torch.load(os.path.join(args.data_dir, "torch_actions_cvx_norm.pth"))
rtgs = torch.load(os.path.join(args.data_dir, "torch_rtgs_cvx_norm.pth"))
ctgs = torch.load(os.path.join(args.data_dir, "torch_ctgs_cvx_norm.pth"))
oe = torch.load(os.path.join(args.data_dir, "torch_oe_cvx_norm.pth"))

params_npz = np.load(os.path.join(args.data_dir, "dataset-rpod-cvx-param.npz"))
dtime = torch.tensor(params_npz["dtime"], dtype=torch.float32)
time = torch.tensor(params_npz["time"], dtype=torch.float32)
horizons = torch.tensor(params_npz["horizons"], dtype=torch.float32)

N, T, Dx = states.shape
Du = actions.shape[2]
print(f" done. N={N}, T={T}, Dx={Dx}, Du={Du}")

# -------------------------------------------------------
# TRAIN/VAL SPLIT
n_train = int(0.9 * N)
train = dict(states=states[:n_train], actions=actions[:n_train],
             rtg=rtgs[:n_train], ctg=ctgs[:n_train],
             oe=oe[:n_train], dt=dtime[:n_train], time=time[:n_train],
             horizon=horizons[:n_train])
val = dict(states=states[n_train:], actions=actions[n_train:],
           rtg=rtgs[n_train:], ctg=ctgs[n_train:],
           oe=oe[n_train:], dt=dtime[n_train:], time=time[n_train:],
           horizon=horizons[n_train:])

# -------------------------------------------------------
# DATASET CLASS
class RpodDataset(Dataset):
    def __init__(self, split): self.data = train if split=="train" else val
    def __len__(self): return len(self.data["states"])
    def __getitem__(self, idx):
        s = self.data["states"][idx]; a = self.data["actions"][idx]
        r = self.data["rtg"][idx]; c = self.data["ctg"][idx]
        oe = self.data["oe"][idx]; dt = self.data["dt"][idx]
        t = torch.arange(s.size(0))
        mask = torch.ones_like(t)
        return s, a, r, c, t, mask, oe, dt

train_loader = DataLoader(RpodDataset("train"), batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(RpodDataset("val"),   batch_size=args.batch_size)

# -------------------------------------------------------
# SETUP MODEL + OPTIMIZER
os.makedirs(args.save_dir, exist_ok=True)
save_path = os.path.join(args.save_dir, "best_model.pt")
log_path  = os.path.join(args.save_dir, "training_log.txt")

# Physics loss coefficients
lambda_dyn = 1e-4
alpha_roll = 1e-5
lambda_dyn_max = 0.01
alpha_roll_max = 0.01
schedule_step = 5000

# validation cadence
validate_every_steps = 2000   # run a validation every N updates
val_max_batches = 50          # cap number of val batches per validation (None = full set)

# early stopping (optional)
early_stop_patience = 8       # number of validations without improvement
metric_name = "val_loss"      # metric to track


# Model config
config = DecisionTransformerConfig(
    state_dim=Dx,
    act_dim=Du,
    hidden_size=384,
    max_ep_len=T,
    n_layer=6,
    n_head=6,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)

model = AutonomousRendezvousTransformer(config).to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)
accelerator = Accelerator(mixed_precision="no", gradient_accumulation_steps=4)
model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

# EMA setup
ema_decay = 0.999
use_ema = True  


ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()} if use_ema else None

def ema_update(model, ema_state, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k].mul_(decay).add_(v, alpha=1.0 - decay)


# Helper functions for physics loss
def physics_step(x_t, u_t, oe_t, oe_next, dt):
    try:
        if args.state_rep == "rtn":
            roe_t = map_rtn_to_roe(x_t, oe_t)
            roe_next = dynamics(roe_t, u_t, oe_t, dt)
            x_next = map_roe_to_rtn(roe_next, oe_next)
        else:
            x_next = dynamics(x_t, u_t, oe_t, dt)
    except Exception:
        return x_t
    
def physics_residual_loss(states_i, actions_pred, oe_i, dt_i):
    B, T, Dx = states_i.shape
    dyn_loss = 0.0
    for b in range(B):
        oe_b = oe_i[b].cpu().numpy().T
        dt = float(dt_i[b].item())
        for t in range(T-1):
            x_t = states_i[b, t].detach().cpu().numpy()
            u_t = actions_pred[b, t].detach().cpu().numpy()
            x_next_phys = physics_step(x_t, u_t, oe_b[:, t], oe_b[:, t+1], dt)
            x_next_phys = torch.from_numpy(x_next_phys).to(device)
            dyn_loss += F.mse_loss(states_i[b, t+1], x_next_phys)
    return dyn_loss / B



@contextmanager
def swap_weights(model, new_state):
    old = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(new_state, strict=False)
    try: yield
    finally: model.load_state_dict(old, strict=False)


@torch.no_grad()
def run_validation(model, val_loader, max_batches=None, device="cpu"):
    model.eval()
    total = 0.0
    n = 0
    it = iter(val_loader)
    batches_run = 0
    while True:
        try:
            batch = next(it)
        except StopIteration:
            break
        s, a, r, c, t, mask, oe, dt = [x.to(device) for x in batch]
        s_pred, a_pred = model(
            states=s, actions=a,
            returns_to_go=r.unsqueeze(-1),
            constraints_to_go=c.unsqueeze(-1),
            timesteps=t, attention_mask=mask,
            return_dict=False,
        )
        loss = F.mse_loss(a_pred, a) + F.mse_loss(s_pred[:, :-1], s[:, 1:])
        total += loss.item()
        n += 1
        batches_run += 1
        if max_batches is not None and batches_run >= max_batches:
            break
    return total / max(1, n)


# Logging setup
logfile = open(log_path, "w", buffering=1)
def log_print(msg): print(msg); logfile.write(msg + "\n")

log_print(f"\nPhysics-Aware ART Training\nData dir: {args.data_dir}\nSave dir: {args.save_dir}\n")


# -------------------------------------------------------------
# TRAINING LOOP
global_step = 0
best_val = float("inf")
best_step = -1
no_improve = 0

for epoch in range(args.epochs):
    model.train()
    log_print(f"\n===== EPOCH {epoch+1}/{args.epochs} =====")

    for step, batch in enumerate(train_loader):
        global_step += 1
        s, a, r, c, t, mask, oe, dt = [x.to(device) for x in batch]

        # Forward
        s_pred, a_pred = model(states=s, actions=a,
                               returns_to_go=r.unsqueeze(-1),
                               constraints_to_go=c.unsqueeze(-1),
                               timesteps=t, attention_mask=mask,
                               return_dict=False)

        # Losses
        loss_act   = F.mse_loss(a_pred, a)
        loss_state = F.mse_loss(s_pred[:, :-1], s[:, 1:])
        loss_dyn   = physics_residual_loss(s, a_pred, oe, dt)

        rollout_loss = 0.0

        H = 3 # rollout horizon
        if H > 0:
                B = s.shape[0]
                for b in range(B):
                    oe_b = oe[b].cpu().numpy().T
                    dt_b = float(dt[b].item())
                    start = np.random.randint(0, max(1, T - H - 1))
                    s_now = s[b, start].detach().cpu().numpy()
                    for k in range(H):
                        u_pred = a_pred[b, start+k].detach().cpu().numpy()
                        x_next_phys = physics_step(s_now, u_pred, oe_b[:, start+k], oe_b[:, start+k+1], dt_b)
                        s_now = x_next_phys.copy()
                        rollout_loss += F.mse_loss(
                            torch.tensor(x_next_phys, device=device),
                            s[b, start+k+1]
                        )
        loss_roll = (rollout_loss / B)


        loss_total = loss_act + loss_state + lambda_dyn * loss_dyn + alpha_roll * loss_roll

        optimizer.zero_grad()
        accelerator.backward(loss_total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # EMA update
        if use_ema:
            ema_update(model, ema_state, ema_decay)

        # Logging
        if global_step % 500 == 0 or global_step == 1:
            log_print(f"[Step {global_step:06d}] "
                      f"L_total={loss_total.item():.4e} | "
                      f"L_act={loss_act.item():.4e} | "
                      f"L_state={loss_state.item():.4e} | "
                      f"L_dyn={loss_dyn.item():.4e} | "
                      f"L_roll={loss_roll.item():.4e} | "
                      f"lambda_dyn={lambda_dyn:.1e} | alpha_roll={alpha_roll:.1e}")

        # Schedules
        if global_step % schedule_step == 0:
            lambda_dyn = min(lambda_dyn * 10, lambda_dyn_max)
            alpha_roll = min(alpha_roll * 10, alpha_roll_max)
            log_print(f"Scheduler updated -> lambda_dyn={lambda_dyn:.1e}, alpha_roll={alpha_roll:.1e}")

        # Step-wise validation + checkpoint
        if global_step % validate_every_steps == 0:
            if use_ema:
                with swap_weights(model, ema_state):
                    current_val = run_validation(model, val_loader, max_batches=val_max_batches, device=device)
            else:
                current_val = run_validation(model, val_loader, max_batches=val_max_batches, device=device)

            log_print(f"[VAL @ step {global_step}] {metric_name}={current_val:.4e} "
                      f"(best={best_val:.4e} @ step {best_step})")

            if current_val < best_val:
                best_val = current_val
                best_step = global_step
                to_save = ema_state if (use_ema and ema_state is not None) else model.state_dict()
                torch.save(to_save, save_path)
                log_print(f" Saved new best model at step {global_step} -> {save_path}")
                no_improve = 0
            else:
                no_improve += 1
                if early_stop_patience is not None and no_improve >= early_stop_patience:
                    log_print(f" Early stopping: no improvement in {no_improve} validations "
                              f"(best={best_val:.4e} @ step {best_step}).")
                    break
    # optional: break outer loop if early-stopped
    if early_stop_patience is not None and no_improve >= early_stop_patience:
        break


logfile.close()
log_print("\n Training completed successfully.")
