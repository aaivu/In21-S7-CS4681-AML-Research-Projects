import os, sys, csv, torch, numpy as np, matplotlib.pyplot as plt

# path setup
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DecisionTransformerConfig
from transformer.art import AutonomousRendezvousTransformer
from dynamics.orbit_dynamics import map_rtn_to_roe, map_roe_to_rtn, dynamics

# -------------------------------------------------------
# Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning evaluation on {device}\n")

data_dir         = os.path.join(root_folder, "dataset") # directory with preprocessed data
ckpt_dir         = os.path.join(root_folder, "saved_files/checkpoints/physicsaware") # directory with physics-aware model checkpoint
baseline_ckpt_dir= os.path.join(root_folder, "saved_files/checkpoints/baseline") # directory with baseline model checkpoint

baseline_ckpt = os.path.join(baseline_ckpt_dir, "baseline_art.pt")
physics_ckpt  = os.path.join(ckpt_dir,           "best_model.pt")

# -------------------------------------------------------
# Load Dataset (normalized inputs) + params (raw OE, dt)
print("Loading dataset...", end="")
states = torch.load(os.path.join(data_dir, "torch_states_rtn_cvx_norm.pth"))  # (N,T,6) normalized
actions = torch.load(os.path.join(data_dir, "torch_actions_cvx_norm.pth"))    # (N,T,3) normalized
rtgs    = torch.load(os.path.join(data_dir, "torch_rtgs_cvx_norm.pth"))       # (N,T)
ctgs    = torch.load(os.path.join(data_dir, "torch_ctgs_cvx_norm.pth"))       # (N,T)
oe_raw  = torch.load(os.path.join(data_dir, "torch_oe_cvx_raw.pth"))          # (N,T,6) physical units

# params (to fetch dt per episode)
param_npz = np.load(os.path.join(data_dir, "dataset-rpod-cvx-param.npz"))
dtime_np  = param_npz["dtime"]  # (N,)

# scaler stats (to denormalize states/actions for physics metrics)
scaler = np.load(os.path.join(data_dir, "scaler_stats.npz"))
mean_vec  = scaler["mean"]   # length 17: [state(6), action(3), rtg, ctg, oe(6)]
scale_vec = scaler["scale"]

# indices as used in your preprocessing script
STATE_SL  = slice(0, 6)
ACT_SL    = slice(6, 9)

state_mean  = torch.tensor(mean_vec[STATE_SL],  dtype=torch.float32, device=device).view(1,1,6)
state_scale = torch.tensor(scale_vec[STATE_SL], dtype=torch.float32, device=device).view(1,1,6)
act_mean    = torch.tensor(mean_vec[ACT_SL],    dtype=torch.float32, device=device).view(1,1,3)
act_scale   = torch.tensor(scale_vec[ACT_SL],   dtype=torch.float32, device=device).view(1,1,3)

def denorm_states(x_norm):   # (B,T,6) -> meters
    return x_norm*state_scale + state_mean

def denorm_actions(u_norm):  # (B,T,3) -> m/s
    return u_norm*act_scale + act_mean

print(" done.")

N, T, Dx = states.shape
Du = actions.shape[2]
n_train = int(0.9 * N)

val = {
    "states":  states[n_train:],         # normalized
    "actions": actions[n_train:],        # normalized
    "rtgs":    rtgs[n_train:],           # normalized
    "ctgs":    ctgs[n_train:],           # normalized
    "oe":      oe_raw[n_train:],         # raw
    "dt":      torch.tensor(dtime_np[n_train:], dtype=torch.float32)  # raw scalar per episode
}
print(f"Dataset: N_val={len(val['states'])}, T={T}, Dx={Dx}, Du={Du}")

# -------------------------------------------------------
# Validation Dataset
class RpodValDataset(Dataset):
    def __init__(self): self.data = val
    def __len__(self):  return len(self.data["states"])
    def __getitem__(self, idx):
        s  = self.data["states"][idx]  # (T,6) norm
        a  = self.data["actions"][idx] # (T,3) norm
        r  = self.data["rtgs"][idx]    # (T,)
        c  = self.data["ctgs"][idx]    # (T,)
        oe = self.data["oe"][idx]      # (T,6) raw
        dt = self.data["dt"][idx]      # scalar
        t   = torch.arange(s.size(0), dtype=torch.long)
        mask= torch.ones_like(t, dtype=torch.long)
        return s, a, r, c, t, mask, oe, dt

val_loader = DataLoader(RpodValDataset(), batch_size=4, shuffle=False)

# -------------------------------------------------------
# Model Config + Load
config = DecisionTransformerConfig(
    state_dim=Dx, act_dim=Du, hidden_size=384, n_layer=6, n_head=6,
    max_ep_len=T, vocab_size=1, resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1
)

def load_model(ckpt_path, name):
    model = AutonomousRendezvousTransformer(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ Loaded {name} from {ckpt_path}")
    return model

baseline = load_model(baseline_ckpt, "Baseline ART")
physics  = load_model(physics_ckpt,  "Physics-Aware ART")

# -------------------------------------------------------
# Metric Helpers
@torch.no_grad()
def eval_loss(model, loader):
    total, total_state, total_act = 0.0, 0.0, 0.0
    for batch in loader:
        s,a,r,c,t,mask,oe,dt = [b.to(device) for b in batch]  # s,a normalized
        B, Tlocal, _ = s.shape
        timesteps     = t.view(B, Tlocal)                      # (B,T)
        attention_mask= mask.view(B, Tlocal)                   # (B,T)
        s_pred, a_pred = model(
            states=s, actions=a,
            returns_to_go=r.unsqueeze(-1), constraints_to_go=c.unsqueeze(-1),
            timesteps=timesteps, attention_mask=attention_mask,
            return_dict=False
        )
        l_a = F.mse_loss(a_pred, a)
        l_s = F.mse_loss(s_pred[:, :-1], s[:, 1:])
        total      += (l_a + l_s).item()
        total_state+= l_s.item()
        total_act  += l_a.item()
    n = len(loader)
    return total/n, total_state/n, total_act/n

def physics_step(x_t, u_t, oe_t, oe_next, dt):
    # All inputs are physical units
    try:
        roe_t   = map_rtn_to_roe(x_t, oe_t)
        roe_next= dynamics(roe_t, u_t, oe_t, float(dt))
        return map_roe_to_rtn(roe_next, oe_next)
    except Exception:
        return x_t

@torch.no_grad()
def physics_residual(model, loader, num_batches=10):
    """Mean || f(x_t, û_t) - x_{t+1} || in meters, using denormalized states/actions and true dt."""
    residuals = []
    for i, batch in enumerate(loader):
        if i >= num_batches: break
        s,a,_,_,t,mask,oe,dt = [b.to(device) for b in batch]  # s,a normalized; oe raw; dt raw
        B, Tlocal, _ = s.shape
        # model forward with proper (B,T) masks
        s_pred, a_pred = model(
            states=s, actions=a,
            returns_to_go=torch.zeros((B,Tlocal,1), device=device),
            constraints_to_go=torch.zeros((B,Tlocal,1), device=device),
            timesteps=t.view(B,Tlocal), attention_mask=mask.view(B,Tlocal),
            return_dict=False
        )
        # denormalize states and predicted actions
        s_phys     = denorm_states(s)        # (B,T,6) meters
        a_pred_phys= denorm_actions(a_pred)  # (B,T,3) m/s (if that's your unit)
        for b in range(B):
            oe_b = oe[b].detach().cpu().numpy().T   # (6,T)
            dt_b = float(dt[b].item())
            s_b  = s_phys[b].detach().cpu().numpy() # (T,6)
            a_b  = a_pred_phys[b].detach().cpu().numpy()
            for tt in range(Tlocal-1):
                x_t     = s_b[tt]
                u_t     = a_b[tt]
                x_next  = physics_step(x_t, u_t, oe_b[:, tt], oe_b[:, tt+1], dt_b)
                residuals.append(np.linalg.norm(x_next - s_b[tt+1]))
    return float(np.mean(residuals)) if residuals else float("nan")

# -------------------------------------------------------
# Rollout + Extended Metrics (in meters)
@torch.no_grad()
@torch.no_grad()
def rollout(model, s0_norm, oe_seq_raw, dt, horizon=100):
    """
    Autoregressive rollout in model space (normalized), then denormalize to meters.

    We maintain growing histories so that at step k we pass sequences of length k+1
    for states, actions, rtg, ctg, timesteps, and attention_mask.
    """
    # init history (normalized space)
    s_hist = s0_norm.view(1, 1, -1).to(device)            # (1,1,6)
    a_hist = torch.zeros((1, 1, 3), device=device)        # (1,1,3) dummy/zero action
    r_hist = torch.zeros((1, 1, 1), device=device)        # (1,1,1)
    c_hist = torch.zeros((1, 1, 1), device=device)        # (1,1,1)

    traj_norm = [s_hist.squeeze(0).squeeze(0)]            # list of (6,)

    for k in range(horizon - 1):
        # build (1, k+1) masks & timesteps
        t = torch.arange(s_hist.size(1), dtype=torch.long, device=device).unsqueeze(0)  # (1, k+1)
        m = torch.ones_like(t, dtype=torch.long)                                        # (1, k+1)

        # forward with full history
        s_pred, a_pred = model(
            states=s_hist, actions=a_hist,
            returns_to_go=r_hist, constraints_to_go=c_hist,
            timesteps=t, attention_mask=m, return_dict=False
        )

        # take the last predictions
        a_last = a_pred[:, -1:]                         # (1,1,3)
        s_last = s_pred[:, -1:]                         # (1,1,6)

        # append to history (grow by one)
        a_hist = torch.cat([a_hist, a_last], dim=1)     # (1, k+2, 3)
        s_hist = torch.cat([s_hist, s_last], dim=1)     # (1, k+2, 6)
        r_hist = torch.cat([r_hist, torch.zeros_like(r_hist[:, :1])], dim=1)
        c_hist = torch.cat([c_hist, torch.zeros_like(c_hist[:, :1])], dim=1)

        traj_norm.append(s_last.squeeze(0).squeeze(0))  # collect normalized state

    traj_norm = torch.stack(traj_norm, dim=0).unsqueeze(0)   # (1, T, 6) normalized
    traj_phys = denorm_states(traj_norm.to(device)).squeeze(0).cpu().numpy()  # (T,6) meters
    return traj_phys


def compute_rollout_metrics(model, val_data, steps=(10,50,100), max_cases=30):
    errors = {f"MSE@{k}": [] for k in steps}
    feasible = 0
    n_cases = min(max_cases, len(val_data["states"]))
    for i in range(n_cases):
        s_true_norm = val_data["states"][i]     # (T,6) normalized
        s_true_phys = denorm_states(s_true_norm.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        oe_seq_raw  = val_data["oe"][i].numpy()
        dt          = float(val_data["dt"][i].item())
        s0_norm     = s_true_norm[0]
        Tlocal      = s_true_norm.shape[0]
        traj_pred_phys = rollout(model, s0_norm, oe_seq_raw, dt, horizon=Tlocal)  # (T,6) meters

        # errors in meters
        for k in steps:
            if k < Tlocal:
                e = np.mean((traj_pred_phys[:k] - s_true_phys[:k])**2)
                errors[f"MSE@{k}"].append(e)

        # crude feasibility: stay within 50 m RTN radius
        if np.all(np.linalg.norm(traj_pred_phys[:, :3], axis=1) < 50.0):
            feasible += 1

    metrics = {k: float(np.mean(v)) if len(v)>0 else float("nan") for k,v in errors.items()}
    metrics["Feasibility_Ratio"] = feasible / float(n_cases)
    return metrics

# -------------------------------------------------------
# Evaluate Models
print("\n Evaluating models ...")
val_loss_base, val_s_base, val_a_base   = eval_loss(baseline, val_loader)
val_loss_phys, val_s_phys, val_a_phys   = eval_loss(physics,  val_loader)
res_base = physics_residual(baseline, val_loader, num_batches=25)
res_phys = physics_residual(physics,  val_loader, num_batches=25)
roll_base = compute_rollout_metrics(baseline, val)
roll_phys = compute_rollout_metrics(physics,  val)

# Aggregate
results = {
    "Baseline_ART": {
        "total": val_loss_base, "state": val_s_base, "action": val_a_base,
        "physics_res": res_base, **roll_base
    },
    "PhysicsAware_ART": {
        "total": val_loss_phys, "state": val_s_phys, "action": val_a_phys,
        "physics_res": res_phys, **roll_phys
    },
}

# -------------------------------------------------------
# Display + CSV
print("\n Validation Summary:")
for name, vals in results.items():
    print(f"\n{name}:")
    for m, n in vals.items():
        print(f"   {m:20s} = {n:.6e}")

csv_path = os.path.join(ckpt_dir, "evaluation_comparison_full.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Model","Total_Loss","State_Loss","Action_Loss","Physics_Residual",
              "MSE@10","MSE@50","MSE@100","Feasibility_Ratio"]
    writer.writerow(header)
    for name, vals in results.items():
        writer.writerow([
            name, vals["total"], vals["state"], vals["action"], vals["physics_res"],
            vals.get("MSE@10", float("nan")), vals.get("MSE@50", float("nan")),
            vals.get("MSE@100", float("nan")), vals.get("Feasibility_Ratio", float("nan"))
        ])
print(f"\n Results exported to {csv_path}")

# -------------------------------------------------------
# Bar Plot Summary (meters-based residuals/MSE)
labels = ["Total Loss", "Physics Residual (m)", "MSE@10 (m^2)", "MSE@50 (m^2)", "MSE@100 (m^2)"]
baseline_vals = [val_loss_base, res_base, roll_base["MSE@10"], roll_base["MSE@50"], roll_base["MSE@100"]]
physics_vals  = [val_loss_phys, res_phys, roll_phys["MSE@10"], roll_phys["MSE@50"], roll_phys["MSE@100"]]

x = np.arange(len(labels)); w = 0.35
plt.figure(figsize=(9,4))
plt.bar(x - w/2, baseline_vals, w, label="Baseline ART", color="crimson")
plt.bar(x + w/2, physics_vals,  w, label="Physics-Aware ART", color="royalblue")
plt.xticks(x, labels, rotation=10)
plt.ylabel("Value (see units)")
plt.title("Model Comparison Summary")
plt.legend(); plt.tight_layout(); plt.show()

# -------------------------------------------------------
# Rollout Plots (in meters)
rng = np.random.default_rng(0)
for idx in rng.choice(len(val["states"]), size=min(2, len(val["states"])), replace=False):
    s_true_norm = val["states"][idx]
    s_true_phys = denorm_states(s_true_norm.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    oe_seq_raw  = val["oe"][idx].numpy()
    dt          = float(val["dt"][idx].item())
    s0_norm     = s_true_norm[0]
    Tlocal      = s_true_norm.shape[0]

    traj_base = rollout(baseline, s0_norm, oe_seq_raw, dt, horizon=Tlocal)
    traj_phys = rollout(physics,  s0_norm, oe_seq_raw, dt, horizon=Tlocal)

    plt.figure(figsize=(6,6))
    plt.plot(s_true_phys[:,1], s_true_phys[:,0], "k--", label="Ground Truth")
    plt.plot(traj_base[:,1],    traj_base[:,0],    "r",   label="Baseline ART")
    plt.plot(traj_phys[:,1],    traj_phys[:,0],    "b",   label="Physics-Aware ART")
    plt.xlabel("Along-track [m]"); plt.ylabel("Radial [m]")
    plt.legend(); plt.title(f"Trajectory #{int(idx)}")
    plt.axis("equal"); plt.tight_layout(); plt.show()

print("\n Evaluation completed successfully.")
