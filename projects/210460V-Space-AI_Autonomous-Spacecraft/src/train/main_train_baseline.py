import os, sys, argparse, torch, numpy as np

# Path Setup
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)


from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import DecisionTransformerConfig, get_scheduler
from accelerate import Accelerator
from transformer.art import AutonomousRendezvousTransformer

parser = argparse.ArgumentParser(description="Baseline ART Training")
parser.add_argument("--data_dir", type=str, required=True, default="dataset")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()
args.data_dir = os.path.join(root_folder, args.data_dir)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on {device}\n")

# -------------------------------------------------------
# Load normalized dataset
print("Loading dataset...", end="")
states = torch.load(os.path.join(args.data_dir, "torch_states_rtn_cvx_norm.pth"))
actions = torch.load(os.path.join(args.data_dir, "torch_actions_cvx_norm.pth"))
rtgs = torch.load(os.path.join(args.data_dir, "torch_rtgs_cvx_norm.pth"))
ctgs = torch.load(os.path.join(args.data_dir, "torch_ctgs_cvx_norm.pth"))
print(" done.")

N, T, Dx = states.shape
Du = actions.shape[2]
print(f"Dataset → N={N}, T={T}, Dx={Dx}, Du={Du}")

# -------------------------------------------------------
# Split train/val
n_train = int(0.9 * N)
train = {"states": states[:n_train], "actions": actions[:n_train],
         "rtgs": rtgs[:n_train], "ctgs": ctgs[:n_train]}
val = {"states": states[n_train:], "actions": actions[n_train:],
       "rtgs": rtgs[n_train:], "ctgs": ctgs[n_train:]}

# -------------------------------------------------------
# Dataset class
class RpodDataset(Dataset):
    def __init__(self, split):
        self.data = train if split == "train" else val
    def __len__(self):
        return len(self.data["states"])
    def __getitem__(self, idx):
        s = self.data["states"][idx]
        a = self.data["actions"][idx]
        r = self.data["rtgs"][idx]
        c = self.data["ctgs"][idx]
        t = torch.arange(s.size(0))
        mask = torch.ones_like(t)
        return s, a, r, c, t, mask

train_loader = DataLoader(RpodDataset("train"), batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(RpodDataset("val"), batch_size=args.batch_size)

# -------------------------------------------------------
# Model setup
config = DecisionTransformerConfig(
    state_dim=Dx,
    act_dim=Du,
    hidden_size=384,
    n_layer=6,
    n_head=6,
    max_ep_len=T,
    vocab_size=1,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1
)
model = AutonomousRendezvousTransformer(config).to(device)

optimizer = AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01)
scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=1000, num_training_steps=50000)
accel = Accelerator()
model, optimizer, train_loader, val_loader = accel.prepare(model, optimizer, train_loader, val_loader)

print(f"Model initialized -> {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# -------------------------------------------------------
# Evaluation function
@torch.no_grad()
def evaluate():
    model.eval()
    total, total_state, total_act = 0, 0, 0
    for batch in val_loader:
        s,a,r,c,t,mask = [b.to(device) for b in batch]
        s_pred,a_pred = model(states=s, actions=a, returns_to_go=r.unsqueeze(-1),
                              constraints_to_go=c.unsqueeze(-1), timesteps=t,
                              attention_mask=mask, return_dict=False)
        l_a = F.mse_loss(a_pred, a)
        l_s = F.mse_loss(s_pred[:,:-1], s[:,1:])
        total += (l_a+l_s).item()
        total_state += l_s.item()
        total_act += l_a.item()
    model.train()
    return total/len(val_loader), total_state/len(val_loader), total_act/len(val_loader)

# -------------------------------------------------------
# Training loop
print("\nStarting Baseline ART Training\n")
save_dir = os.path.join(root_folder, "transformer/saved_files/baselinerun/checkpoints")
os.makedirs(save_dir, exist_ok=True)
best_val = float("inf")
step = 0
eval_steps = 500
epochs = args.epochs

for epoch in range(epochs):
    for batch in train_loader:
        s,a,r,c,t,mask = [b.to(device) for b in batch]
        with accel.accumulate(model):
            s_pred,a_pred = model(states=s, actions=a, returns_to_go=r.unsqueeze(-1),
                                  constraints_to_go=c.unsqueeze(-1), timesteps=t,
                                  attention_mask=mask, return_dict=False)
            loss_u = F.mse_loss(a_pred, a)
            loss_x = F.mse_loss(s_pred[:,:-1], s[:,1:])
            total_loss = loss_u + loss_x

            accel.backward(total_loss)
            accel.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            # Logging
            if step % 100 == 0:
                accel.print(f"[step {step}] total={total_loss.item():.4f} "
                            f"(act={loss_u.item():.4f}, state={loss_x.item():.4f})")

            # Evaluate
            if step % eval_steps == 0:
                val_loss, val_s, val_a = evaluate()
                accel.print(f"Eval@{step}: total={val_loss:.4f}, state={val_s:.4f}, action={val_a:.4f}")

                # Save best model
                if val_loss < best_val:
                    best_val = val_loss
                    best_path = os.path.join(save_dir, "baseline_art.pt")
                    torch.save(model.state_dict(), best_path)
                    accel.print(f"🏆 New best model saved at step {step} (val={best_val:.4f})")

            # Save periodic checkpoint
            if step % 5000 == 0:
                ckpt_path = os.path.join(save_dir, f"baseline_art_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                accel.print(f"💾 Checkpoint saved: {ckpt_path}")

accel.save_state(os.path.join(save_dir, "checkpoint_bseline_final"))
print("\n Baseline ART training complete.")
print(f"Best validation loss: {best_val:.4f}")
