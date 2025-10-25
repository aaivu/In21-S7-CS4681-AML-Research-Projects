import os, sys, json, math, time, argparse
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.model import LSTMLanguageModel
from src.data_utils import build_vocab, numericalize, batchify, get_batch
from src.train_utils import train_epoch, evaluate, count_parameters, ppl_per_million, time_per_million


# ======================================
# Parse Arguments
# ======================================
# parser = argparse.ArgumentParser()
# parser.add_argument('--use-moe', action='store_true', help='Use Mixture of Experts head')
# parser.add_argument('--mode', type=str, default='hybrid', choices=['baseline', 'switch', 'lora', 'hybrid', 'none'])
# parser.add_argument('--data-dir', type=str, default='./data/wikitext')
# parser.add_argument('--output-dir', type=str, default='./experiments/outputs')
# parser.add_argument('--embed-size', type=int, default=256)
# parser.add_argument('--hidden-size', type=int, default=256)
# parser.add_argument('--num-layers', type=int, default=2)
# parser.add_argument('--num-experts', type=int, default=8)
# parser.add_argument('--expert-hidden', type=int, default=256)
# parser.add_argument('--batch-size', type=int, default=16)
# parser.add_argument('--seq-len', type=int, default=32)
# parser.add_argument('--epochs', type=int, default=8)
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--aux-coef', type=float, default=1e-3)
# parser.add_argument('--clip', type=float, default=0.25)
# parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
# args = parser.parse_args()

default_args = {
    "use_moe": False,
    "mode": "hybrid",
    "data_dir": "./data/wikitext",
    "output_dir": "./experiments/outputs",
    "embed_size": 256,
    "hidden_size": 256,
    "num_layers": 2,
    "num_experts": 8,
    "expert_hidden": 256,
    "batch_size": 16,
    "seq_len": 32,
    "epochs": 8,
    "lr": 1e-4,
    "aux_coef": 1e-3,
    "clip": 0.25,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to JSON config file')
parser.add_argument('--use-moe', type=bool, default=None, help='Use Mixture of Experts head')
parser.add_argument('--mode', type=str, choices=['baseline', 'switch', 'lora', 'hybrid', 'none'])
parser.add_argument('--data-dir', type=str)
parser.add_argument('--output-dir', type=str)
parser.add_argument('--embed-size', type=int)
parser.add_argument('--hidden-size', type=int)
parser.add_argument('--num-layers', type=int)
parser.add_argument('--num-experts', type=int)
parser.add_argument('--expert-hidden', type=int)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--seq-len', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--aux-coef', type=float)
parser.add_argument('--clip', type=float)
parser.add_argument('--device', type=str)

args = parser.parse_args()


# Start with hardcoded defaults
final_args = default_args.copy()


if args.config:
    with open(args.config, "r") as f:
        config = json.load(f)
    final_args.update(config)


for key in default_args.keys():
    value = getattr(args, key, None)
    if value is not None:
        final_args[key] = value

for key, value in final_args.items():
    setattr(args, key, value)

print("MoE: ", args.use_moe)



# ======================================
# Setup Paths
# ======================================
device = torch.device(args.device)
exp_dir = os.path.join(args.output_dir, args.mode if args.use_moe else "no_moe")
os.makedirs(exp_dir, exist_ok=True)
plot_dir = os.path.join(exp_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)


# ======================================
# Data Loading
# ======================================
train_df = pq.read_table(os.path.join(args.data_dir, "train-00000-of-00001.parquet")).to_pandas()
val_df = pq.read_table(os.path.join(args.data_dir, "validation-00000-of-00001.parquet")).to_pandas()
test_df = pq.read_table(os.path.join(args.data_dir, "test-00000-of-00001.parquet")).to_pandas()

train_lines = train_df["text"].tolist()
val_lines = val_df["text"].tolist()
test_lines = test_df["text"].tolist()

special_tokens = ['<pad>', '<unk>', '<eos>']
vocab, stoi, itos, unk_idx = build_vocab(train_lines, min_freq=1, special_tokens=special_tokens)
ntokens = len(stoi)

train_data = batchify(numericalize(train_lines, stoi, unk_idx), args.batch_size, device)
val_data = batchify(numericalize(val_lines, stoi, unk_idx), args.batch_size, device)
test_data = batchify(numericalize(test_lines, stoi, unk_idx), args.batch_size, device)


# ======================================
# Model Setup
# ======================================
model = LSTMLanguageModel(
    vocab_size=ntokens,
    embed_size=args.embed_size,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    use_moe=args.use_moe,
    moe_output_size=ntokens if args.use_moe else None,
    moe_num_experts=args.num_experts,
    moe_hidden_size=args.expert_hidden,
    moe_mode=args.mode,  
)

# Apply mode if using MoE
if args.use_moe:
    model.moe.mode = args.mode
    if args.mode == 'hybrid':
        model.moe.lora_rank = 8

model = model.to(device)
n_params = count_parameters(model)
print(f"[INFO] Parameters: {n_params/1e6:.2f}M | Mode: {args.mode}")


# ======================================
# Training Configuration
# ======================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

history = []
best_val_ppl = None


# ======================================
# Training Loop
# ======================================
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train_loss = train_epoch(model, train_data, optimizer, criterion, args.seq_len, args.clip, device, get_batch, aux_coef=args.aux_coef)
    val_loss = evaluate(model, val_data, criterion, args.seq_len, device, get_batch)

    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss)
    epoch_time = time.time() - t0

    scheduler.step(val_loss)
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
        "epoch_time": epoch_time,
        "ppl_per_million": ppl_per_million(val_ppl, n_params),
        "time_per_million": time_per_million(epoch_time, n_params)
    })

    print(f"[Epoch {epoch}] Train PPL={train_ppl:.2f} | Val PPL={val_ppl:.2f} | Time={epoch_time:.2f}s")


# ======================================
# Evaluate on Test
# ======================================
test_loss = evaluate(model, test_data, criterion, args.seq_len, device, get_batch)
test_ppl = math.exp(test_loss)
history.append({"epoch": "final_test", "test_loss": test_loss, "test_ppl": test_ppl})

print(f"[TEST] Final Perplexity: {test_ppl:.2f}")


# ======================================
# Save Logs and Summary
# ======================================
log_path = os.path.join(exp_dir, f"training_log_{args.mode if args.use_moe else 'no_moe'}.json")
with open(log_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"[SAVED] Training log -> {log_path}")

summary_path = os.path.join(exp_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Mode: {args.mode if args.use_moe else 'no_moe'}\n")
    f.write(f"Use MoE: {args.use_moe}\n")
    f.write(f"Parameters: {n_params/1e6:.2f}M\n")
    f.write(f"Final Test Perplexity: {test_ppl:.2f}\n")

# ======================================
# Plot results
# ======================================
df = pd.DataFrame([h for h in history if isinstance(h['epoch'], int)])

plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["val_ppl"], marker="o", label="Validation PPL")
plt.xlabel("Epoch"); plt.ylabel("Perplexity"); plt.title(f"Validation Perplexity ({args.mode})")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "validation_perplexity_vs_epoch.png"))

plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["epoch_time"], marker="o", label="Epoch Time")
plt.xlabel("Epoch"); plt.ylabel("Time (s)"); plt.title(f"Epoch Time ({args.mode})")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "epoch_time_vs_epoch.png"))

plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["time_per_million"], marker="o", label="Time per Million Params")
plt.xlabel("Epoch"); plt.ylabel("Time / Million Params (s)")
plt.title(f"Training Efficiency ({args.mode})")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "time_per_million_vs_epoch.png"))

print(f"[SAVED] Plots -> {plot_dir}")
