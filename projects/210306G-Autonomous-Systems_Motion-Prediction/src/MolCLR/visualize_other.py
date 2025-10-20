import os
import yaml
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Root folder containing the 32 experiment folders
root_dir = "finetune/temp1"

# Collect all experiment folders
exp_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, f))]

# Assign colors to optimizers
optimizer_colors = {"adam": "tab:blue", "adamw": "tab:orange"}

plt.figure(figsize=(12, 7))

for folder in exp_folders:
    config_path = os.path.join(folder, "checkpoints", "config_finetune.yaml")
    if not os.path.exists(config_path):
        continue
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    optimizer_name = config.get("optimizer", "NA").lower()
    color = optimizer_colors.get(optimizer_name, "black")

    # Find tfevents
    tfevent_files = [os.path.join(folder, f) for f in os.listdir(folder) if "tfevents" in f]
    if not tfevent_files:
        continue
    event_file = tfevent_files[0]

    ea = EventAccumulator(event_file)
    ea.Reload()

    tag = "validation_loss"
    if tag not in ea.Tags()['scalars']:
        continue

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # Label is just optimizer
    label = optimizer_name.upper()
    plt.plot(steps, values, label=label, color=color)

plt.xlabel("epoch", fontsize=14)
plt.ylabel("Validation Loss", fontsize=14)
plt.title("Validation Loss Across Experiments: Adam vs AdamW", fontsize=16)
plt.grid(True)

# Legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()