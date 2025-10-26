import os
import yaml
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.cm as cm

# Root folder containing the 32 experiment folders
root_dir = "finetune/temp"

# Hyperparameters
hp_keys = ["batch_size", "init_lr", "init_base_lr", "weight_decay", "base_weight_decay"]

# Collect all experiment folders
exp_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
               if os.path.isdir(os.path.join(root_dir, f))]

# Get unique batch sizes
batch_sizes = sorted(list({yaml.safe_load(open(os.path.join(f, "checkpoints", "config_finetune.yaml"))).get("batch_size") 
                           for f in exp_folders}))

# Use only two colors for batch sizes
color_map = {bs: cm.tab10(i % 2) for i, bs in enumerate(batch_sizes)}  # two colors: i % 2
linestyles = ["-", "--", "-.", ":"]  # Different line styles for init_lr

plt.figure(figsize=(14, 8))

for folder in exp_folders:
    config_path = os.path.join(folder, "checkpoints", "config_finetune.yaml")
    if not os.path.exists(config_path):
        continue
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    batch_size = config.get("batch_size", "NA")
    init_lr = config.get("init_lr", "NA")
    init_base_lr = config.get("init_base_lr", "NA")
    weight_decay = config.get("weight_decay", "NA")
    base_weight_decay = config.get("base_weight_decay", "NA")

    color = color_map[batch_size]
    linestyle = linestyles[init_lr % len(linestyles)] if isinstance(init_lr, int) else linestyles[0]

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

    # Short label for legend
    label = f"bs:{batch_size}, lr:{init_lr}, blr:{init_base_lr}, wd:{weight_decay}, bwd:{base_weight_decay}"
    plt.plot(steps, values, label=label, color=color, linestyle=linestyle, marker=None)

plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Validation Loss", fontsize=14)
plt.title("Validation Loss Across 32 Experiments", fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True)

# Adjust bottom to make space for text
plt.subplots_adjust(bottom=0.2)

# --- Add explanation of abbreviations below the plot ---
abbrev_text = (
    "Legend abbreviations:\n"
    "bs   = Batch size\n"
    "lr   = Initial learning rate\n"
    "blr  = Initial base learning rate\n"
    "wd   = Weight decay\n"
    "bwd  = Base weight decay"
)

plt.gcf().text(0.3, 0.05, abbrev_text, ha='left', fontsize=10)

plt.show()