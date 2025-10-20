import os
import csv
from src.dataset import load_imdb_subset
from src.model import HybridAttentionClassifier
from src.train import train_model

os.makedirs("results", exist_ok=True)

train_dataset, test_dataset = load_imdb_subset()

experiments = [
    ("Baseline", False, False),
    ("FlashAttention", True, False),
    ("Linear Attention", False, True),
    ("Hybrid Flash+Linear", True, True)
]

results_list = []

for name, flash, linear in experiments:
    print(f"\n--- {name} ---")
    model = HybridAttentionClassifier(use_flash=flash, use_linear=linear)
    result = train_model(model, train_dataset, test_dataset)
    results_list.append({"experiment": name, **result})

# Save results to CSV
csv_path = os.path.join("results", "experiment_results.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
    writer.writeheader()
    writer.writerows(results_list)

print(f"\n Results saved to {csv_path}")
