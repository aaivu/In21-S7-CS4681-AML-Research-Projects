
import os
from datasets import load_dataset

print("Loading XSum dataset...")
print("This may take a while on first run (downloading ~500MB data)...\n")

# Load the dataset from HuggingFace Hub
# trust_remote_code=True is required for XSum
dataset = load_dataset("xsum", trust_remote_code=True)

# Create output directory
output_dir = "xsum_parquet"
os.makedirs(output_dir, exist_ok=True)

# Save each split to Parquet
for split_name in ["train", "validation", "test"]:
    output_path = os.path.join(output_dir, f"{split_name}.parquet")
    print(f"Saving {split_name} split ({len(dataset[split_name])} examples) to {output_path}...")
    dataset[split_name].to_parquet(output_path)
