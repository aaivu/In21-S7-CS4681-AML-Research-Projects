import os
import itertools

# --- Hyperparameter Search Space ---
# Define the ranges you want to test. Keep them small to run quickly.
params = {
    'lr': [0.001, 0.0005],
    'embed_dim': [64, 128],
    'n_layers': [2, 3, 4],
    'reg': [1e-4, 1e-5]
}

# --- Generate all combinations ---
keys, values = zip(*params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Generated {len(experiments)} experiments to run.")

# --- Run Experiments ---
for i, exp_params in enumerate(experiments):
    print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
    print(f"Parameters: {exp_params}")
    
    # Construct the command to run the training script
    command = (
        f"python src/train_content.py "
        f"--epochs 25 " # Train for a reasonable number of epochs
        f"--lr {exp_params['lr']} "
        f"--embed_dim {exp_params['embed_dim']} "
        f"--n_layers {exp_params['n_layers']} "
        f"--reg {exp_params['reg']}"
    )
    
    # Execute the command
    os.system(command)

print("\n--- Hyperparameter tuning complete! ---")
print("Check the 'results' folder for all experiment logs.")
