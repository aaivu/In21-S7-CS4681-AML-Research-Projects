import os
import torch
import random
import numpy as np
from config import Config
from train import Trainer

def main():
    # Load configuration
    args = Config()

    # Set random seeds for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Create checkpoints directory if it doesn't exist
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    print(f"Starting ensemble training for {args.n_ensemble} models...")

    for i in range(args.n_ensemble):
        print(f">>>>>>> Training Model {i+1} of {args.n_ensemble} <<<<<<<<")
        
        # Each model needs a unique seed for its weights, but the data sequence remains the same.
        # We achieve this by setting a new seed for torch before initializing each model.
        torch.manual_seed(fix_seed + i)

        # Setting for this specific model in the ensemble
        setting = f'{args.model_id}_sl{args.seq_len}_pl{args.pred_len}_ensemble_{i}'
        
        # Initialize and train the model
        trainer = Trainer(args, setting)
        trainer.train()

        # Clear CUDA cache if using GPU
        if args.use_gpu:
            torch.cuda.empty_cache()

    print(f"Ensemble training complete. {args.n_ensemble} models saved.")


if __name__ == '__main__':
    main()
