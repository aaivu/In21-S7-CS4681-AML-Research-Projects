"""
Evaluation and Ensembling Script (RoBERTa)
Loads all N fold models, creates ensemble predictions, and calculates bias metrics.
Memory-optimized with sequential model loading and mixed precision.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from RoBERTaForToxicity import RoBERTaForToxicity
from JigsawDataset import JigsawDataset
from jigsaw_metrics import calculate_all_metrics


class EnsembleModel:
    """
    Memory-efficient ensemble of RoBERTa models using Power-Weighted averaging.
    Models are loaded sequentially to save memory.
    """

    def __init__(self, model_paths, model_name='roberta-base', device=None):
        """
        Args:
            model_paths (list[str]): paths to model checkpoints
            model_name (str): HF model id (e.g., 'roberta-base' or 'roberta-large')
            device: torch.device
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model_paths = model_paths
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        print(f"\nEnsemble will use {len(model_paths)} models:")
        for i, path in enumerate(model_paths):
            print(f"  {i+1}. {path}")

    def predict(self, dataloader, power=3.5, use_mixed_precision=True):
        """
        Generate ensemble predictions using power-weighted mean.
        Memory-efficient: loads one model at a time.

        Args:
            dataloader: DataLoader for the dataset
            power (float): Power for weighted averaging
            use_mixed_precision (bool): Use mixed precision for inference

        Returns:
            np.ndarray: shape (num_samples,)
        """
        all_predictions = []
        use_amp = use_mixed_precision and self.device.type == 'cuda'

        # Process each model sequentially to save memory
        for model_idx, model_path in enumerate(self.model_paths):
            print(f"\nProcessing Model {model_idx+1}/{len(self.model_paths)}")
            
            # Load model
            model = RoBERTaForToxicity(model_name=self.model_name)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            # Get predictions for this model
            model_preds = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"  Fold {model_idx+1} inference"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Use mixed precision if available
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(input_ids, attention_mask)
                            probs = torch.sigmoid(outputs['toxicity']).cpu().numpy()
                    else:
                        outputs = model(input_ids, attention_mask)
                        probs = torch.sigmoid(outputs['toxicity']).cpu().numpy()
                    
                    model_preds.extend(probs)

            all_predictions.append(np.asarray(model_preds))
            
            # Clean up to free memory
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Stack predictions: (n_models, n_samples)
        all_predictions = np.stack(all_predictions, axis=0)

        # Power-weighted mean across models
        print(f"\nComputing ensemble with power={power}...")
        powered = np.power(all_predictions, power)
        mean_powered = powered.mean(axis=0)
        ensemble = np.power(mean_powered, 1.0 / power)
        
        return ensemble


def discover_checkpoints(pattern, n_folds, auto_discover=False, fold_ids=None, allow_missing=False):
    """
    Discover checkpoint files based on pattern and options.
    
    Args:
        pattern (str): Pattern with {fold} placeholder
        n_folds (int): Expected number of folds
        auto_discover (bool): Auto-discover all matching files
        fold_ids (str): Comma-separated fold ids to use
        allow_missing (bool): Allow some folds to be missing
        
    Returns:
        list[str]: List of checkpoint paths
    """
    model_paths = []
    
    if fold_ids:
        # Use specific fold IDs
        specified_folds = [int(x.strip()) for x in fold_ids.split(',')]
        print(f"Using specified folds: {specified_folds}")
        for fold in specified_folds:
            path = pattern.format(fold=fold)
            if os.path.exists(path):
                model_paths.append(path)
            elif not allow_missing:
                raise FileNotFoundError(f"Checkpoint not found: {path}")
        return model_paths
    
    if auto_discover:
        # Auto-discover all matching checkpoints
        base_pattern = pattern.replace('{fold}', '*')
        discovered = glob.glob(base_pattern)
        if discovered:
            model_paths = sorted(discovered)
            print(f"Auto-discovered {len(model_paths)} checkpoints")
            return model_paths
    
    # Default: try to load n_folds checkpoints sequentially
    for i in range(n_folds):
        path = pattern.format(fold=i)
        if os.path.exists(path):
            model_paths.append(path)
        elif not allow_missing:
            # Try fallback to DeBERTa naming
            fallback = f"models/deberta_fold{i}_best.pt"
            if os.path.exists(fallback):
                print(f"⚠ Using fallback: {fallback}")
                model_paths.append(fallback)
    
    return model_paths


def evaluate_on_validation():
    """
    Load validation/test parquet, run ensemble inference, compute bias metrics, and save outputs.
    """
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa toxicity classifier ensemble")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="HF model id used at train time (roberta-base or roberta-large)")
    parser.add_argument("--n_folds", type=int, default=5, 
                        help="Number of folds to load")
    parser.add_argument("--ckpt_pattern", type=str, default="models/roberta_fold{fold}_best.pt",
                        help="Pattern for fold checkpoints (use {fold} placeholder)")
    parser.add_argument("--val_file", type=str, default="jigsaw_val_preprocessed.parquet",
                        help="Validation parquet file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--power", type=float, default=3.5, 
                        help="Power for power-weighted averaging")
    parser.add_argument("--auto_discover", action="store_true",
                        help="Auto-discover all matching checkpoints")
    parser.add_argument("--fold_ids", type=str, default=None,
                        help="Comma-separated fold ids to load, e.g. '0,1,3,4'")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Proceed even if some folds are missing")
    parser.add_argument("--no_mixed_precision", action="store_true",
                        help="Disable mixed precision inference")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-sample predictions to CSV")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("CAFE-Jigsaw Model Evaluation (RoBERTa)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Expected folds: {args.n_folds}")
    print(f"Checkpoint pattern: {args.ckpt_pattern}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Power: {args.power}")
    print(f"Mixed precision: {'Disabled' if args.no_mixed_precision else 'Enabled'}")

    # ----- Load validation data -----
    print("\nLoading validation data...")
    val_file = args.val_file
    if not os.path.exists(val_file):
        print(f"⚠  {val_file} not found. Trying alternative: jigsaw_test_preprocessed.parquet")
        alt = "jigsaw_test_preprocessed.parquet"
        if not os.path.exists(alt):
            raise FileNotFoundError(
                "No validation/test parquet found. Please run preprocessing.py first."
            )
        val_file = alt

    val_df = pd.read_parquet(val_file)
    print(f"✓ Loaded {len(val_df)} validation samples")
    
    if 'target' in val_df.columns:
        print(f"  Target distribution: mean={val_df['target'].mean():.4f}, "
              f"std={val_df['target'].std():.4f}")

    # ----- Discover checkpoints -----
    print("\nDiscovering model checkpoints...")
    model_paths = discover_checkpoints(
        args.ckpt_pattern, 
        args.n_folds, 
        args.auto_discover, 
        args.fold_ids, 
        args.allow_missing
    )

    if not model_paths:
        raise FileNotFoundError(
            f"No checkpoints found matching pattern '{args.ckpt_pattern}'. "
            f"Please train models first or check the pattern."
        )
    
    print(f"✓ Found {len(model_paths)} checkpoint(s)")

    # ----- Create dataset & loader -----
    print("\nPreparing data loader...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    val_dataset = JigsawDataset(val_df, tokenizer, max_length=args.max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=JigsawDataset.collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"✓ Created DataLoader with {len(val_loader)} batches")

    # ----- Create ensemble and predict -----
    print("\n" + "=" * 80)
    print("Running Ensemble Inference")
    print("=" * 80)
    ensemble = EnsembleModel(model_paths, model_name=args.model_name, device=device)
    
    predictions = ensemble.predict(
        val_loader, 
        power=args.power,
        use_mixed_precision=not args.no_mixed_precision
    )
    val_df['prediction'] = predictions
    
    print(f"\n✓ Generated predictions: mean={predictions.mean():.4f}, "
          f"std={predictions.std():.4f}, min={predictions.min():.4f}, max={predictions.max():.4f}")

    # ----- Calculate metrics -----
    print("\n" + "=" * 80)
    print("Calculating Bias Metrics")
    print("=" * 80)
    
    try:
        results_df = calculate_all_metrics(val_df, pred_col='prediction', target_col='target')
        
        # Save results
        results_df.to_csv('evaluation_results.csv', index=False)
        print("\n✓ Results saved to evaluation_results.csv")

        # Display results
        print("\n" + "=" * 80)
        print("Validation Set Performance by Identity Group")
        print("=" * 80)

        display_df = results_df.copy()
        if 'subgroup_size' in display_df.columns:
            display_df['subgroup_size'] = display_df['subgroup_size'].astype(int)
        
        # Format numeric columns
        for col in ['overall_auc', 'subgroup_auc', 'bpsn_auc', 'bnsp_auc']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
        
        print(display_df.to_string(index=False))

        # Print summary statistics
        if 'identity' in results_df.columns:
            print("\n" + "=" * 80)
            print("Summary Statistics")
            print("=" * 80)
            
            numeric = results_df[results_df['identity'] != 'overall']
            for k in ['subgroup_auc', 'bpsn_auc', 'bnsp_auc']:
                if k in numeric.columns and len(numeric[k].dropna()) > 0:
                    metric_name = k.replace('_', ' ').upper()
                    print(f"\n{metric_name}:")
                    print(f"  Mean: {numeric[k].mean():.4f}")
                    print(f"  Std:  {numeric[k].std():.4f}")
                    min_idx = numeric[k].idxmin()
                    max_idx = numeric[k].idxmax()
                    print(f"  Min:  {numeric[k].min():.4f} ({numeric.loc[min_idx, 'identity']})")
                    print(f"  Max:  {numeric[k].max():.4f} ({numeric.loc[max_idx, 'identity']})")

    except Exception as e:
        print(f"\n⚠ Error calculating metrics: {e}")
        print("Continuing without bias metrics...")
        results_df = None

    # ----- Save predictions -----
    if args.save_predictions:
        cols = [c for c in ['comment_text', 'comment_text_cleaned', 'target', 'prediction'] 
                if c in val_df.columns]
        if cols:
            output_file = 'jigsaw_val_predictions.csv'
            val_df[cols].to_csv(output_file, index=False)
            print(f"\n✓ Predictions saved to {output_file}")

    return results_df


if __name__ == "__main__":
    try:
        results = evaluate_on_validation()
        print("\n" + "=" * 80)
        print("✓ Evaluation Complete!")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)