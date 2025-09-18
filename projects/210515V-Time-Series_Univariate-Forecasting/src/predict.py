import os
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

from config import Config
from train import Trainer
from data_loader import data_provider
from utils.metrics import metric

def predict_with_uncertainty():
    # Load configuration
    args = Config()
    
    # Load the test data to get ground truth, scaler, and feature names
    test_data, test_loader = data_provider(args, flag='test')
    
    # Load the original data to get the correct dates
    original_df = pd.read_csv(os.path.join(args.root_path, args.data_path))
    original_dates = pd.to_datetime(original_df['date'])
    
    trues = []
    for _, (_, batch_y, _, _) in enumerate(test_loader):
        trues.append(batch_y)
    trues = np.concatenate(trues, axis=0)[:, -args.pred_len:, :]
    
    # Store predictions and attributions from each model in the ensemble
    ensemble_preds = []
    # Use defaultdict to easily aggregate attribution scores
    ensemble_ig_scores = defaultdict(list)
    ensemble_perm_scores = defaultdict(list)

    # Get feature names from the dataset object
    feature_names = test_data.feature_names
    target_feature_idx = feature_names.index(args.target)
    
    print("Loading ensemble models and making predictions...")
    all_inputs = None # To store inputs for attribution analysis

    for i in range(args.n_ensemble):
        setting = f'{args.model_id}_sl{args.seq_len}_pl{args.pred_len}_ensemble_{i}'
        print(f">>> Predicting with model {i+1}: {setting} <<<")
        
        trainer = Trainer(args, setting)
        
        # Get predictions and the inputs that generated them
        preds, inputs = trainer.predict(load_model=True)
        ensemble_preds.append(preds)
        
        if all_inputs is None:
            all_inputs = inputs

        # Get feature attribution for the first batch of test data for demonstration
        print(f"--- Calculating feature attribution for model {i+1} ---")
        first_batch_x, first_batch_y, _, _ = next(iter(test_loader))
        first_batch_x = first_batch_x.float().to(trainer.device)
        first_batch_y = first_batch_y.float().to(trainer.device)

        attribution = trainer.model.feature_attribution(first_batch_x, first_batch_y, feature_names, target_feature_idx)
        
        # Aggregate scores
        for name, score in attribution['integrated_gradients'].items():
            ensemble_ig_scores[name].append(score)
        for name, score in attribution['permutation_importance'].items():
            ensemble_perm_scores[name].append(score)

    print("\nAggregating ensemble predictions...")
    ensemble_preds = np.stack(ensemble_preds)
    
    mean_preds = np.mean(ensemble_preds, axis=0)
    std_preds = np.std(ensemble_preds, axis=0)
    
    n_samples, pred_len, n_features = mean_preds.shape
    
    # --- Inverse scale the results ---
    mean_preds_reshaped = mean_preds.reshape(-1, n_features)
    trues_reshaped = trues.reshape(-1, n_features)
    mean_preds_inv = test_data.inverse_transform(mean_preds_reshaped).reshape(n_samples, pred_len, n_features)
    trues_inv = test_data.inverse_transform(trues_reshaped).reshape(n_samples, pred_len, n_features)

    # --- Calculate overall metrics ---
    mae, mse, rmse, mape, mspe, rse, corr = metric(mean_preds_inv, trues_inv)
    print(f"\n" + "="*80)
    print(" " * 25 + "OVERALL ENSEMBLE METRICS")
    print("="*80)
    print(f"MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.4f}%, CORR: {corr:.6f}")
    print("="*80)
    
    # --- Feature Attribution Analysis ---
    print("\n" + "="*80)
    print(" " * 22 + "FEATURE ATTRIBUTION ANALYSIS (Sample 0)")
    print("="*80)
    print("This analysis shows which input features were most influential for the prediction.")
    
    # Average the scores across the ensemble
    avg_ig_scores = {name: np.mean(scores) for name, scores in ensemble_ig_scores.items()}
    avg_perm_scores = {name: np.mean(scores) for name, scores in ensemble_perm_scores.items()}

    # Create a DataFrame for nice printing
    attr_df = pd.DataFrame({
        'Feature': feature_names,
        'Integrated Gradients': [avg_ig_scores.get(name, 0) for name in feature_names],
        'Permutation Importance': [avg_perm_scores.get(name, 0) for name in feature_names]
    })
    attr_df = attr_df.sort_values(by='Permutation Importance', ascending=False).reset_index(drop=True)
    
    print(attr_df.to_string())
    print("-"*80)
    print("Interpretation:")
    print(" - Integrated Gradients: Shows feature contribution. Positive = pushes prediction up, Negative = pushes down. Magnitude = importance.")
    print(" - Permutation Importance: Shows model reliance on a feature. Higher value means the feature is more important for accuracy.")
    print("="*80)

    # --- Sample Prediction Visualization ---
    sample_idx = 0
    target_true_sample = trues_inv[sample_idx, :, target_feature_idx]
    target_pred_sample = mean_preds_inv[sample_idx, :, target_feature_idx]
    
    # Calculate confidence interval
    z_score = stats.norm.ppf(1 - (1 - args.confidence_level) / 2)
    std_preds_inv = std_preds * test_data.scaler.scale_
    sample_std = std_preds_inv[sample_idx, :, target_feature_idx]
    lower_bound = target_pred_sample - z_score * sample_std
    upper_bound = target_pred_sample + z_score * sample_std
    
    # Calculate sample metrics
    sample_mae = np.mean(np.abs(target_true_sample - target_pred_sample))
    sample_rmse = np.sqrt(np.mean((target_true_sample - target_pred_sample)**2))
    
    print(f"\n" + "="*80)
    print(" " * 20 + f"SAMPLE PREDICTION ANALYSIS (Sample {sample_idx})")
    print("="*80)
    print(f"MAE for sample {sample_idx}: {sample_mae:.6f}")
    print(f"RMSE for sample {sample_idx}: {sample_rmse:.6f}")
    coverage = np.mean((target_true_sample >= lower_bound) & (target_true_sample <= upper_bound))
    print(f"Coverage ({args.confidence_level*100:.0f}% CI): {coverage*100:.1f}%")
    
    # Generate line graph
    plt.figure(figsize=(15, 7))
    plt.plot(target_true_sample, color='blue', label='Actual OT')
    plt.plot(target_pred_sample, color='green', label='Predicted OT')
    plt.fill_between(range(args.pred_len), lower_bound, upper_bound, color='green', alpha=0.2, label=f'{args.confidence_level*100:.0f}% CI')
    
    plt.title(f'Actual vs Predicted OT - Sample {sample_idx}', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('OT Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_filename = f'OT_prediction_sample_{sample_idx}.png'
    plt.savefig(plot_filename)
    print(f"\nPlot saved as: {plot_filename}")
    plt.show()

if __name__ == '__main__':
    predict_with_uncertainty()