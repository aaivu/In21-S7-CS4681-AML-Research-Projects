import os
import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import Config
from train import Trainer
from data_loader import data_provider
from utils.metrics import metric

def predict_with_uncertainty():
    # Load configuration
    args = Config()
    
    # Load the test data to get ground truth and scaler
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
    ensemble_attributions = []
    
    print("Loading ensemble models and making predictions...")
    for i in range(args.n_ensemble):
        setting = f'{args.model_id}_sl{args.seq_len}_pl{args.pred_len}_ensemble_{i}'
        print(f">>> Predicting with model {i+1}: {setting} <<<")
        
        trainer = Trainer(args, setting)
        
        # Get predictions
        preds = trainer.predict(load_model=True)
        ensemble_preds.append(preds)
        
        # Get temporal attribution after prediction
        if hasattr(trainer.model, 'get_attribution_analysis'):
            attribution = trainer.model.get_attribution_analysis()
            if attribution:  # Only append if we got valid attribution results
                ensemble_attributions.append(attribution)
            else:
                print(f"Warning: Model {i+1} returned empty attribution analysis")
        else:
            print(f"Warning: Model {i+1} does not support temporal attribution analysis")
    
    print("\nAggregating ensemble predictions and temporal attributions...")
    # Stack predictions along a new axis to have (n_ensemble, n_samples, pred_len, n_features)
    ensemble_preds = np.stack(ensemble_preds)
    
    # Calculate mean and standard deviation across the ensemble dimension
    mean_preds = np.mean(ensemble_preds, axis=0)
    std_preds = np.std(ensemble_preds, axis=0)
    
    # Get dimensions from mean_preds
    n_samples, pred_len, n_features = mean_preds.shape
    
    # Process temporal attributions
    print("\nProcessing temporal attribution analysis...")
    attribution_results = []
    
    # Skip attribution analysis if no models support it
    if not ensemble_attributions:
        print("No models in the ensemble support temporal attribution analysis")
        ensemble_has_attribution = False
    else:
        ensemble_has_attribution = True
        
    # For each test sample
    for sample_idx in range(n_samples):
        sample_attribution = {
            'daily': {},
            'weekly': {},
            'monthly': {},
            'regime_change': False,
            'regime_change_point': None
        }
        
        # Average attributions across ensemble
        for ensemble_attr in ensemble_attributions:
            if not ensemble_attr:  # Skip if attribution is empty
                continue
                
            try:
                # Daily scores
                if 'daily' in ensemble_attr:
                    for day, score in ensemble_attr['daily'].items():
                        if day not in sample_attribution['daily']:
                            sample_attribution['daily'][day] = []
                        sample_attribution['daily'][day].append(score)
                
                # Weekly scores
                if 'weekly' in ensemble_attr:
                    for week, score in ensemble_attr['weekly'].items():
                        if week not in sample_attribution['weekly']:
                            sample_attribution['weekly'][week] = []
                        sample_attribution['weekly'][week].append(score)
                
                # Monthly scores
                if 'monthly' in ensemble_attr:
                    for month, score in ensemble_attr['monthly'].items():
                        if month not in sample_attribution['monthly']:
                            sample_attribution['monthly'][month] = []
                        sample_attribution['monthly'][month].append(score)
                
                # Regime change
                if 'regime_change' in ensemble_attr:
                    sample_attribution['regime_change'] = sample_attribution['regime_change'] or ensemble_attr['regime_change']['detected']
                    if ensemble_attr['regime_change']['detected']:
                        sample_attribution['regime_change_point'] = ensemble_attr['regime_change']['point']
            except Exception as e:
                print(f"Warning: Error processing attribution for sample {sample_idx}: {str(e)}")
        
        # Average the scores
        sample_attribution['daily'] = {k: np.mean(v) for k, v in sample_attribution['daily'].items()}
        sample_attribution['weekly'] = {k: np.mean(v) for k, v in sample_attribution['weekly'].items()}
        sample_attribution['monthly'] = {k: np.mean(v) for k, v in sample_attribution['monthly'].items()}
        
        attribution_results.append(sample_attribution)

    # --- FIX STARTS HERE ---
    # Reshape data for the scaler, which expects a 2D array
    n_samples, pred_len, n_features = mean_preds.shape
    mean_preds_reshaped = mean_preds.reshape(-1, n_features)
    trues_reshaped = trues.reshape(-1, n_features)

    # Inverse scale the results
    mean_preds_inv_reshaped = test_data.inverse_transform(mean_preds_reshaped)
    trues_inv_reshaped = test_data.inverse_transform(trues_reshaped)

    # Reshape back to the original 3D shape
    mean_preds_inv = mean_preds_inv_reshaped.reshape(n_samples, pred_len, n_features)
    trues_inv = trues_inv_reshaped.reshape(n_samples, pred_len, n_features)
    
    # Calculate metrics on the mean prediction (ALL test samples and timesteps)
    mae, mse, rmse, mape, mspe, rse, corr = metric(mean_preds_inv, trues_inv)
    
    print(f"\n" + "="*80)
    print(" " * 25 + "OVERALL ENSEMBLE METRICS")
    print("="*80)
    print(f"Test Samples: {n_samples}")
    print(f"Prediction Length: {pred_len}")
    print(f"Total Predictions: {n_samples * pred_len}")
    print(f"Number of Features: {n_features}")
    print("-"*80)
    print(f"MAE (Mean Absolute Error):     {mae:.6f}")
    print(f"MSE (Mean Squared Error):      {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
    print(f"MSPE (Mean Squared Percentage Error):  {mspe:.6f}")
    print(f"RSE (Root Squared Error):      {rse:.6f}")
    print(f"CORR (Correlation):            {corr:.6f}")
    print("="*80)
    
    # Calculate feature-specific metrics for the target feature (OT)
    feature_idx = 7  # OT is the last column (index 7)
    target_true_all = trues_inv[:, :, feature_idx]  # All samples, all timesteps
    target_pred_all = mean_preds_inv[:, :, feature_idx]  # All samples, all timesteps
    
    # Feature-specific metrics
    feature_mae = np.mean(np.abs(target_pred_all - target_true_all))
    feature_mse = np.mean((target_pred_all - target_true_all) ** 2)
    feature_rmse = np.sqrt(feature_mse)
    
    # Calculate MAPE avoiding division by zero
    feature_mape = np.mean(np.abs((target_pred_all - target_true_all) / (target_true_all + 1e-8))) * 100
    
    # Calculate R-squared
    ss_res = np.sum((target_true_all - target_pred_all) ** 2)
    ss_tot = np.sum((target_true_all - np.mean(target_true_all)) ** 2)
    feature_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\n" + "="*80)
    print(f" " * 20 + f"TARGET FEATURE ({args.target}) METRICS")
    print("="*80)
    print(f"Total {args.target} predictions: {target_pred_all.size}")
    print("-"*80)
    print(f"MAE:  {feature_mae:.6f}")
    print(f"MSE:  {feature_mse:.6f}")
    print(f"RMSE: {feature_rmse:.6f}")
    print(f"MAPE: {feature_mape:.4f}%")
    print(f"R²:   {feature_r2:.6f}")
    print("="*80)
    
    # Calculate confidence interval
    z_score = stats.norm.ppf(1 - (1 - args.confidence_level) / 2)
    
    # The standard deviation must also be scaled back using the scaler's `scale_` attribute
    std_preds_inv = std_preds * test_data.scaler.scale_

    # Display a sample prediction with uncertainty for the first sample and first feature
    sample_idx = 0
    feature_idx = 7  # OT is the last column (index 7)
    
    sample_mean = mean_preds_inv[sample_idx, :, feature_idx]
    sample_std = std_preds_inv[sample_idx, :, feature_idx]
    
    # Calculate the correct date indices for the prediction targets
    # The test data starts at a certain index in the original data
    # We need to find where the test data starts and then add the sequence length and sample index
    num_train = int(len(original_df) * 0.7)
    num_test = int(len(original_df) * 0.2)
    test_start_idx = len(original_df) - num_test
    
    # For the first sample (sample_idx=0), the prediction targets start at:
    # test_start_idx + seq_len + sample_idx
    prediction_start_idx = test_start_idx + args.seq_len + sample_idx
    prediction_dates = original_dates[prediction_start_idx:prediction_start_idx + args.pred_len]
    
    # Get the exact original true values directly from the source data
    original_true_values = []
    for t in range(args.pred_len):
        actual_idx = prediction_start_idx + t
        if actual_idx < len(original_df):
            original_true_values.append(original_df.iloc[actual_idx]['OT'])
        else:
            original_true_values.append(np.nan)
    original_true_values = np.array(original_true_values)
    
    # Calculate confidence bounds in the original scale
    lower_bound = sample_mean - z_score * sample_std
    upper_bound = sample_mean + z_score * sample_std
    
    print(f"\n" + "="*80)
    print(" " * 20 + f"SAMPLE PREDICTION ANALYSIS")
    print("="*80)
    print(f"Showing detailed analysis for SAMPLE {sample_idx + 1} (out of {n_samples} test samples)")
    print(f"Target Feature: {args.target} (feature index: {feature_idx})")
    print(f"Prediction period: {args.pred_len} timesteps")
    print("-"*80)
    print(" Timestep |    Date     |  True  |  Pred  | Lower Bound | Upper Bound")
    print("-"*80)
    for t in range(min(10, len(prediction_dates))): # Print first 10 timesteps or available dates
        if t < len(prediction_dates):
            date_str = prediction_dates.iloc[t].strftime('%Y-%m-%d')
            true_value = original_true_values[t]
            print(f" {t+1:^8} | {date_str} | {true_value:6.4f} | {sample_mean[t]:6.4f} |   {lower_bound[t]:6.4f}   |   {upper_bound[t]:6.4f}")
    
    if len(prediction_dates) > 10:
        print(f"    ... (showing first 10 out of {len(prediction_dates)} prediction timesteps)")
    
    # Verification: Calculate metrics using exact original true values for this sample
    print(f"\n--- Sample {sample_idx + 1} Metrics (all {args.pred_len} timesteps) ---")
    sample_mae = np.mean(np.abs(original_true_values - sample_mean))
    sample_mse = np.mean((original_true_values - sample_mean)**2)
    sample_rmse = np.sqrt(sample_mse)
    
    # Calculate sample MAPE
    sample_mape = np.mean(np.abs((original_true_values - sample_mean) / (original_true_values + 1e-8))) * 100
    
    print(f"MAE for sample {sample_idx + 1}: {sample_mae:.6f}")
    print(f"MSE for sample {sample_idx + 1}: {sample_mse:.6f}")
    print(f"RMSE for sample {sample_idx + 1}: {sample_rmse:.6f}")
    print(f"MAPE for sample {sample_idx + 1}: {sample_mape:.4f}%")
    
    # Calculate coverage: how many true values fall within the confidence intervals
    within_bounds = np.sum((original_true_values >= lower_bound) & (original_true_values <= upper_bound))
    coverage = within_bounds / len(original_true_values)
    print(f"Coverage (true values within {args.confidence_level*100:.0f}% CI): {coverage*100:.1f}% ({within_bounds}/{len(original_true_values)})")
    print("="*80)
    
    # Generate line graph of actual vs predicted values for OT
    print("\nGenerating line graph of actual vs predicted values for OT...")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot actual values in blue
    plt.plot(range(len(original_true_values)), original_true_values, 
             color='blue', linewidth=2, label='Actual OT', marker='o', markersize=4)
    
    # Plot predicted values in green
    plt.plot(range(len(sample_mean)), sample_mean, 
             color='green', linewidth=2, label='Predicted OT', marker='s', markersize=4)
    
    # Add confidence interval as a shaded area
    plt.fill_between(range(len(sample_mean)), lower_bound, upper_bound, 
                     color='green', alpha=0.2, label=f'{args.confidence_level*100:.0f}% Confidence Interval')
    
    # Customize the plot
    plt.title(f'Actual vs Predicted OT Values - Sample {sample_idx + 1}\n'
              f'Prediction Length: {args.pred_len} timesteps', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('OT Value', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add text box with metrics
    metrics_text = f'MAE: {sample_mae:.4f}\nRMSE: {sample_rmse:.4f}\nMAPE: {sample_mape:.2f}%\nCoverage: {coverage*100:.1f}%'
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'OT_prediction_sample_{sample_idx + 1}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    # Generate a more detailed view with more samples if available
    if n_samples > 1:
        print(f"\nGenerating detailed comparison for first 500 timesteps across multiple samples...")
        
        # Flatten the data to show continuous predictions across samples
        max_timesteps = min(500, n_samples * pred_len)
        
        # Create flattened arrays
        flattened_true = trues_inv[:, :, feature_idx].flatten()[:max_timesteps]
        flattened_pred = mean_preds_inv[:, :, feature_idx].flatten()[:max_timesteps]
        
        # Create another plot for the detailed view
        plt.figure(figsize=(15, 8))
        
        # Plot actual values in blue
        plt.plot(range(len(flattened_true)), flattened_true, 
                 color='blue', linewidth=1.5, label='Actual OT', alpha=0.8)
        
        # Plot predicted values in green
        plt.plot(range(len(flattened_pred)), flattened_pred, 
                 color='green', linewidth=1.5, label='Predicted OT', alpha=0.8)
        
        # Customize the plot
        plt.title(f'Actual vs Predicted OT Values - Detailed View\n'
                  f'Showing {max_timesteps} timesteps across {min(max_timesteps//pred_len, n_samples)} samples', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('OT Value', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add overall metrics text box
        overall_metrics_text = f'Overall MAE: {feature_mae:.4f}\nOverall RMSE: {feature_rmse:.4f}\nOverall MAPE: {feature_mape:.2f}%\nR²: {feature_r2:.4f}'
        plt.text(0.02, 0.98, overall_metrics_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the detailed plot
        detailed_plot_filename = f'OT_prediction_detailed_view_{max_timesteps}.png'
        plt.savefig(detailed_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved as: {detailed_plot_filename}")
        
        # Show the plot
        plt.show()
        
        # Generate complete test set visualization
        print(f"\nGenerating complete test set visualization...")
        
        # Create a plot for the entire test set
        plt.figure(figsize=(20, 10))
        
        # Use all available data
        complete_true = trues_inv[:, :, feature_idx].flatten()
        complete_pred = mean_preds_inv[:, :, feature_idx].flatten()
        
        # Plot actual values in blue
        plt.plot(range(len(complete_true)), complete_true, 
                 color='blue', linewidth=1, label='Actual OT', alpha=0.7)
        
        # Plot predicted values in green
        plt.plot(range(len(complete_pred)), complete_pred, 
                 color='green', linewidth=1, label='Predicted OT', alpha=0.7)
        
        # Customize the plot
        plt.title(f'Actual vs Predicted OT Values - Complete Test Set\n'
                  f'Total timesteps: {len(complete_true)} across {n_samples} samples', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('OT Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add comprehensive metrics text box
        complete_metrics_text = f'MAE: {feature_mae:.4f}\nMSE: {feature_mse:.4f}\nRMSE: {feature_rmse:.4f}\nMAPE: {feature_mape:.2f}%\nR²: {feature_r2:.4f}\nCorrelation: {corr:.4f}'
        plt.text(0.02, 0.98, complete_metrics_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the complete plot
        complete_plot_filename = 'OT_prediction_complete_test_set.png'
        plt.savefig(complete_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Complete test set plot saved as: {complete_plot_filename}")
        
        # Show the plot
        plt.show()
    
    print("\nSaving prediction results and temporal attribution analysis...")
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save predictions with uncertainty
    predictions_data = []
    for t in range(args.pred_len):
        for sample_idx in range(n_samples):
            predictions_data.append({
                'sample': sample_idx + 1,
                'timestep': t + 1,
                'true_value': trues_inv[sample_idx, t, feature_idx],
                'predicted_value': mean_preds_inv[sample_idx, t, feature_idx],
                'confidence_lower': mean_preds_inv[sample_idx, t, feature_idx] - z_score * std_preds_inv[sample_idx, t, feature_idx],
                'confidence_upper': mean_preds_inv[sample_idx, t, feature_idx] + z_score * std_preds_inv[sample_idx, t, feature_idx]
            })
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
    
    # Save temporal attribution analysis if available
    if ensemble_has_attribution:
        attribution_data = []
        for sample_idx, attr in enumerate(attribution_results):
            # Daily importance scores
            for day, score in attr['daily'].items():
                attribution_data.append({
                    'sample': sample_idx + 1,
                    'time_scale': 'daily',
                    'period': day,
                    'importance_score': score
                })
            
            # Weekly importance scores
            for week, score in attr['weekly'].items():
                attribution_data.append({
                    'sample': sample_idx + 1,
                    'time_scale': 'weekly',
                    'period': week,
                    'importance_score': score
                })
            
            # Monthly importance scores
            for month, score in attr['monthly'].items():
                attribution_data.append({
                    'sample': sample_idx + 1,
                    'time_scale': 'monthly',
                    'period': month,
                    'importance_score': score
                })
            
            # Regime change
            if attr['regime_change']:
                attribution_data.append({
                    'sample': sample_idx + 1,
                    'time_scale': 'regime_change',
                    'period': 'detected',
                    'importance_score': attr['regime_change_point']
                })
        
        attribution_df = pd.DataFrame(attribution_data)
        attribution_df.to_csv(os.path.join(results_dir, 'temporal_attribution.csv'), index=False)
        
        attribution_df.to_csv(os.path.join(results_dir, 'temporal_attribution.csv'), index=False)
        print(f"2. temporal_attribution.csv - Contains temporal attribution analysis")
    
    print(f"\nResults saved in the '{results_dir}' directory:")
    print(f"1. predictions.csv - Contains predicted values with uncertainty bounds")
    print(f"2. temporal_attribution.csv - Contains temporal attribution analysis")
    
    print("\nAll visualizations and results export completed successfully!")
    print("="*80)

if __name__ == '__main__':
    predict_with_uncertainty()

