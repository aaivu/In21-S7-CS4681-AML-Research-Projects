#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load the results
df = pd.read_csv('a9a_sweep_results.csv')

# Analyze the results
print('=== A9A Sweep Results Analysis ===')
print(f'Total experiments: {len(df)}')
print(f'Best AUC: {df["test_auc"].max():.6f}')
print(f'Worst AUC: {df["test_auc"].min():.6f}')
print(f'Mean AUC: {df["test_auc"].mean():.6f}')
print(f'Std AUC: {df["test_auc"].std():.6f}')

print('\n=== Best Performing Configurations ===')
best_configs = df.nlargest(5, 'test_auc')[['lr', 'warmup_ratio', 'loss_type', 'label_smoothing', 'focal_gamma', 'test_auc', 'test_error']]
print(best_configs.to_string(index=False))

print('\n=== Performance by Loss Type ===')
loss_analysis = df.groupby(['loss_type', 'label_smoothing', 'focal_gamma']).agg({
    'test_auc': ['mean', 'std', 'max'],
    'test_error': ['mean', 'std', 'min']
}).round(6)
print(loss_analysis)

print('\n=== Performance by Learning Rate ===')
lr_analysis = df.groupby('lr').agg({
    'test_auc': ['mean', 'std', 'max'],
    'test_error': ['mean', 'std', 'min']
}).round(6)
print(lr_analysis)

print('\n=== Performance by Warmup Ratio ===')
wr_analysis = df.groupby('warmup_ratio').agg({
    'test_auc': ['mean', 'std', 'max'],
    'test_error': ['mean', 'std', 'min']
}).round(6)
print(wr_analysis)

# Calculate improvements over baseline
baseline_auc = 0.9045569044313844  # Best result from the sweep
print(f'\n=== Baseline Comparison ===')
print(f'Best AUC achieved: {baseline_auc:.6f}')
print(f'Improvement over worst: {baseline_auc - df["test_auc"].min():.6f}')
print(f'Standard deviation: {df["test_auc"].std():.6f}')

# Find the best configuration details
best_idx = df['test_auc'].idxmax()
best_config = df.loc[best_idx]
print(f'\n=== Best Configuration Details ===')
print(f'Learning Rate: {best_config["lr"]}')
print(f'Warmup Ratio: {best_config["warmup_ratio"]}')
print(f'Loss Type: {best_config["loss_type"]}')
print(f'Label Smoothing: {best_config["label_smoothing"]}')
print(f'Focal Gamma: {best_config["focal_gamma"]}')
print(f'Test AUC: {best_config["test_auc"]:.6f}')
print(f'Test Error: {best_config["test_error"]:.6f}')

