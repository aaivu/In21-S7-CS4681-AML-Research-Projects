#!/usr/bin/env python3
"""
Baseline Evaluation Script for Neural Oblivious Decision Ensembles (NODE)

This script runs comprehensive baseline experiments on multiple datasets
to establish performance benchmarks before implementing enhancements.

Usage:
    python baseline_evaluation.py --dataset EPSILON --config shallow
    python baseline_evaluation.py --dataset YEAR --config deep
    python baseline_evaluation.py --all  # Run all configurations
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from IPython.display import clear_output

# Add lib to path
sys.path.insert(0, '.')
import lib
from qhoptim.pyt import QHAdam


class BaselineEvaluator:
    """Comprehensive baseline evaluation for NODE architecture"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def get_dataset_configs(self):
        """Define dataset configurations for evaluation"""
        return {
            # Classification datasets
            'EPSILON': {
                'type': 'classification',
                'random_state': 1337,
                'quantile_transform': True,
                'quantile_noise': 1e-3,
                'batch_size': 1024,
                'early_stopping': 10000,
                'report_freq': 100
            },
            'HIGGS': {
                'type': 'classification', 
                'random_state': 1337,
                'quantile_transform': True,
                'quantile_noise': 1e-3,
                'batch_size': 1024,
                'early_stopping': 10000,
                'report_freq': 100,
                'train_size': 2_000_000,
                'valid_size': 100_000
            },
            'A9A': {
                'type': 'classification',
                'random_state': 1337,
                'quantile_transform': True,
                'quantile_noise': 1e-3,
                'batch_size': 512,
                'early_stopping': 5000,
                'report_freq': 50
            },
            'CLICK': {
                'type': 'classification',
                'random_state': 1337,
                'quantile_transform': True,
                'quantile_noise': 1e-3,
                'batch_size': 128,  # Further reduced for deep model
                'early_stopping': 5000,
                'report_freq': 50,
                'valid_size': 25_000,  # Further reduced validation set
                'validation_seed': 1337,
                'gradient_accumulation_steps': 2  # Simulate larger batches
            },
            # Regression datasets
            'YEAR': {
                'type': 'regression',
                'random_state': 1337,
                'quantile_transform': True,
                'quantile_noise': 1e-3,
                'batch_size': 1024,
                'early_stopping': 5000,
                'report_freq': 100,
                'normalize_target': True
            }
        }
    
    def get_model_configs(self):
        """Define model configurations for different NODE variants"""
        return {
            'shallow': {
                'description': 'Single layer NODE (2048 trees)',
                'layer_dim': 2048,
                'num_layers': 1,
                'depth': 6,
                'tree_dim': 3,
                'choice_function': lib.entmax15,
                'bin_function': lib.entmoid15
            },
            'deep': {
                'description': 'Multi-layer NODE (8 layers, 128 trees each)',
                'layer_dim': 128,
                'num_layers': 8,
                'depth': 6,
                'tree_dim': 3,
                'choice_function': lib.entmax15,
                'bin_function': lib.entmoid15
            },
            'click_optimized': {
                'description': 'Memory-optimized NODE for Click (4 layers, 64 trees each)',
                'layer_dim': 64,
                'num_layers': 4,
                'depth': 5,  # Reduced depth
                'tree_dim': 2,  # Reduced tree dimension
                'choice_function': lib.entmax15,
                'bin_function': lib.entmoid15
            },
            'medium': {
                'description': 'Medium NODE (2 layers, 1024 trees each)',
                'layer_dim': 1024,
                'num_layers': 2,
                'depth': 6,
                'tree_dim': 3,
                'choice_function': lib.entmax15,
                'bin_function': lib.entmoid15
            }
        }
    
    def create_model(self, dataset_config, model_config, data):
        """Create NODE model based on configuration"""
        in_features = data.X_train.shape[1]
        
        if dataset_config['type'] == 'classification':
            num_classes = len(set(data.y_train))
            tree_dim = num_classes + 1
            
            model = nn.Sequential(
                lib.DenseBlock(
                    in_features, 
                    model_config['layer_dim'], 
                    num_layers=model_config['num_layers'],
                    tree_dim=tree_dim,
                    depth=model_config['depth'],
                    flatten_output=False,
                    choice_function=model_config['choice_function'],
                    bin_function=model_config['bin_function']
                ),
                lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2))
            ).to(self.device)
            
        else:  # regression
            tree_dim = model_config['tree_dim']
            
            model = nn.Sequential(
                lib.DenseBlock(
                    in_features,
                    model_config['layer_dim'],
                    num_layers=model_config['num_layers'],
                    tree_dim=tree_dim,
                    depth=model_config['depth'],
                    flatten_output=False,
                    choice_function=model_config['choice_function'],
                    bin_function=model_config['bin_function']
                ),
                lib.Lambda(lambda x: x[..., 0].mean(dim=-1))
            ).to(self.device)
        
        # Data-aware initialization
        with torch.no_grad():
            init_batch_size = min(2000, len(data.X_train))
            model(torch.as_tensor(data.X_train[:init_batch_size], device=self.device))
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        return model
    
    def create_trainer(self, model, dataset_config, experiment_name):
        """Create trainer with appropriate loss function and optimizer"""
        if dataset_config['type'] == 'classification':
            loss_function = F.cross_entropy
        else:
            loss_function = F.mse_loss
            
        trainer = lib.Trainer(
            model=model,
            loss_function=loss_function,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=True,
            n_last_checkpoints=5
        )
        
        return trainer
    
    def train_model(self, trainer, data, dataset_config, model_config):
        """Train the model with early stopping and evaluation"""
        batch_size = dataset_config['batch_size']
        early_stopping_rounds = dataset_config['early_stopping']
        report_frequency = dataset_config['report_freq']
        gradient_accumulation_steps = dataset_config.get('gradient_accumulation_steps', 1)

        loss_history = []
        metric_history = []
        best_metric = float('inf')
        best_step = 0

        print(f"Starting training with batch_size={batch_size}, early_stopping={early_stopping_rounds}")
        if gradient_accumulation_steps > 1:
            print(f"Using gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {batch_size * gradient_accumulation_steps})")

        accumulation_counter = 0
        for batch in lib.iterate_minibatches(
            data.X_train, data.y_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=float('inf')
        ):
            # Memory management for large datasets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            metrics = self.train_on_batch_with_accumulation(*batch, trainer, gradient_accumulation_steps, accumulation_counter)
            accumulation_counter = (accumulation_counter + 1) % gradient_accumulation_steps
            
            if metrics is not None:  # Only when we actually stepped
                loss_history.append(metrics['loss'])

                if trainer.step % report_frequency == 0:
                    trainer.save_checkpoint()
                    trainer.average_checkpoints(out_tag='avg')
                    trainer.load_checkpoint(tag='avg')
                    
                    # Evaluate on validation set
                    if dataset_config['type'] == 'classification':
                        metric = trainer.evaluate_classification_error(
                            data.X_valid, data.y_valid, device=self.device, batch_size=batch_size
                        )
                    else:
                        metric = trainer.evaluate_mse(
                            data.X_valid, data.y_valid, device=self.device, batch_size=batch_size*4
                        )
                    
                    if metric < best_metric:
                        best_metric = metric
                        best_step = trainer.step
                        trainer.save_checkpoint(tag='best')
                    
                    metric_history.append(metric)
                    trainer.load_checkpoint()
                    trainer.remove_old_temp_checkpoints()
                    
                    print(f"Step {trainer.step}: Loss={metrics['loss']:.5f}, "
                          f"Val {'Error' if dataset_config['type'] == 'classification' else 'MSE'}={metric:.5f}")
            
            # Check early stopping only when we have actual steps
            if metrics is not None and trainer.step > best_step + early_stopping_rounds:
                print(f'Early stopping: No improvement for {early_stopping_rounds} steps')
                print(f"Best step: {best_step}")
                print(f"Best Val {'Error' if dataset_config['type'] == 'classification' else 'MSE'}: {best_metric:.5f}")
                break
        
        return best_step, best_metric, loss_history, metric_history
    
    def train_on_batch_with_accumulation(self, x_batch, y_batch, trainer, accumulation_steps, counter):
        """Training with gradient accumulation to simulate larger batches"""
        x_batch = torch.as_tensor(x_batch, device=self.device)
        y_batch = torch.as_tensor(y_batch, device=self.device)

        trainer.model.train()
        
        # Forward pass
        logits = trainer.model(x_batch)
        loss = trainer._compute_loss(logits, y_batch)
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        # Only step optimizer and zero gradients after accumulation_steps
        if (counter + 1) % accumulation_steps == 0:
            trainer.opt.step()
            trainer.opt.zero_grad()
            trainer.step += 1
            trainer.writer.add_scalar('train loss', loss.item() * accumulation_steps, trainer.step)
            return {'loss': loss * accumulation_steps}  # Return unscaled loss for logging
        
        return None  # No metrics when not stepping
    
    def evaluate_model(self, trainer, data, dataset_config):
        """Evaluate the trained model on test set"""
        trainer.load_checkpoint(tag='best')
        
        if dataset_config['type'] == 'classification':
            test_error = trainer.evaluate_classification_error(
                data.X_test, data.y_test, device=self.device, batch_size=1024
            )
            test_auc = trainer.evaluate_auc(
                data.X_test, data.y_test, device=self.device, batch_size=512
            )
            test_logloss = trainer.evaluate_logloss(
                data.X_test, data.y_test, device=self.device, batch_size=512
            )
            
            return {
                'test_error': test_error,
                'test_auc': test_auc,
                'test_logloss': test_logloss
            }
        else:
            test_mse = trainer.evaluate_mse(
                data.X_test, data.y_test, device=self.device, batch_size=1024
            )
            
            return {
                'test_mse': test_mse
            }
    
    def run_single_experiment(self, dataset_name, model_config_name, save_results=True):
        """Run a single experiment with given dataset and model configuration"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {dataset_name} + {model_config_name}")
        print(f"{'='*60}")
        
        # Get configurations
        dataset_configs = self.get_dataset_configs()
        model_configs = self.get_model_configs()
        
        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if model_config_name not in model_configs:
            raise ValueError(f"Unknown model config: {model_config_name}")
            
        dataset_config = dataset_configs[dataset_name]
        model_config = model_configs[model_config_name]
        
        # Load data
        print(f"Loading dataset: {dataset_name}")
        # Filter out keys that are not accepted by Dataset/fetch_* factories
        excluded_keys = ['batch_size', 'early_stopping', 'report_freq', 'type', 'normalize_target', 'valid_size', 'validation_seed', 'gradient_accumulation_steps']
        dataset_kwargs = {k: v for k, v in dataset_config.items() if k not in excluded_keys}
        data = lib.Dataset(dataset_name, **dataset_kwargs)
        
        # Normalize target for regression
        if dataset_config['type'] == 'regression' and dataset_config.get('normalize_target', False):
            mu, std = data.y_train.mean(), data.y_train.std()
            normalize = lambda x: ((x - mu) / std).astype(np.float32)
            data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])
            print(f"Target normalized: mean={mu:.5f}, std={std:.5f}")
        
        # Create model
        print(f"Creating model: {model_config['description']}")
        model = self.create_model(dataset_config, model_config, data)
        
        # Create experiment name
        experiment_name = f"{dataset_name.lower()}_{model_config_name}_{int(time.time())}"
        
        # Create trainer
        trainer = self.create_trainer(model, dataset_config, experiment_name)
        
        # Train model
        print("Starting training...")
        start_time = time.time()
        best_step, best_val_metric, loss_history, metric_history = self.train_model(
            trainer, data, dataset_config, model_config
        )
        training_time = time.time() - start_time
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = self.evaluate_model(trainer, data, dataset_config)
        
        # Compile results
        results = {
            'dataset': dataset_name,
            'model_config': model_config_name,
            'model_description': model_config['description'],
            'best_step': best_step,
            'best_val_metric': best_val_metric,
            'training_time': training_time,
            'test_results': test_results,
            'dataset_info': {
                'train_size': len(data.X_train),
                'valid_size': len(data.X_valid),
                'test_size': len(data.X_test),
                'num_features': data.X_train.shape[1]
            }
        }
        
        # Print summary
        print(f"\n{'='*40}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*40}")
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_config['description']}")
        print(f"Best validation step: {best_step}")
        print(f"Training time: {training_time:.2f}s")
        
        if dataset_config['type'] == 'classification':
            print(f"Test Error Rate: {test_results['test_error']:.5f}")
            print(f"Test AUC: {test_results['test_auc']:.5f}")
            print(f"Test LogLoss: {test_results['test_logloss']:.5f}")
        else:
            print(f"Test MSE: {test_results['test_mse']:.5f}")
        
        # Save results
        if save_results:
            self.results[f"{dataset_name}_{model_config_name}"] = results
            self.save_results()
        
        return results
    
    def run_all_experiments(self):
        """Run all dataset and model configuration combinations"""
        dataset_configs = self.get_dataset_configs()
        model_configs = self.get_model_configs()
        
        print(f"Running {len(dataset_configs)} datasets × {len(model_configs)} models = "
              f"{len(dataset_configs) * len(model_configs)} experiments")
        
        for dataset_name in dataset_configs.keys():
            for model_config_name in model_configs.keys():
                try:
                    self.run_single_experiment(dataset_name, model_config_name)
                except Exception as e:
                    print(f"Error in {dataset_name}_{model_config_name}: {e}")
                    continue
    
    def save_results(self, filename='baseline_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def create_comparison_table(self):
        """Create comparison table of all results"""
        if not self.results:
            print("No results to compare")
            return
        
        # Create DataFrame
        rows = []
        for key, result in self.results.items():
            row = {
                'Dataset': result['dataset'],
                'Model': result['model_config'],
                'Description': result['model_description'],
                'Best Step': result['best_step'],
                'Training Time (s)': result['training_time']
            }
            
            if 'test_error' in result['test_results']:
                row.update({
                    'Test Error': result['test_results']['test_error'],
                    'Test AUC': result['test_results']['test_auc'],
                    'Test LogLoss': result['test_results']['test_logloss']
                })
            else:
                row['Test MSE'] = result['test_results']['test_mse']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        df.to_csv('baseline_comparison.csv', index=False)
        print("\nComparison table saved to baseline_comparison.csv")
        print(df.to_string(index=False))
        
        return df


def main():
    parser = argparse.ArgumentParser(description='NODE Baseline Evaluation')
    parser.add_argument('--dataset', type=str, help='Dataset name (EPSILON, HIGGS, A9A, YEAR)')
    parser.add_argument('--config', type=str, help='Model config (shallow, deep, medium)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create evaluator
    evaluator = BaselineEvaluator(device=device)
    
    if args.all:
        evaluator.run_all_experiments()
    elif args.dataset and args.config:
        evaluator.run_single_experiment(args.dataset, args.config)
    else:
        print("Please specify --dataset and --config, or use --all")
        return
    
    # Create comparison table
    evaluator.create_comparison_table()


if __name__ == '__main__':
    main()

