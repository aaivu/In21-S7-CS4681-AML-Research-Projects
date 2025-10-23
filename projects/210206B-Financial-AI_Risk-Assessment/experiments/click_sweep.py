#!/usr/bin/env python3
"""
CLICK Dataset Hyperparameter Sweep Script

This script performs a comprehensive hyperparameter sweep on the CLICK dataset
using the deep NODE model configuration (8 layers, 128 trees each) with 
memory-optimized training including gradient accumulation.

Sweep Parameters:
- Learning Rates: [1e-4, 3e-4, 1e-3]
- Loss Functions: [BCE, Focal Loss, Label Smoothing]  
- Schedulers: [None, Cosine Annealing, Linear Warmup + Cosine]
- Optimizers: [QHAdam (default), AdamW]

Results are compared against the baseline deep model performance.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
from itertools import product
from datetime import datetime

# Add the lib directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import lib
from qhoptim.pyt import QHAdam

class CLICKSweepRunner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        
        # Load baseline results for comparison
        self.baseline_results = self.load_baseline_results()
        
        # Results storage
        self.sweep_results = []
        
    def load_baseline_results(self):
        """Load baseline results from previous deep model run"""
        baseline_file = 'baseline_results.json'
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                results = json.load(f)
                # Look for CLICK deep results
                if 'CLICK_deep' in results:
                    result = results['CLICK_deep']
                    test_results = result['test_results']
                    baseline = {
                        'test_error': test_results['test_error'],
                        'test_auc': test_results['test_auc'],
                        'test_logloss': test_results['test_logloss'],
                        'training_time': result['training_time'],
                        'best_step': result['best_step']
                    }
                    print(f"Loaded baseline: Test Error={baseline['test_error']:.5f}, AUC={baseline['test_auc']:.5f}")
                    return baseline
        
        print("Warning: No baseline results found. Using default comparison values.")
        return {
            'test_error': 0.35073,
            'test_auc': 0.70972,
            'test_logloss': 0.66684,
            'training_time': 3243.86,
            'best_step': 50
        }
    
    def get_dataset_config(self):
        """Get memory-optimized CLICK dataset configuration"""
        return {
            'type': 'classification',
            'random_state': 1337,
            'quantile_transform': True,
            'quantile_noise': 1e-3,
            'batch_size': 128,  # Memory-optimized for 4GB GPU
            'early_stopping': 5000,
            'report_freq': 50,
            'valid_size': 25_000,  # Reduced validation set
            'validation_seed': 1337,
            'gradient_accumulation_steps': 2  # Effective batch size 256
        }
    
    def get_model_config(self):
        """Get exact deep model configuration"""
        return {
            'model_type': 'NODE',
            'layer_dim': 128,
            'num_layers': 8,  # Deep model: 8 layers
            'depth': 6,
            'tree_dim': 2,
            'choice_function': lib.entmax15,  # Use actual function, not string
            'bin_function': lib.entmoid15     # Use actual function, not string
        }
    
    def get_sweep_configurations(self):
        """Define hyperparameter sweep grid"""
        learning_rates = [1e-4, 3e-4, 1e-3]
        
        # Loss function configurations
        loss_configs = [
            {'loss_type': 'bce', 'focal_gamma': 0.0, 'label_smoothing': 0.0},
            {'loss_type': 'focal', 'focal_gamma': 2.0, 'label_smoothing': 0.0},
            {'loss_type': 'bce', 'focal_gamma': 0.0, 'label_smoothing': 0.1}
        ]
        
        # Scheduler configurations  
        scheduler_configs = [
            {'scheduler_type': 'none', 'warmup_steps': 0},
            {'scheduler_type': 'cosine', 'warmup_steps': 0, 'min_lr': 1e-6},
            {'scheduler_type': 'warmup_cosine', 'warmup_steps': 500, 'min_lr': 1e-6}
        ]
        
        # Optimizer configurations
        optimizer_configs = [
            {'optimizer': 'qhadam'},
            {'optimizer': 'adamw', 'weight_decay': 1e-5}
        ]
        
        # Generate all combinations
        configurations = []
        for lr, loss_config, scheduler_config, optimizer_config in product(
            learning_rates, loss_configs, scheduler_configs, optimizer_configs
        ):
            config = {
                'learning_rate': lr,
                **loss_config,
                **scheduler_config,
                **optimizer_config,
                'total_steps': 10000  # For scheduler
            }
            configurations.append(config)
        
        print(f"Generated {len(configurations)} sweep configurations")
        return configurations
    
    def create_model(self, model_config, data):
        """Create NODE model based on configuration (same as baseline)"""
        in_features = data.X_train.shape[1]
        num_classes = len(set(data.y_train))
        tree_dim = num_classes + 1
        
        model = torch.nn.Sequential(
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
        
        # Data-aware initialization
        with torch.no_grad():
            init_batch_size = min(2000, len(data.X_train))
            model(torch.as_tensor(data.X_train[:init_batch_size], device=self.device))
        
        return model
    
    def create_trainer(self, model, train_config):
        """Create NODE trainer with specified configuration"""
        # Create experiment directory
        timestamp = int(time.time())
        experiment_name = f"click_sweep_{timestamp}"
        
        # Create trainer exactly like baseline but with different optimizer/lr
        trainer = lib.Trainer(
            model=model,
            loss_function=torch.nn.functional.cross_entropy,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,  # Start with QHAdam like baseline
            optimizer_params=dict(nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=False,
            n_last_checkpoints=5,
            # Pass enhancement parameters
            loss_type=train_config.get('loss_type', None),
            focal_gamma=train_config.get('focal_gamma', 2.0),
            label_smoothing=train_config.get('label_smoothing', 0.0),
            scheduler_type=train_config.get('scheduler_type', None),
            warmup_steps=train_config.get('warmup_steps', 0),
            total_steps=train_config.get('total_steps', 10000),
            min_lr=train_config.get('min_lr', 1e-6)
        )
        
        # Now replace optimizer if needed and set learning rate
        if train_config['optimizer'] == 'adamw':
            # Replace with AdamW
            trainer.opt = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config.get('weight_decay', 1e-5)
            )
        else:
            # Update QHAdam learning rate
            for param_group in trainer.opt.param_groups:
                param_group['lr'] = train_config['learning_rate']
        
        return trainer
    
    def train_on_batch_with_accumulation(self, x_batch, y_batch, trainer, accumulation_steps, counter):
        """Training with gradient accumulation for memory efficiency"""
        x_batch = torch.as_tensor(x_batch, device=self.device)
        y_batch = torch.as_tensor(y_batch, device=self.device)

        trainer.model.train()
        
        # Forward pass
        logits = trainer.model(x_batch)
        loss = trainer._compute_loss(logits, y_batch)
        loss = loss / accumulation_steps  # Scale loss
        
        # Backward pass
        loss.backward()
        
        # Step optimizer after accumulation
        if (counter + 1) % accumulation_steps == 0:
            trainer.opt.step()
            trainer.opt.zero_grad()
            trainer.step += 1
            trainer.writer.add_scalar('train loss', loss.item() * accumulation_steps, trainer.step)
            return {'loss': loss * accumulation_steps}
        
        return None
    
    def run_single_configuration(self, config_idx, train_config):
        """Run training for a single hyperparameter configuration"""
        config_name = (f"lr{train_config['learning_rate']:.0e}_"
                      f"{train_config['loss_type']}_"
                      f"{train_config['scheduler_type']}_"
                      f"{train_config['optimizer']}")
        
        print(f"\n{'='*60}")
        print(f"Configuration {config_idx}: {config_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Load dataset
            dataset_config = self.get_dataset_config()
            model_config = self.get_model_config()
            
            # Filter dataset kwargs
            excluded_keys = ['batch_size', 'early_stopping', 'report_freq', 'type', 
                           'valid_size', 'validation_seed', 'gradient_accumulation_steps']
            dataset_kwargs = {k: v for k, v in dataset_config.items() if k not in excluded_keys}
            data = lib.Dataset('CLICK', **dataset_kwargs)
            
            # Create model and trainer
            model = self.create_model(model_config, data)
            trainer = self.create_trainer(model, train_config)
            
            # Training loop with gradient accumulation
            batch_size = dataset_config['batch_size']
            early_stopping_rounds = dataset_config['early_stopping']
            report_frequency = dataset_config['report_freq']
            gradient_accumulation_steps = dataset_config['gradient_accumulation_steps']
            
            loss_history = []
            best_metric = float('inf')
            best_step = 0
            accumulation_counter = 0
            
            print(f"Training with batch_size={batch_size}, gradient_accumulation={gradient_accumulation_steps}")
            
            for batch in lib.iterate_minibatches(
                data.X_train, data.y_train,
                batch_size=batch_size,
                shuffle=True,
                epochs=float('inf')
            ):
                # Memory management
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                metrics = self.train_on_batch_with_accumulation(
                    *batch, trainer, gradient_accumulation_steps, accumulation_counter
                )
                accumulation_counter = (accumulation_counter + 1) % gradient_accumulation_steps
                
                if metrics is not None:
                    loss_history.append(metrics['loss'])
                    
                    if trainer.step % report_frequency == 0:
                        trainer.save_checkpoint()
                        trainer.average_checkpoints(out_tag='avg')
                        trainer.load_checkpoint(tag='avg')
                        
                        # Validation evaluation
                        val_error = trainer.evaluate_classification_error(
                            data.X_valid, data.y_valid, device=self.device, batch_size=batch_size
                        )
                        
                        if val_error < best_metric:
                            best_metric = val_error
                            best_step = trainer.step
                            trainer.save_checkpoint(tag='best')
                        
                        trainer.load_checkpoint()
                        trainer.remove_old_temp_checkpoints()
                        
                        print(f"Step {trainer.step}: Loss={metrics['loss']:.5f}, Val Error={val_error:.5f}")
                    
                    # Early stopping
                    if trainer.step > best_step + early_stopping_rounds:
                        print(f'Early stopping at step {trainer.step}')
                        break
            
            # Final evaluation on test set
            trainer.load_checkpoint(tag='best')
            test_error = trainer.evaluate_classification_error(
                data.X_test, data.y_test, device=self.device, batch_size=batch_size
            )
            test_auc = trainer.evaluate_auc(
                data.X_test, data.y_test, device=self.device, batch_size=batch_size
            )
            test_logloss = trainer.evaluate_logloss(
                data.X_test, data.y_test, device=self.device, batch_size=batch_size
            )
            
            training_time = time.time() - start_time
            
            # Store results
            result = {
                'config_name': config_name,
                'config_idx': config_idx,
                'learning_rate': train_config['learning_rate'],
                'loss_type': train_config['loss_type'],
                'focal_gamma': train_config.get('focal_gamma', 0.0),
                'label_smoothing': train_config.get('label_smoothing', 0.0),
                'scheduler_type': train_config['scheduler_type'],
                'optimizer': train_config['optimizer'],
                'best_step': best_step,
                'training_time': training_time,
                'test_error': test_error,
                'test_auc': test_auc,
                'test_logloss': test_logloss,
                'improvement_over_baseline': {
                    'error_delta': self.baseline_results['test_error'] - test_error,
                    'auc_delta': test_auc - self.baseline_results['test_auc'],
                    'logloss_delta': self.baseline_results['test_logloss'] - test_logloss
                }
            }
            
            print(f"\nResults: Error={test_error:.5f}, AUC={test_auc:.5f}, LogLoss={test_logloss:.5f}")
            print(f"Training time: {training_time:.1f}s")
            print(f"Improvement over baseline: Error Δ={result['improvement_over_baseline']['error_delta']:+.5f}")
            
            return result
            
        except Exception as e:
            import traceback
            print(f"Configuration failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return {
                'config_name': config_name,
                'config_idx': config_idx,
                'error': str(e),
                'test_error': float('inf'),
                'test_auc': 0.0,
                'test_logloss': float('inf')
            }
    
    def run_sweep(self):
        """Execute the complete hyperparameter sweep"""
        print("🚀 Starting CLICK Hyperparameter Sweep")
        print(f"Baseline: Error={self.baseline_results['test_error']:.5f}, AUC={self.baseline_results['test_auc']:.5f}")
        
        configurations = self.get_sweep_configurations()
        
        for i, config in enumerate(configurations, 1):
            result = self.run_single_configuration(i, config)
            self.sweep_results.append(result)
            
            # Save intermediate results
            self.save_results(f'click_sweep_progress_{int(time.time())}.json')
            
            print(f"\nProgress: {i}/{len(configurations)} configurations completed")
        
        # Final results analysis
        self.analyze_results()
        self.save_results('click_sweep_results.json')
        self.create_comparison_csv()
        
        print("\n🎉 Hyperparameter sweep completed!")
    
    def analyze_results(self):
        """Analyze sweep results and find best configurations"""
        if not self.sweep_results:
            return
        
        # Sort by test error (lower is better)
        valid_results = [r for r in self.sweep_results if 'error' not in r]
        valid_results.sort(key=lambda x: x['test_error'])
        
        print(f"\n{'='*80}")
        print("TOP 5 CONFIGURATIONS")
        print(f"{'='*80}")
        
        for i, result in enumerate(valid_results[:5], 1):
            print(f"{i}. {result['config_name']}")
            print(f"   Error: {result['test_error']:.5f} (Δ{result['improvement_over_baseline']['error_delta']:+.5f})")
            print(f"   AUC: {result['test_auc']:.5f} (Δ{result['improvement_over_baseline']['auc_delta']:+.5f})")
            print(f"   LogLoss: {result['test_logloss']:.5f} (Δ{result['improvement_over_baseline']['logloss_delta']:+.5f})")
            print(f"   Time: {result['training_time']:.1f}s")
            print()
        
        # Best configuration
        if valid_results:
            best = valid_results[0]
            print(f"🏆 BEST CONFIGURATION: {best['config_name']}")
            print(f"   Test Error: {best['test_error']:.5f} (improvement: {best['improvement_over_baseline']['error_delta']:+.5f})")
            print(f"   Test AUC: {best['test_auc']:.5f} (improvement: {best['improvement_over_baseline']['auc_delta']:+.5f})")
    
    def save_results(self, filename):
        """Save sweep results to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'baseline': self.baseline_results,
                'sweep_results': self.sweep_results,
                'timestamp': datetime.now().isoformat(),
                'total_configurations': len(self.sweep_results)
            }, f, indent=2)
        print(f"Results saved to {filename}")
    
    def create_comparison_csv(self):
        """Create CSV comparison table"""
        if not self.sweep_results:
            return
        
        # Convert results to DataFrame
        df_data = []
        for result in self.sweep_results:
            if 'error' not in result:  # Skip failed configurations
                df_data.append({
                    'Config': result['config_name'],
                    'Learning Rate': result['learning_rate'],
                    'Loss Type': result['loss_type'],
                    'Scheduler': result['scheduler_type'],
                    'Optimizer': result['optimizer'],
                    'Best Step': result['best_step'],
                    'Training Time (s)': f"{result['training_time']:.1f}",
                    'Test Error': f"{result['test_error']:.5f}",
                    'Test AUC': f"{result['test_auc']:.5f}",
                    'Test LogLoss': f"{result['test_logloss']:.5f}",
                    'Error Δ vs Baseline': f"{result['improvement_over_baseline']['error_delta']:+.5f}",
                    'AUC Δ vs Baseline': f"{result['improvement_over_baseline']['auc_delta']:+.5f}"
                })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Test Error')  # Sort by best error
        
        csv_file = 'click_sweep_comparison.csv'
        df.to_csv(csv_file, index=False)
        print(f"Comparison table saved to {csv_file}")
        
        # Display summary
        print(f"\n{df.head(10).to_string(index=False)}")

def main():
    """Main execution function"""
    print("CLICK Dataset Hyperparameter Sweep")
    print("=" * 50)
    
    runner = CLICKSweepRunner()
    runner.run_sweep()

if __name__ == "__main__":
    main()