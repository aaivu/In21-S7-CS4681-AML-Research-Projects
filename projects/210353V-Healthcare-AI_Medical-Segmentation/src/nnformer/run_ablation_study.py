"""
Ablation Study Runner for Enhanced nnFormer
Automates training and evaluation across different configurations
"""

import os
import subprocess
import argparse
import json
import time
from pathlib import Path
import pandas as pd


class AblationStudy:
    """
    Manages ablation study experiments for Enhanced nnFormer
    """
    
    def __init__(self, cuda_device=0, task_id=4, base_output_dir='./ablation_results'):
        self.cuda_device = cuda_device
        self.task_id = task_id
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Ablation configurations
        self.configs = {
            'baseline': {
                'enable_cross_attention': False,
                'enable_adaptive_fusion': False,
                'enable_enhanced_training': False,
                'description': 'Baseline nnFormer without enhancements'
            },
            'cross_attn': {
                'enable_cross_attention': True,
                'enable_adaptive_fusion': False,
                'enable_enhanced_training': False,
                'description': 'Baseline + Multi-scale cross-attention'
            },
            'fusion': {
                'enable_cross_attention': True,
                'enable_adaptive_fusion': True,
                'enable_enhanced_training': False,
                'description': 'Baseline + Cross-attention + Adaptive fusion'
            },
            'training': {
                'enable_cross_attention': True,
                'enable_adaptive_fusion': False,
                'enable_enhanced_training': True,
                'description': 'Baseline + Cross-attention + Enhanced training'
            },
            'full': {
                'enable_cross_attention': True,
                'enable_adaptive_fusion': True,
                'enable_enhanced_training': True,
                'description': 'Full enhancement (all components)'
            }
        }
        
        self.results = {}
    
    def run_training(self, config_name, num_folds=1, max_epochs=1000):
        """
        Run training for a specific configuration
        
        Args:
            config_name: Name of ablation configuration
            num_folds: Number of cross-validation folds
            max_epochs: Maximum training epochs
        """
        if config_name not in self.configs:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        config = self.configs[config_name]
        output_dir = self.base_output_dir / config_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*80}")
        print(f"Training configuration: {config_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}\n")
        
        # Save configuration
        config_file = output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training for each fold
        for fold in range(num_folds):
            print(f"\nTraining fold {fold}...")
            
            cmd = [
                'bash', 'train_inference_brats.sh',
                '-c', str(self.cuda_device),
                '-n', config_name,
                '-t', str(self.task_id),
                '-r',
                '-a', config_name
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Log output
                log_file = output_dir / f'training_fold{fold}.log'
                with open(log_file, 'w') as f:
                    f.write(result.stdout)
                    f.write(result.stderr)
                
                print(f"Fold {fold} completed successfully")
                
            except subprocess.CalledProcessError as e:
                print(f"Error training fold {fold}: {e}")
                log_file = output_dir / f'training_fold{fold}_error.log'
                with open(log_file, 'w') as f:
                    f.write(e.stdout)
                    f.write(e.stderr)
    
    def run_inference(self, config_name, num_folds=1):
        """
        Run inference for a specific configuration
        
        Args:
            config_name: Name of ablation configuration
            num_folds: Number of cross-validation folds
        """
        if config_name not in self.configs:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        output_dir = self.base_output_dir / config_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*80}")
        print(f"Running inference: {config_name}")
        print(f"{'='*80}\n")
        
        for fold in range(num_folds):
            print(f"\nInference fold {fold}...")
            
            cmd = [
                'bash', 'train_inference_brats.sh',
                '-c', str(self.cuda_device),
                '-n', config_name,
                '-t', str(self.task_id),
                '-p',
                '-a', config_name
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Log output
                log_file = output_dir / f'inference_fold{fold}.log'
                with open(log_file, 'w') as f:
                    f.write(result.stdout)
                    f.write(result.stderr)
                
                print(f"Fold {fold} inference completed")
                
            except subprocess.CalledProcessError as e:
                print(f"Error in inference fold {fold}: {e}")
    
    def evaluate_all(self):
        """
        Evaluate all configurations and generate comparison
        """
        print(f"\n{'='*80}")
        print("Evaluating all configurations")
        print(f"{'='*80}\n")
        
        results = {}
        
        for config_name in self.configs.keys():
            config_dir = self.base_output_dir / config_name
            results_file = config_dir / 'predictions' / 'results.csv'
            
            if results_file.exists():
                df = pd.read_csv(results_file)
                results[config_name] = df
                
                print(f"\n{config_name}:")
                print(f"  Dice WT: {df['dice_wt'].mean():.4f} ± {df['dice_wt'].std():.4f}")
                print(f"  Dice TC: {df['dice_tc'].mean():.4f} ± {df['dice_tc'].std():.4f}")
                print(f"  Dice ET: {df['dice_et'].mean():.4f} ± {df['dice_et'].std():.4f}")
                print(f"  HD95 Avg: {(df['hd95_wt'].mean() + df['hd95_tc'].mean() + df['hd95_et'].mean())/3:.2f}")
            else:
                print(f"\nWarning: Results not found for {config_name}")
        
        # Create comparison table
        if len(results) > 0:
            comparison = []
            
            for config_name, df in results.items():
                row = {
                    'Configuration': config_name,
                    'Description': self.configs[config_name]['description'],
                    'Dice_WT': f"{df['dice_wt'].mean():.3f}±{df['dice_wt'].std():.3f}",
                    'Dice_TC': f"{df['dice_tc'].mean():.3f}±{df['dice_tc'].std():.3f}",
                    'Dice_ET': f"{df['dice_et'].mean():.3f}±{df['dice_et'].std():.3f}",
                    'HD95_Avg': f"{(df['hd95_wt'].mean() + df['hd95_tc'].mean() + df['hd95_et'].mean())/3:.1f}"
                }
                comparison.append(row)
            
            comp_df = pd.DataFrame(comparison)
            
            # Save comparison
            comp_file = self.base_output_dir / 'ablation_comparison.csv'
            comp_df.to_csv(comp_file, index=False)
            
            print(f"\n{'='*80}")
            print("ABLATION STUDY COMPARISON")
            print(f"{'='*80}")
            print(comp_df.to_string(index=False))
            print(f"\nComparison saved to {comp_file}")
            
            # Create LaTeX table
            latex_file = self.base_output_dir / 'ablation_table.tex'
            with open(latex_file, 'w') as f:
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Ablation Study Results on BraTS 2021 Validation Set}\n")
                f.write("\\label{tab:ablation}\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("Configuration & Dice WT & Dice TC & Dice ET & HD95 Avg \\\\\n")
                f.write("\\hline\n")
                
                for _, row in comp_df.iterrows():
                    f.write(f"{row['Configuration']} & {row['Dice_WT']} & {row['Dice_TC']} & "
                           f"{row['Dice_ET']} & {row['HD95_Avg']} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            
            print(f"LaTeX table saved to {latex_file}")
    
    def run_full_ablation(self, configs_to_run=None, num_folds=1, max_epochs=1000):
        """
        Run complete ablation study: train, infer, and evaluate
        
        Args:
            configs_to_run: List of config names to run (None = all)
            num_folds: Number of cross-validation folds
            max_epochs: Maximum training epochs
        """
        if configs_to_run is None:
            configs_to_run = list(self.configs.keys())
        
        print(f"\n{'='*80}")
        print("ABLATION STUDY: Enhanced nnFormer")
        print(f"{'='*80}")
        print(f"Configurations to run: {', '.join(configs_to_run)}")
        print(f"Number of folds: {num_folds}")
        print(f"Max epochs: {max_epochs}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # Run each configuration
        for config_name in configs_to_run:
            config_start = time.time()
            
            print(f"\n{'='*80}")
            print(f"Processing configuration: {config_name}")
            print(f"{'='*80}\n")
            
            # Training
            self.run_training(config_name, num_folds, max_epochs)
            
            # Inference
            self.run_inference(config_name, num_folds)
            
            config_time = time.time() - config_start
            print(f"\nConfiguration {config_name} completed in {config_time/3600:.2f} hours")
        
        # Evaluate all
        self.evaluate_all()
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Ablation study completed in {total_time/3600:.2f} hours")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for Enhanced nnFormer')
    
    parser.add_argument('--cuda', type=int, default=0,
                       help='CUDA device ID')
    parser.add_argument('--task', type=int, default=4,
                       help='Task ID (4 for BraTS)')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                       help='Output directory for results')
    parser.add_argument('--configs', nargs='+', default=None,
                       help='Configurations to run (default: all)')
    parser.add_argument('--num_folds', type=int, default=1,
                       help='Number of cross-validation folds')
    parser.add_argument('--max_epochs', type=int, default=1000,
                       help='Maximum training epochs')
    parser.add_argument('--train_only', action='store_true',
                       help='Only run training')
    parser.add_argument('--infer_only', action='store_true',
                       help='Only run inference')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation')
    
    args = parser.parse_args()
    
    # Create ablation study manager
    study = AblationStudy(
        cuda_device=args.cuda,
        task_id=args.task,
        base_output_dir=args.output_dir
    )
    
    # Determine which configs to run
    configs_to_run = args.configs if args.configs else list(study.configs.keys())
    
    # Run study
    if args.eval_only:
        study.evaluate_all()
    elif args.train_only:
        for config in configs_to_run:
            study.run_training(config, args.num_folds, args.max_epochs)
    elif args.infer_only:
        for config in configs_to_run:
            study.run_inference(config, args.num_folds)
    else:
        study.run_full_ablation(configs_to_run, args.num_folds, args.max_epochs)


if __name__ == '__main__':
    main()
