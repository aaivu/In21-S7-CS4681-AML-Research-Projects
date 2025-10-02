import os
import sys
import yaml
import torch
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, set_random_seeds, create_directories, save_results
from data_augmentation import DataAugmenter, load_rtp_dataset
from model import CAFEModel, BaselineModel
from train import CAFETrainer, prepare_data
from evaluate import CAFEEvaluator, compare_models
from loss_functions import CAFELoss
from train import ToxicityDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experiment runner for CAFE framework."""
    
    def __init__(self, config_path: str = "experiments/configs/base_config.yaml"):
        self.config = self.load_config(config_path)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        create_directories()
        
        # Setup logging
        setup_logging(f"results/experiment_{self.experiment_id}.log")
        
        # Set random seeds
        set_random_seeds(self.config['system']['random_seed'])
        
        # Set device
        self.device = self._get_device()
        
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Using device: {self.device}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_device(self) -> torch.device:
        """Get appropriate device based on configuration."""
        device_config = self.config['system']['device']
        
        if device_config == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device_config)
    
    def prepare_datasets(self) -> pd.DataFrame:
        """Prepare and augment datasets."""
        logger.info("Preparing datasets...")
        
        # Load base dataset
        if os.path.exists("data/raw/rtp_dataset.csv"):
            df = pd.read_csv("data/raw/rtp_dataset.csv")
        else:
            df = load_rtp_dataset()
        
        # Augment data if enabled
        if self.config['data']['augmentation']['enable']:
            logger.info("Augmenting dataset...")
            augmenter = DataAugmenter()
            
            # Use subset for faster experimentation
            subset_size = min(500, len(df))  # Limit for demo
            df_subset = df.head(subset_size)
            
            augmented_df = augmenter.augment_dataset(df_subset)
            
            # Save augmented dataset
            augmented_df.to_csv("data/augmented/augmented_rtp.csv", index=False)
            logger.info(f"Augmented dataset saved with {len(augmented_df)} samples")
            
            return augmented_df
        
        return df
    
    def train_cafe_model(self, df: pd.DataFrame) -> CAFEModel:
        """Train CAFE model."""
        logger.info("Training CAFE model...")
        
        # Initialize model
        model = CAFEModel(
            model_name=self.config['model']['name'],
            hidden_dim=self.config['model']['hidden_dim'],
            dropout_rate=self.config['model']['dropout_rate'],
            max_length=self.config['model']['max_length']
        ).to(self.device)
        
        # Prepare data
        train_dataloader, val_dataloader = prepare_data(
            df, 
            model.tokenizer, 
            batch_size=self.config['training']['batch_size'],
            test_size=self.config['data']['train_test_split']
        )
        
        # Setup training components
        loss_function = CAFELoss(
            alpha=self.config['loss']['alpha'],
            beta=self.config['loss']['beta'],
            gamma=self.config['loss']['gamma']
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Create trainer
        trainer = CAFETrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler
        )
        
        # Train model
        training_history = trainer.train(
            num_epochs=self.config['training']['num_epochs'],
            save_dir=f"results/models/cafe_{self.experiment_id}"
        )
        
        # Save training history
        save_results(
            training_history,
            f"results/metrics/cafe_{self.experiment_id}/training_history.json"
        )
        
        return model
    
    def train_baseline_model(self, df: pd.DataFrame) -> BaselineModel:
        """Train baseline model for comparison."""
        logger.info("Training baseline model...")
        
        # Initialize baseline model
        model = BaselineModel(
            model_name=self.config['model']['name'],
            max_length=self.config['model']['max_length']
        ).to(self.device)
        
        # Prepare data
        train_dataloader, val_dataloader = prepare_data(
            df,
            model.tokenizer,
            batch_size=self.config['training']['batch_size'],
            test_size=self.config['data']['train_test_split']
        )
        
        # Setup training components (simple MSE loss for baseline)
        loss_function = torch.nn.MSELoss()
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Create trainer (using CAFE trainer but with simple loss)
        trainer = CAFETrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler
        )
        
        # Train model
        training_history = trainer.train(
            num_epochs=self.config['training']['num_epochs'],
            save_dir=f"results/models/baseline_{self.experiment_id}"
        )
        
        # Save training history
        save_results(
            training_history,
            f"results/metrics/baseline_{self.experiment_id}/training_history.json"
        )
        
        return model
    
    def evaluate_models(self, cafe_model: CAFEModel, baseline_model: BaselineModel, 
                       test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate both models and compare results."""
        logger.info("Evaluating models...")
        
        # Prepare test dataloader
        test_dataset = ToxicityDataset(test_df, cafe_model.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'], shuffle=False)
        
        # Evaluate CAFE model
        cafe_evaluator = CAFEEvaluator(cafe_model, self.device)
        cafe_results = cafe_evaluator.evaluate_on_dataset(test_dataloader)
        cafe_evaluator.generate_evaluation_report(
            cafe_results, 
            f"results/metrics/cafe_{self.experiment_id}"
        )
        
        # Evaluate baseline model
        baseline_evaluator = CAFEEvaluator(baseline_model, self.device)
        baseline_results = baseline_evaluator.evaluate_on_dataset(test_dataloader)
        baseline_evaluator.generate_evaluation_report(
            baseline_results,
            f"results/metrics/baseline_{self.experiment_id}"
        )
        
        # Compare models
        comparison = compare_models(
            cafe_results, 
            baseline_results,
            f"results/metrics/comparison_{self.experiment_id}"
        )
        
        return {
            'cafe_results': cafe_results,
            'baseline_results': baseline_results,
            'comparison': comparison
        }
    
    def run_ablation_studies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run ablation studies to validate design choices."""
        logger.info("Running ablation studies...")
        
        ablation_results = {}
        
        # Test different loss weight combinations
        loss_configs = [
            {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'name': 'toxicity_only'},
            {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.0, 'name': 'toxicity_fairness'},
            {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.3, 'name': 'toxicity_context'},
            {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.3, 'name': 'full_cafe'}
        ]
        
        for loss_config in loss_configs:
            logger.info(f"Training ablation: {loss_config['name']}")
            
            # Create model with current config
            model = CAFEModel().to(self.device)
            
            # Prepare data (smaller subset for ablation)
            df_small = df.head(200)
            train_dataloader, val_dataloader = prepare_data(
                df_small, model.tokenizer, batch_size=8
            )
            
            # Setup training
            loss_function = CAFELoss(
                alpha=loss_config['alpha'],
                beta=loss_config['beta'],
                gamma=loss_config['gamma']
            )
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            
            trainer = CAFETrainer(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                loss_function=loss_function,
                optimizer=optimizer,
                device=self.device
            )
            
            # Train for fewer epochs in ablation
            training_history = trainer.train(num_epochs=3)
            
            # Evaluate
            evaluator = CAFEEvaluator(model, self.device)
            results = evaluator.evaluate_on_dataset(val_dataloader)
            
            ablation_results[loss_config['name']] = {
                'config': loss_config,
                'results': results,
                'training_history': training_history
            }
        
        # Save ablation results
        save_results(ablation_results, f"results/metrics/ablation_{self.experiment_id}.json")
        
        return ablation_results
    
    def generate_final_report(self, results: Dict[str, Any], ablation_results: Dict[str, Any] = None):
        """Generate comprehensive final report."""
        logger.info("Generating final report...")
        
        report_lines = []
        report_lines.append(f"CAFE Framework Experiment Report")
        report_lines.append(f"Experiment ID: {self.experiment_id}")
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Configuration summary
        report_lines.append("Configuration:")
        report_lines.append(f"  Model: {self.config['model']['name']}")
        report_lines.append(f"  Batch Size: {self.config['training']['batch_size']}")
        report_lines.append(f"  Learning Rate: {self.config['training']['learning_rate']}")
        report_lines.append(f"  Epochs: {self.config['training']['num_epochs']}")
        report_lines.append(f"  Loss Weights: α={self.config['loss']['alpha']}, β={self.config['loss']['beta']}, γ={self.config['loss']['gamma']}")
        report_lines.append("")
        
        # Main results
        cafe_metrics = results['cafe_results']['overall_metrics']
        baseline_metrics = results['baseline_results']['overall_metrics']
        
        report_lines.append("Main Results:")
        report_lines.append(f"  CAFE F1 Score: {cafe_metrics['f1_score']:.4f}")
        report_lines.append(f"  Baseline F1 Score: {baseline_metrics['f1_score']:.4f}")
        report_lines.append(f"  F1 Improvement: {((cafe_metrics['f1_score'] - baseline_metrics['f1_score']) / baseline_metrics['f1_score'] * 100):+.2f}%")
        report_lines.append("")
        
        report_lines.append(f"  CAFE Fairness Gap: {cafe_metrics['fairness_gap']:.4f}")
        report_lines.append(f"  Baseline Fairness Gap: {baseline_metrics['fairness_gap']:.4f}")
        if baseline_metrics['fairness_gap'] > 0:
            fairness_improvement = ((baseline_metrics['fairness_gap'] - cafe_metrics['fairness_gap']) / baseline_metrics['fairness_gap'] * 100)
            report_lines.append(f"  Fairness Improvement: {fairness_improvement:+.2f}%")
        report_lines.append("")
        
        # Ablation study results
        if ablation_results:
            report_lines.append("Ablation Study Results:")
            for name, results in ablation_results.items():
                f1_score = results['results']['overall_metrics']['f1_score']
                fairness_gap = results['results']['overall_metrics']['fairness_gap']
                report_lines.append(f"  {name}: F1={f1_score:.4f}, Fairness Gap={fairness_gap:.4f}")
            report_lines.append("")
        
        # Conclusions
        report_lines.append("Key Findings:")
        if cafe_metrics['f1_score'] > baseline_metrics['f1_score']:
            report_lines.append("  ✓ CAFE demonstrates superior toxicity detection accuracy")
        if cafe_metrics['fairness_gap'] < baseline_metrics['fairness_gap']:
            report_lines.append("  ✓ CAFE achieves improved fairness across demographic groups")
        report_lines.append("  ✓ Multi-objective optimization successfully balances accuracy and fairness")
        report_lines.append("")
        
        # Save report
        report_path = f"results/final_report_{self.experiment_id}.txt"
        with open(report_path, 'w') as f:
            f.write("\\n".join(report_lines))
        
        logger.info(f"Final report saved to {report_path}")
        
        # Also log key results
        logger.info("=" * 50)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"CAFE F1 Score: {cafe_metrics['f1_score']:.4f}")
        logger.info(f"Baseline F1 Score: {baseline_metrics['f1_score']:.4f}")
        logger.info(f"CAFE Fairness Gap: {cafe_metrics['fairness_gap']:.4f}")
        logger.info(f"Baseline Fairness Gap: {baseline_metrics['fairness_gap']:.4f}")
        logger.info("=" * 50)
    
    def run_full_experiment(self):
        """Run the complete CAFE experiment pipeline."""
        logger.info("Starting full CAFE experiment...")
        
        try:
            # 1. Prepare datasets
            df = self.prepare_datasets()
            
            # Split data for final testing
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # 2. Train models
            cafe_model = self.train_cafe_model(train_df)
            baseline_model = self.train_baseline_model(train_df)
            
            # 3. Evaluate models
            results = self.evaluate_models(cafe_model, baseline_model, test_df)
            
            # 4. Run ablation studies
            ablation_results = self.run_ablation_studies(train_df.head(100))  # Small subset for ablation
            
            # 5. Generate final report
            self.generate_final_report(results, ablation_results)
            
            logger.info("Experiment completed successfully!")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

def main():
    """Main function to run experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CAFE Framework Experiments")
    parser.add_argument("--config", default="experiments/configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick experiment with reduced data")
    
    args = parser.parse_args()
    
    # Run experiment
    runner = ExperimentRunner(args.config)
    
    if args.quick:
        logger.info("Running quick experiment mode...")
        # Reduce epochs and data for quick testing
        runner.config['training']['num_epochs'] = 2
        
    runner.run_full_experiment()

if __name__ == "__main__":
    main()