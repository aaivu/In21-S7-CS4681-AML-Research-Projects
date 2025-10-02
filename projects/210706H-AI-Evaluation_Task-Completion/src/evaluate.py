import torch
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os
import logging
from tqdm import tqdm

from model import CAFEModel, BaselineModel
from utils import MetricsCalculator, save_results
from train import ToxicityDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class CAFEEvaluator:
    """Comprehensive evaluation framework for CAFE model."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
        
    def evaluate_on_dataset(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_sensitive_groups = []
        all_contexts = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                if isinstance(outputs, dict):
                    predictions = outputs['toxicity_scores']
                else:
                    predictions = outputs
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['toxicity_score'].numpy())
                all_sensitive_groups.extend(batch['sensitive_group'].numpy())
                all_contexts.extend(batch.get('context_label', torch.zeros_like(batch['sensitive_group'])).numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        sensitive_groups = np.array(all_sensitive_groups)
        contexts = np.array(all_contexts)
        
        return self._calculate_comprehensive_metrics(predictions, targets, sensitive_groups, contexts)
    
    def _calculate_comprehensive_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                       sensitive_groups: np.ndarray, contexts: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Convert targets to binary for classification metrics
        targets_binary = (targets >= 0.5).astype(int)
        predictions_binary = (predictions >= 0.5).astype(int)
        
        # Basic metrics
        f1_score = MetricsCalculator.calculate_f1_score(targets_binary, predictions)
        fairness_gap = MetricsCalculator.calculate_fairness_gap(predictions, sensitive_groups)
        
        # Classification report
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        accuracy = accuracy_score(targets_binary, predictions_binary)
        precision = precision_score(targets_binary, predictions_binary, zero_division=0)
        recall = recall_score(targets_binary, predictions_binary, zero_division=0)
        
        try:
            auc_score = roc_auc_score(targets_binary, predictions)
        except ValueError:
            auc_score = 0.0
        
        # Fairness metrics by group
        group_0_mask = (sensitive_groups == 0)
        group_1_mask = (sensitive_groups == 1)
        
        fairness_metrics = {}
        if np.sum(group_0_mask) > 0 and np.sum(group_1_mask) > 0:
            # F1 scores by group
            f1_group_0 = MetricsCalculator.calculate_f1_score(
                targets_binary[group_0_mask], predictions[group_0_mask]
            ) if np.sum(group_0_mask) > 0 else 0.0
            
            f1_group_1 = MetricsCalculator.calculate_f1_score(
                targets_binary[group_1_mask], predictions[group_1_mask]
            ) if np.sum(group_1_mask) > 0 else 0.0
            
            fairness_metrics = {
                'f1_group_0': f1_group_0,
                'f1_group_1': f1_group_1,
                'f1_difference': abs(f1_group_0 - f1_group_1),
                'mean_prediction_group_0': np.mean(predictions[group_0_mask]) if np.sum(group_0_mask) > 0 else 0.0,
                'mean_prediction_group_1': np.mean(predictions[group_1_mask]) if np.sum(group_1_mask) > 0 else 0.0
            }
        
        # Context-aware metrics
        context_metrics = {}
        literal_mask = (contexts == 0)
        non_literal_mask = (contexts == 1)
        
        if np.sum(literal_mask) > 0 and np.sum(non_literal_mask) > 0:
            f1_literal = MetricsCalculator.calculate_f1_score(
                targets_binary[literal_mask], predictions[literal_mask]
            ) if np.sum(literal_mask) > 0 else 0.0
            
            f1_non_literal = MetricsCalculator.calculate_f1_score(
                targets_binary[non_literal_mask], predictions[non_literal_mask]
            ) if np.sum(non_literal_mask) > 0 else 0.0
            
            context_metrics = {
                'f1_literal': f1_literal,
                'f1_non_literal': f1_non_literal,
                'context_f1_difference': abs(f1_literal - f1_non_literal)
            }
        
        return {
            'overall_metrics': {
                'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'auc_score': auc_score,
                'fairness_gap': fairness_gap
            },
            'fairness_metrics': fairness_metrics,
            'context_metrics': context_metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }
    
    def generate_evaluation_report(self, results: Dict[str, Any], save_dir: str = "results/metrics"):
        """Generate comprehensive evaluation report."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw results
        save_results(results, f"{save_dir}/evaluation_results.json")
        
        # Create visualizations
        self._create_evaluation_plots(results, save_dir)
        
        # Generate text report
        self._generate_text_report(results, save_dir)
        
        logger.info(f"Evaluation report saved to {save_dir}")
    
    def _create_evaluation_plots(self, results: Dict[str, Any], save_dir: str):
        """Create evaluation visualizations."""
        
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        
        # 1. Prediction distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(predictions, bins=30, alpha=0.7, label='Predictions', color='blue')
        plt.hist(targets, bins=30, alpha=0.7, label='Targets', color='red')
        plt.xlabel('Toxicity Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()
        
        # 2. Scatter plot
        plt.subplot(1, 3, 2)
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlabel('True Toxicity Score')
        plt.ylabel('Predicted Toxicity Score')
        plt.title('Prediction vs Truth')
        
        # 3. Confusion matrix
        plt.subplot(1, 3, 3)
        targets_binary = (targets >= 0.5).astype(int)
        predictions_binary = (predictions >= 0.5).astype(int)
        cm = confusion_matrix(targets_binary, predictions_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/evaluation_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Metrics comparison plot
        self._plot_metrics_comparison(results, save_dir)
    
    def _plot_metrics_comparison(self, results: Dict[str, Any], save_dir: str):
        """Create metrics comparison visualization."""
        
        metrics = results['overall_metrics']
        
        plt.figure(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = plt.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        plt.ylabel('Score')
        plt.title('Overall Performance Metrics')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results: Dict[str, Any], save_dir: str):
        """Generate text-based evaluation report."""
        
        report_lines = []
        report_lines.append("CAFE Model Evaluation Report")
        report_lines.append("=" * 40)
        report_lines.append("")
        
        # Overall metrics
        report_lines.append("Overall Performance:")
        for metric, value in results['overall_metrics'].items():
            report_lines.append(f"  {metric}: {value:.4f}")
        report_lines.append("")
        
        # Fairness metrics
        if results['fairness_metrics']:
            report_lines.append("Fairness Analysis:")
            for metric, value in results['fairness_metrics'].items():
                report_lines.append(f"  {metric}: {value:.4f}")
            report_lines.append("")
        
        # Context metrics
        if results['context_metrics']:
            report_lines.append("Context-Aware Analysis:")
            for metric, value in results['context_metrics'].items():
                report_lines.append(f"  {metric}: {value:.4f}")
            report_lines.append("")
        
        # Save report
        with open(f"{save_dir}/evaluation_report.txt", 'w') as f:
            f.write("\\n".join(report_lines))

def compare_models(cafe_results: Dict[str, Any], baseline_results: Dict[str, Any], 
                  save_dir: str = "results/metrics") -> Dict[str, Any]:
    """Compare CAFE model with baseline."""
    
    comparison = {}
    
    # Compare overall metrics
    overall_comparison = {}
    for metric in cafe_results['overall_metrics'].keys():
        cafe_value = cafe_results['overall_metrics'][metric]
        baseline_value = baseline_results['overall_metrics'][metric]
        improvement = ((cafe_value - baseline_value) / baseline_value * 100) if baseline_value != 0 else 0
        
        overall_comparison[metric] = {
            'cafe': cafe_value,
            'baseline': baseline_value,
            'improvement_percent': improvement
        }
    
    comparison['overall_comparison'] = overall_comparison
    
    # Fairness comparison
    if cafe_results.get('fairness_metrics') and baseline_results.get('fairness_metrics'):
        fairness_comparison = {}
        for metric in cafe_results['fairness_metrics'].keys():
            if metric in baseline_results['fairness_metrics']:
                cafe_value = cafe_results['fairness_metrics'][metric]
                baseline_value = baseline_results['fairness_metrics'][metric]
                improvement = ((baseline_value - cafe_value) / baseline_value * 100) if baseline_value != 0 else 0  # Lower is better for fairness gap
                
                fairness_comparison[metric] = {
                    'cafe': cafe_value,
                    'baseline': baseline_value,
                    'improvement_percent': improvement
                }
        
        comparison['fairness_comparison'] = fairness_comparison
    
    # Save comparison results
    save_results(comparison, f"{save_dir}/model_comparison.json")
    
    # Create comparison visualization
    _plot_model_comparison(comparison, save_dir)
    
    return comparison

def _plot_model_comparison(comparison: Dict[str, Any], save_dir: str):
    """Create model comparison visualization."""
    
    overall_comp = comparison['overall_comparison']
    
    metrics = list(overall_comp.keys())
    cafe_values = [overall_comp[m]['cafe'] for m in metrics]
    baseline_values = [overall_comp[m]['baseline'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, cafe_values, width, label='CAFE', color='#2ca02c')
    plt.bar(x + width/2, baseline_values, width, label='Baseline', color='#d62728')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('CAFE vs Baseline Model Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    
    # Add improvement percentages as text
    for i, metric in enumerate(metrics):
        improvement = overall_comp[metric]['improvement_percent']
        plt.text(i, max(cafe_values[i], baseline_values[i]) + 0.02, 
                f'{improvement:+.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function."""
    from utils import setup_logging, load_config
    
    setup_logging()
    
    # Load test data
    if os.path.exists("data/processed/test_data.csv"):
        test_df = pd.read_csv("data/processed/test_data.csv")
    else:
        # Use subset of augmented data for testing
        df = pd.read_csv("data/augmented/augmented_rtp.csv")
        test_df = df.tail(200)  # Use last 200 samples for testing
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate CAFE model
    logger.info("Evaluating CAFE model...")
    cafe_model = CAFEModel().to(device)
    cafe_evaluator = CAFEEvaluator(cafe_model, device)
    
    if os.path.exists("results/models/best_model.pt"):
        cafe_evaluator.load_model("results/models/best_model.pt")
    else:
        logger.warning("No trained CAFE model found. Using untrained model.")
    
    # Prepare test dataloader
    test_dataset = ToxicityDataset(test_df, cafe_model.tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluate CAFE
    cafe_results = cafe_evaluator.evaluate_on_dataset(test_dataloader)
    cafe_evaluator.generate_evaluation_report(cafe_results, "results/metrics/cafe")
    
    # Evaluate baseline model
    logger.info("Evaluating baseline model...")
    baseline_model = BaselineModel().to(device)
    baseline_evaluator = CAFEEvaluator(baseline_model, device)
    
    if os.path.exists("results/models/baseline_model.pt"):
        baseline_evaluator.load_model("results/models/baseline_model.pt")
    else:
        logger.warning("No trained baseline model found. Using untrained model.")
    
    baseline_results = baseline_evaluator.evaluate_on_dataset(test_dataloader)
    baseline_evaluator.generate_evaluation_report(baseline_results, "results/metrics/baseline")
    
    # Compare models
    logger.info("Comparing models...")
    comparison = compare_models(cafe_results, baseline_results)
    
    # Print summary
    logger.info("Evaluation Summary:")
    logger.info(f"CAFE F1 Score: {cafe_results['overall_metrics']['f1_score']:.4f}")
    logger.info(f"Baseline F1 Score: {baseline_results['overall_metrics']['f1_score']:.4f}")
    logger.info(f"CAFE Fairness Gap: {cafe_results['overall_metrics']['fairness_gap']:.4f}")
    logger.info(f"Baseline Fairness Gap: {baseline_results['overall_metrics']['fairness_gap']:.4f}")

if __name__ == "__main__":
    main()