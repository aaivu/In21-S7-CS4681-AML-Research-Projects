import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], 
                         labels: List[str] = None, 
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: str = None) -> None:
    """
    Plot confusion matrix with proper formatting
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         title: str = 'Training History',
                         figsize: Tuple[int, int] = (15, 5),
                         save_path: str = None) -> None:
    """
    Plot training history with loss and metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss
    axes[0].plot(history['loss'], label='Training Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # AUC
    if 'auc' in history and 'val_auc' in history:
        axes[2].plot(history['auc'], label='Training AUC')
        axes[2].plot(history['val_auc'], label='Validation AUC')
        axes[2].set_title('Model AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_metrics_per_class(y_true: List[str], y_pred: List[str], 
                               labels: List[str] = None) -> pd.DataFrame:
    """
    Calculate detailed metrics per class
    """
    if labels is None:
        labels = sorted(list(set(y_true + y_pred)))
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = []
    for label in labels:
        # Binary classification metrics for each class
        y_true_binary = [1 if y == label else 0 for y in y_true]
        y_pred_binary = [1 if y == label else 0 for y in y_pred]
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Support (number of true instances)
        support = sum(y_true_binary)
        
        metrics.append({
            'Class': label,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
    
    return pd.DataFrame(metrics)


def create_ensemble_report(results: Dict[str, Any], 
                          method_name: str = 'Ensemble') -> Dict[str, Any]:
    """
    Create comprehensive evaluation report
    """
    report = {
        'method': method_name,
        'overall_metrics': {
            'accuracy': results.get('accuracy', 0),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'f1_score': results.get('f1_score', 0)
        },
        'confusion_matrix': results.get('confusion_matrix'),
        'classification_report': results.get('classification_report'),
        'predictions': results.get('predictions'),
        'confidences': results.get('confidences')
    }
    
    # Calculate per-class metrics if predictions available
    if 'predictions' in results and results['predictions']:
        y_true = results.get('y_true', [])
        y_pred = results['predictions']
        
        if y_true:
            report['per_class_metrics'] = calculate_metrics_per_class(y_true, y_pred)
    
    return report


def compare_ensemble_methods(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple ensemble methods
    """
    comparison = []
    
    for method, result in results.items():
        if result is not None:
            comparison.append({
                'Method': method.title(),
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1-Score': result.get('f1_score', 0)
            })
    
    df = pd.DataFrame(comparison)
    if not df.empty:
        # Round numerical columns
        numerical_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        df[numerical_cols] = df[numerical_cols].round(4)
    
    return df


def plot_method_comparison(comparison_df: pd.DataFrame, 
                          title: str = 'Ensemble Methods Comparison',
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: str = None) -> None:
    """
    Plot comparison of different ensemble methods
    """
    if comparison_df.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i*width, comparison_df[metric], width, label=metric)
    
    axes[0].set_xlabel('Methods')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Metrics by Method')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(comparison_df['Method'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Radar plot (if multiple methods)
    if len(comparison_df) > 1:
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for _, row in comparison_df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            axes[1].plot(angles, values, 'o-', linewidth=2, label=row['Method'])
            axes[1].fill(angles, values, alpha=0.25)
        
        axes[1].set_xticks(angles[:-1])
        axes[1].set_xticklabels(metrics)
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Performance Radar Chart')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_results_to_csv(results: Dict[str, Any], filepath: str) -> None:
    """
    Save evaluation results to CSV
    """
    # Create summary data
    summary_data = []
    
    for method, result in results.items():
        if result is not None:
            summary_data.append({
                'Method': method,
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1_Score': result.get('f1_score', 0)
            })
    
    # Save to CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
    else:
        print("No valid results to save")


def generate_research_summary(results: Dict[str, Any], 
                            model_info: Dict[str, Any]) -> str:
    """
    Generate a research-ready summary of results
    """
    summary = """
NETWORK INTRUSION DETECTION ENSEMBLE - RESEARCH SUMMARY
=========================================================

Model Architecture:
- Specialized binary classifiers for each attack category (DoS, Probe, R2L, U2R)
- CNN-LSTM architecture with focal loss for handling class imbalance
- SMOTE oversampling for training data augmentation
- RFE feature selection (13 features per attack type)

Loaded Models:
"""
    
    summary += f"- Deep Learning Models: {len(model_info.get('deep_learning_models', []))}\n"
    summary += f"- Traditional ML Models: {len(model_info.get('traditional_ml_models', []))}\n"
    summary += f"- Total Models: {model_info.get('total_models', 0)}\n\n"
    
    summary += "Ensemble Methods Evaluated:\n"
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    for method, result in valid_results.items():
        summary += f"\n{method.upper()} METHOD:\n"
        summary += f"- Accuracy: {result.get('accuracy', 0):.4f}\n"
        summary += f"- Precision: {result.get('precision', 0):.4f}\n"
        summary += f"- Recall: {result.get('recall', 0):.4f}\n"
        summary += f"- F1-Score: {result.get('f1_score', 0):.4f}\n"
    
    if valid_results:
        best_method = max(valid_results.keys(), key=lambda x: valid_results[x].get('f1_score', 0))
        summary += f"\nBEST PERFORMING METHOD: {best_method.upper()}\n"
        summary += f"F1-Score: {valid_results[best_method].get('f1_score', 0):.4f}\n"
    
    summary += "\n" + "="*60 + "\n"
    
    return summary
