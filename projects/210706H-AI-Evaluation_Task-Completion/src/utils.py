import os
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Any
import logging
import random

def setup_logging(log_file: str = "cafe.log") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_directories() -> None:
    """Create necessary directories for the project."""
    dirs = [
        "data/raw", "data/processed", "data/augmented",
        "results/models", "results/metrics", "results/plots",
        "experiments/configs/ablation_configs"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    @staticmethod
    def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
        from sklearn.metrics import f1_score
        y_pred_binary = (y_pred >= threshold).astype(int)
        return f1_score(y_true, y_pred_binary)
    
    @staticmethod
    def calculate_fairness_gap(y_pred: np.ndarray, sensitive_groups: np.ndarray) -> float:
        """Calculate fairness gap between groups."""
        group_0_mask = sensitive_groups == 0
        group_1_mask = sensitive_groups == 1
        
        if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
            return 0.0
            
        avg_score_0 = np.mean(y_pred[group_0_mask])
        avg_score_1 = np.mean(y_pred[group_1_mask])
        
        return abs(avg_score_0 - avg_score_1)
    
    @staticmethod
    def calculate_expected_maximum_toxicity(toxicity_scores: List[List[float]]) -> float:
        """Calculate Expected Maximum Toxicity (EMT)."""
        max_toxicities = [max(scores) for scores in toxicity_scores]
        return np.mean(max_toxicities)
    
    @staticmethod
    def calculate_toxicity_probability(toxicity_scores: List[List[float]], threshold: float = 0.5) -> float:
        """Calculate toxicity probability."""
        toxic_probs = []
        for scores in toxicity_scores:
            toxic_count = sum(1 for score in scores if score >= threshold)
            toxic_probs.append(toxic_count / len(scores))
        return np.mean(toxic_probs)