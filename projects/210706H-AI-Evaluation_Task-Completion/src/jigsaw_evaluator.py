import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class JigsawFairnessEvaluator:
    """
    Evaluate on Jigsaw for fairness metrics.
    No training - only evaluation of pre-trained CAFE vs Perspective.
    """
    
    def __init__(self, cafe_model, perspective_client, device):
        self.cafe_model = cafe_model
        self.perspective_client = perspective_client
        self.device = device
    
    def evaluate_fairness(self, jigsaw_df: pd.DataFrame) -> Dict:
        """
        Evaluate fairness metrics on Jigsaw Civil Comments.
        
        Metrics:
        - Overall F1, AUC
        - Subgroup AUC (for identity groups)
        - Fairness gap metrics
        """
        logger.info("Evaluating on Jigsaw for fairness metrics...")
        
        # Get comment texts and labels
        texts = jigsaw_df['comment_text'].tolist()
        true_labels = (jigsaw_df['toxicity'] >= 0.5).astype(int).values
        
        # For Jigsaw, treat full comment as both prompt and continuation
        # (since there's no clear separation)
        empty_prompts = [''] * len(texts)
        
        # Get CAFE predictions
        logger.info("Getting CAFE predictions...")
        cafe_scores = self.cafe_model.predict(empty_prompts, texts)
        cafe_binary = (cafe_scores >= 0.5).astype(int)
        
        # Get Perspective predictions
        logger.info("Getting Perspective predictions...")
        perspective_scores = []
        for text in texts:
            result = self.perspective_client.get_toxicity_score(text)
            perspective_scores.append(result.get('toxicity', 0.0))
        perspective_scores = np.array(perspective_scores)
        perspective_binary = (perspective_scores >= 0.5).astype(int)
        
        # Overall metrics
        cafe_f1 = f1_score(true_labels, cafe_binary)
        perspective_f1 = f1_score(true_labels, perspective_binary)
        
        try:
            cafe_auc = roc_auc_score(true_labels, cafe_scores)
            perspective_auc = roc_auc_score(true_labels, perspective_scores)
        except:
            cafe_auc = 0.0
            perspective_auc = 0.0
        
        # Subgroup AUC and fairness gaps
        identity_columns = [col for col in jigsaw_df.columns 
                          if any(identity in col.lower() 
                                for identity in ['male', 'female', 'black', 'white', 
                                               'christian', 'muslim', 'jewish', 'lgbtq'])]
        
        subgroup_metrics = {}
        
        for identity_col in identity_columns[:5]:  # Limit to first 5 for demo
            if identity_col in jigsaw_df.columns:
                # Get subgroup mask
                subgroup_mask = jigsaw_df[identity_col] >= 0.5
                
                if subgroup_mask.sum() > 10:  # Need sufficient samples
                    try:
                        cafe_subgroup_auc = roc_auc_score(
                            true_labels[subgroup_mask],
                            cafe_scores[subgroup_mask]
                        )
                        perspective_subgroup_auc = roc_auc_score(
                            true_labels[subgroup_mask],
                            perspective_scores[subgroup_mask]
                        )
                        
                        subgroup_metrics[identity_col] = {
                            'cafe_auc': cafe_subgroup_auc,
                            'perspective_auc': perspective_subgroup_auc,
                            'sample_count': subgroup_mask.sum()
                        }
                    except:
                        continue
        
        results = {
            'overall': {
                'cafe_f1': cafe_f1,
                'perspective_f1': perspective_f1,
                'cafe_auc': cafe_auc,
                'perspective_auc': perspective_auc
            },
            'subgroup_metrics': subgroup_metrics,
            'cafe_scores': cafe_scores.tolist(),
            'perspective_scores': perspective_scores.tolist()
        }
        
        logger.info(f"CAFE F1: {cafe_f1:.4f}, Perspective F1: {perspective_f1:.4f}")
        logger.info(f"CAFE AUC: {cafe_auc:.4f}, Perspective AUC: {perspective_auc:.4f}")
        
        return results