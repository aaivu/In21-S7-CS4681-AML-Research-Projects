"""
Evaluation Metrics for Jigsaw Toxicity Classification
Implements bias-aware metrics: Overall AUC, Subgroup AUC, BPSN AUC, BNSP AUC
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings


def calculate_overall_auc(y_true, y_pred):
    """
    Calculate overall AUC-ROC score.
    
    Args:
        y_true (array-like): Ground truth binary labels
        y_pred (array-like): Predicted probabilities
        
    Returns:
        float: AUC-ROC score
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        warnings.warn(f"Could not calculate overall AUC: {e}")
        return np.nan


def calculate_subgroup_auc(y_true, y_pred, identity_series):
    """
    Calculate AUC within a specific identity subgroup.
    Only considers samples where the identity is mentioned (identity_series > 0).
    
    Args:
        y_true (pd.Series): Ground truth binary labels
        y_pred (pd.Series): Predicted probabilities
        identity_series (pd.Series): Binary indicator if identity is mentioned
        
    Returns:
        float: Subgroup AUC score
    """
    # Filter to subgroup samples (where identity is mentioned)
    subgroup_mask = identity_series > 0
    
    if subgroup_mask.sum() < 2:
        warnings.warn("Not enough subgroup samples to calculate AUC")
        return np.nan
    
    y_true_subgroup = y_true[subgroup_mask]
    y_pred_subgroup = y_pred[subgroup_mask]
    
    # Check if we have both classes
    if len(np.unique(y_true_subgroup)) < 2:
        warnings.warn("Subgroup contains only one class")
        return np.nan
    
    try:
        return roc_auc_score(y_true_subgroup, y_pred_subgroup)
    except ValueError as e:
        warnings.warn(f"Could not calculate subgroup AUC: {e}")
        return np.nan


def calculate_bpsn_auc(y_true, y_pred, identity_series):
    """
    Calculate Background Positive, Subgroup Negative (BPSN) AUC.
    
    BPSN measures how well the model distinguishes between:
    - Positive class (toxic): samples from the general background (no identity)
    - Negative class (non-toxic): samples from the subgroup (identity mentioned)
    
    This metric reveals if the model incorrectly associates the identity
    with toxicity (lower scores indicate bias).
    
    Args:
        y_true (pd.Series): Ground truth binary labels (1 = toxic, 0 = non-toxic)
        y_pred (pd.Series): Predicted probabilities
        identity_series (pd.Series): Binary indicator if identity is mentioned
        
    Returns:
        float: BPSN AUC score
    """
    # Background positive: toxic samples without identity mention
    background_positive_mask = (y_true >= 0.5) & (identity_series <= 0)
    
    # Subgroup negative: non-toxic samples with identity mention
    subgroup_negative_mask = (y_true < 0.5) & (identity_series > 0)
    
    # Combine masks
    bpsn_mask = background_positive_mask | subgroup_negative_mask
    
    if bpsn_mask.sum() < 2:
        warnings.warn("Not enough BPSN samples to calculate AUC")
        return np.nan
    
    y_true_bpsn = y_true[bpsn_mask]
    y_pred_bpsn = y_pred[bpsn_mask]
    
    # Check if we have both classes
    if len(np.unique(y_true_bpsn >= 0.5)) < 2:
        warnings.warn("BPSN set contains only one class")
        return np.nan
    
    try:
        return roc_auc_score(y_true_bpsn >= 0.5, y_pred_bpsn)
    except ValueError as e:
        warnings.warn(f"Could not calculate BPSN AUC: {e}")
        return np.nan


def calculate_bnsp_auc(y_true, y_pred, identity_series):
    """
    Calculate Background Negative, Subgroup Positive (BNSP) AUC.
    
    BNSP measures how well the model distinguishes between:
    - Negative class (non-toxic): samples from the general background (no identity)
    - Positive class (toxic): samples from the subgroup (identity mentioned)
    
    This metric reveals if the model fails to detect toxicity when an identity
    is mentioned (lower scores indicate bias).
    
    Args:
        y_true (pd.Series): Ground truth binary labels (1 = toxic, 0 = non-toxic)
        y_pred (pd.Series): Predicted probabilities
        identity_series (pd.Series): Binary indicator if identity is mentioned
        
    Returns:
        float: BNSP AUC score
    """
    # Background negative: non-toxic samples without identity mention
    background_negative_mask = (y_true < 0.5) & (identity_series <= 0)
    
    # Subgroup positive: toxic samples with identity mention
    subgroup_positive_mask = (y_true >= 0.5) & (identity_series > 0)
    
    # Combine masks
    bnsp_mask = background_negative_mask | subgroup_positive_mask
    
    if bnsp_mask.sum() < 2:
        warnings.warn("Not enough BNSP samples to calculate AUC")
        return np.nan
    
    y_true_bnsp = y_true[bnsp_mask]
    y_pred_bnsp = y_pred[bnsp_mask]
    
    # Check if we have both classes
    if len(np.unique(y_true_bnsp >= 0.5)) < 2:
        warnings.warn("BNSP set contains only one class")
        return np.nan
    
    try:
        return roc_auc_score(y_true_bnsp >= 0.5, y_pred_bnsp)
    except ValueError as e:
        warnings.warn(f"Could not calculate BNSP AUC: {e}")
        return np.nan


def calculate_all_metrics(df, pred_col='prediction', target_col='target', identity_columns=None):
    """
    Calculate all bias metrics for all identity groups.
    
    Args:
        df (pd.DataFrame): DataFrame with predictions and labels
        pred_col (str): Name of prediction column
        target_col (str): Name of target column
        identity_columns (list): List of identity column names
        
    Returns:
        pd.DataFrame: Results table with metrics for each identity
    """
    if identity_columns is None:
        # Default Jigsaw identity columns
        identity_columns = [
            'male', 'female', 'transgender', 'other_gender',
            'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
            'other_sexual_orientation', 'christian', 'jewish', 'muslim',
            'hindu', 'buddhist', 'atheist', 'other_religion',
            'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
            'physical_disability', 'intellectual_or_learning_disability',
            'psychiatric_or_mental_illness', 'other_disability'
        ]
    
    # Filter to available columns
    available_identity_cols = [col for col in identity_columns if col in df.columns]
    
    results = []
    
    # Overall metric
    overall_auc = calculate_overall_auc(df[target_col], df[pred_col])
    results.append({
        'identity': 'overall',
        'subgroup_size': len(df),
        'overall_auc': overall_auc,
        'subgroup_auc': np.nan,
        'bpsn_auc': np.nan,
        'bnsp_auc': np.nan
    })
    
    # Calculate for each identity group
    for identity in available_identity_cols:
        subgroup_size = (df[identity] > 0).sum()
        
        if subgroup_size < 10:  # Skip very small subgroups
            continue
        
        metrics = {
            'identity': identity,
            'subgroup_size': subgroup_size,
            'overall_auc': overall_auc,
            'subgroup_auc': calculate_subgroup_auc(df[target_col], df[pred_col], df[identity]),
            'bpsn_auc': calculate_bpsn_auc(df[target_col], df[pred_col], df[identity]),
            'bnsp_auc': calculate_bnsp_auc(df[target_col], df[pred_col], df[identity])
        }
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    return results_df