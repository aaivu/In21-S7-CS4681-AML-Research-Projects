"""
Enhanced Evaluation Metrics for BraTS 2021
Comprehensive metrics including Dice, HD95, Sensitivity, and Specificity

Evaluates three tumor regions:
- Enhancing Tumor (ET): Label 4
- Tumor Core (TC): Labels 1, 3, 4
- Whole Tumor (WT): Labels 1, 2, 3, 4
"""

import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from scipy import ndimage
import pandas as pd
from typing import Dict, List, Tuple
import json


def read_nii(path):
    """Read NIfTI file and return array with spacing"""
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img), spacing


def dice_coefficient(pred, gt):
    """
    Compute Dice Similarity Coefficient
    
    Args:
        pred: Prediction binary mask
        gt: Ground truth binary mask
    Returns:
        Dice score (0-1)
    """
    if (pred.sum() + gt.sum()) == 0:
        return 1.0  # Both empty, perfect match
    else:
        return 2.0 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum())


def hausdorff_distance_95(pred, gt):
    """
    Compute 95th percentile Hausdorff Distance
    
    Args:
        pred: Prediction binary mask
        gt: Ground truth binary mask
    Returns:
        HD95 in mm (or 0 if one mask is empty)
    """
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            hd95 = binary.hd95(pred, gt)
            return hd95
        except Exception as e:
            print(f"HD95 computation failed: {e}")
            return 0.0
    else:
        return 0.0


def sensitivity(pred, gt):
    """
    Compute Sensitivity (Recall, True Positive Rate)
    
    Sensitivity = TP / (TP + FN)
    
    Args:
        pred: Prediction binary mask
        gt: Ground truth binary mask
    Returns:
        Sensitivity score (0-1)
    """
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    
    if (tp + fn) == 0:
        return 1.0  # No positive samples
    
    return tp / (tp + fn)


def specificity(pred, gt):
    """
    Compute Specificity (True Negative Rate)
    
    Specificity = TN / (TN + FP)
    
    Args:
        pred: Prediction binary mask
        gt: Ground truth binary mask
    Returns:
        Specificity score (0-1)
    """
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    
    if (tn + fp) == 0:
        return 1.0  # No negative samples
    
    return tn / (tn + fp)


def process_brats_labels(label):
    """
    Process BraTS labels into three tumor regions
    
    BraTS labels:
    - 0: Background
    - 1: Necrotic and non-enhancing tumor core (NCR/NET)
    - 2: Peritumoral edema (ED)
    - 4: GD-enhancing tumor (ET)
    
    Regions:
    - ET: Label 4
    - TC (Tumor Core): Labels 1, 3, 4
    - WT (Whole Tumor): Labels 1, 2, 4
    
    Args:
        label: 3D array with BraTS labels
    Returns:
        Tuple of (ET, TC, WT) binary masks
    """
    et = (label == 4).astype(np.uint8)
    tc = np.logical_or(label == 1, label == 4).astype(np.uint8)
    wt = np.logical_or(np.logical_or(label == 1, label == 2), label == 4).astype(np.uint8)
    
    return et, tc, wt


def compute_case_metrics(pred_path: str, gt_path: str) -> Dict[str, float]:
    """
    Compute all metrics for a single case
    
    Args:
        pred_path: Path to prediction NIfTI file
        gt_path: Path to ground truth NIfTI file
    Returns:
        Dictionary with all metrics
    """
    # Read files
    pred, pred_spacing = read_nii(pred_path)
    gt, gt_spacing = read_nii(gt_path)
    
    # Process labels into regions
    pred_et, pred_tc, pred_wt = process_brats_labels(pred)
    gt_et, gt_tc, gt_wt = process_brats_labels(gt)
    
    # Compute metrics for each region
    metrics = {}
    
    # Enhancing Tumor (ET)
    metrics['dice_et'] = dice_coefficient(pred_et, gt_et)
    metrics['hd95_et'] = hausdorff_distance_95(pred_et, gt_et)
    metrics['sensitivity_et'] = sensitivity(pred_et, gt_et)
    metrics['specificity_et'] = specificity(pred_et, gt_et)
    
    # Tumor Core (TC)
    metrics['dice_tc'] = dice_coefficient(pred_tc, gt_tc)
    metrics['hd95_tc'] = hausdorff_distance_95(pred_tc, gt_tc)
    metrics['sensitivity_tc'] = sensitivity(pred_tc, gt_tc)
    metrics['specificity_tc'] = specificity(pred_tc, gt_tc)
    
    # Whole Tumor (WT)
    metrics['dice_wt'] = dice_coefficient(pred_wt, gt_wt)
    metrics['hd95_wt'] = hausdorff_distance_95(pred_wt, gt_wt)
    metrics['sensitivity_wt'] = sensitivity(pred_wt, gt_wt)
    metrics['specificity_wt'] = specificity(pred_wt, gt_wt)
    
    return metrics


def evaluate_brats_folder(pred_folder: str, gt_folder: str, 
                          output_file: str = None) -> pd.DataFrame:
    """
    Evaluate all cases in a folder
    
    Args:
        pred_folder: Folder containing prediction NIfTI files
        gt_folder: Folder containing ground truth NIfTI files
        output_file: Optional path to save results CSV
    Returns:
        DataFrame with all metrics
    """
    # Get file lists
    pred_files = sorted(glob.glob(os.path.join(pred_folder, '*.nii.gz')))
    gt_files = sorted(glob.glob(os.path.join(gt_folder, '*.nii.gz')))
    
    if len(pred_files) == 0:
        raise ValueError(f"No prediction files found in {pred_folder}")
    if len(gt_files) == 0:
        raise ValueError(f"No ground truth files found in {gt_folder}")
    
    print(f"Found {len(pred_files)} prediction files")
    print(f"Found {len(gt_files)} ground truth files")
    
    # Match files by name
    results = []
    
    for pred_file in pred_files:
        pred_name = os.path.basename(pred_file)
        case_id = pred_name.replace('.nii.gz', '')
        
        # Find corresponding ground truth
        gt_file = None
        for gt in gt_files:
            if case_id in os.path.basename(gt):
                gt_file = gt
                break
        
        if gt_file is None:
            print(f"Warning: No ground truth found for {pred_name}")
            continue
        
        print(f"Evaluating {case_id}...")
        
        try:
            metrics = compute_case_metrics(pred_file, gt_file)
            metrics['case_id'] = case_id
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {case_id}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['case_id',
            'dice_et', 'dice_tc', 'dice_wt',
            'hd95_et', 'hd95_tc', 'hd95_wt',
            'sensitivity_et', 'sensitivity_tc', 'sensitivity_wt',
            'specificity_et', 'specificity_tc', 'specificity_wt']
    df = df[cols]
    
    # Compute summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for region in ['et', 'tc', 'wt']:
        print(f"\n{region.upper()} Metrics:")
        print(f"  Dice:        {df[f'dice_{region}'].mean():.4f} ± {df[f'dice_{region}'].std():.4f}")
        print(f"  HD95:        {df[f'hd95_{region}'].mean():.2f} ± {df[f'hd95_{region}'].std():.2f}")
        print(f"  Sensitivity: {df[f'sensitivity_{region}'].mean():.4f} ± {df[f'sensitivity_{region}'].std():.4f}")
        print(f"  Specificity: {df[f'specificity_{region}'].mean():.4f} ± {df[f'specificity_{region}'].std():.4f}")
    
    # Save results
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Save summary JSON
        summary_file = output_file.replace('.csv', '_summary.json')
        summary = {}
        for region in ['et', 'tc', 'wt']:
            summary[region] = {
                'dice_mean': float(df[f'dice_{region}'].mean()),
                'dice_std': float(df[f'dice_{region}'].std()),
                'hd95_mean': float(df[f'hd95_{region}'].mean()),
                'hd95_std': float(df[f'hd95_{region}'].std()),
                'sensitivity_mean': float(df[f'sensitivity_{region}'].mean()),
                'sensitivity_std': float(df[f'sensitivity_{region}'].std()),
                'specificity_mean': float(df[f'specificity_{region}'].mean()),
                'specificity_std': float(df[f'specificity_{region}'].std()),
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")
    
    return df


def compare_methods(results_dict: Dict[str, pd.DataFrame], 
                   output_file: str = None) -> pd.DataFrame:
    """
    Compare multiple methods
    
    Args:
        results_dict: Dictionary mapping method names to result DataFrames
        output_file: Optional path to save comparison table
    Returns:
        Comparison DataFrame
    """
    comparison = []
    
    for method_name, df in results_dict.items():
        row = {'Method': method_name}
        
        for region in ['et', 'tc', 'wt']:
            row[f'Dice_{region.upper()}'] = f"{df[f'dice_{region}'].mean():.3f}±{df[f'dice_{region}'].std():.3f}"
            row[f'HD95_{region.upper()}'] = f"{df[f'hd95_{region}'].mean():.1f}±{df[f'hd95_{region}'].std():.1f}"
        
        comparison.append(row)
    
    comp_df = pd.DataFrame(comparison)
    
    if output_file:
        comp_df.to_csv(output_file, index=False)
        print(f"Comparison table saved to {output_file}")
    
    return comp_df


def run_ablation_evaluation(base_path: str, gt_folder: str, 
                            output_dir: str = './ablation_results'):
    """
    Run evaluation for ablation study
    
    Args:
        base_path: Base path containing subdirectories for each configuration
        gt_folder: Path to ground truth folder
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    configs = ['baseline', 'cross_attn', 'fusion', 'training', 'full']
    results = {}
    
    for config in configs:
        pred_folder = os.path.join(base_path, config, 'predictions')
        
        if not os.path.exists(pred_folder):
            print(f"Warning: {pred_folder} does not exist, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Evaluating configuration: {config}")
        print(f"{'='*80}")
        
        output_file = os.path.join(output_dir, f'{config}_results.csv')
        df = evaluate_brats_folder(pred_folder, gt_folder, output_file)
        results[config] = df
    
    # Generate comparison table
    if len(results) > 0:
        comparison = compare_methods(results, os.path.join(output_dir, 'comparison.csv'))
        print("\n" + "="*80)
        print("ABLATION STUDY COMPARISON")
        print("="*80)
        print(comparison.to_string(index=False))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate BraTS predictions')
    parser.add_argument('--pred_folder', type=str, required=True,
                       help='Folder containing prediction files')
    parser.add_argument('--gt_folder', type=str, required=True,
                       help='Folder containing ground truth files')
    parser.add_argument('--output_file', type=str, default='results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study evaluation')
    parser.add_argument('--ablation_base', type=str, default=None,
                       help='Base path for ablation study')
    
    args = parser.parse_args()
    
    if args.ablation and args.ablation_base:
        run_ablation_evaluation(args.ablation_base, args.gt_folder)
    else:
        evaluate_brats_folder(args.pred_folder, args.gt_folder, args.output_file)
