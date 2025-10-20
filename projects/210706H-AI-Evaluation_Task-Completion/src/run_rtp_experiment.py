"""
CAFE vs Perspective API Experiment on RealToxicityPrompts Dataset
Uses preprocessed RTP data with existing Perspective scores
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from RoBERTaForToxicity import RoBERTaForToxicity
from JigsawDataset import JigsawDataset
from evaluate import EnsembleModel


def merge_rtp_parts(data_dir='.', pattern='rtp_part_*_with_perspectiveScore.csv', 
                    num_parts=30, save_merged=True):
    """
    Merge all RTP CSV parts into a single DataFrame.
    
    Args:
        data_dir (str): Directory containing the CSV files
        pattern (str): Glob pattern to match CSV files
        num_parts (int): Expected number of parts
        save_merged (bool): Save merged dataset to disk
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    print("=" * 80)
    print("Merging RTP Dataset Parts")
    print("=" * 80)
    
    # Find all matching CSV files
    search_pattern = os.path.join(data_dir, pattern)
    csv_files = sorted(glob.glob(search_pattern))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern: {search_pattern}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files[:5]:  # Show first 5
        print(f"  {os.path.basename(f)}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more")
    
    if len(csv_files) != num_parts:
        print(f"⚠ Warning: Expected {num_parts} parts, found {len(csv_files)}")
    
    # Load and merge all parts
    print("\nLoading CSV files...")
    dfs = []
    for csv_file in tqdm(csv_files):
        df_part = pd.read_csv(csv_file)
        dfs.append(df_part)
    
    # Concatenate all parts
    print("Merging all parts...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print(f"✓ Merged dataset: {len(merged_df)} total samples")
    print(f"  Columns: {list(merged_df.columns)}")
    
    # Check for required columns
    required_cols = ['full_text']
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Rename perspective score column if needed
    perspective_cols = [col for col in merged_df.columns if 'perspective' in col.lower()]
    if perspective_cols:
        main_perspective_col = perspective_cols[0]
        if main_perspective_col != 'perspective_score':
            merged_df.rename(columns={main_perspective_col: 'perspective_score'}, inplace=True)
            print(f"  Renamed '{main_perspective_col}' → 'perspective_score'")
    
    # Remove duplicates if any
    original_len = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['full_text']).reset_index(drop=True)
    if len(merged_df) < original_len:
        print(f"  Removed {original_len - len(merged_df)} duplicate entries")
    
    # Save merged dataset
    if save_merged:
        output_file = 'rtp_merged_full.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved merged dataset to {output_file}")
    
    return merged_df


def create_rtp_proxy_labels(df, text_column='full_text'):
    """
    Create proxy identity labels for RealToxicityPrompts data.
    Uses keyword matching to identify mentions of identity groups.
    
    Args:
        df (pd.DataFrame): DataFrame with text data
        text_column (str): Name of column containing text
        
    Returns:
        pd.DataFrame: DataFrame with added identity columns
    """
    print("\nCreating proxy identity labels...")
    
    # Define identity keywords (aligned with Jigsaw categories)
    identity_keywords = {
        'black': ['black', 'african', 'african american', 'african-american'],
        'white': ['white', 'caucasian'],
        'asian': ['asian', 'chinese', 'japanese', 'korean', 'indian', 'south asian', 'east asian'],
        'latino': ['latino', 'latina', 'hispanic', 'mexican', 'latinx'],
        'jewish': ['jew', 'jewish', 'judaism'],
        'muslim': ['muslim', 'islam', 'islamic'],
        'christian': ['christian', 'christianity', 'catholic', 'protestant'],
        'male': ['man', 'men', 'male', 'he', 'him', 'his', 'guy', 'boy'],
        'female': ['woman', 'women', 'female', 'she', 'her', 'girl', 'lady'],
        'homosexual_gay_or_lesbian': ['gay', 'lesbian', 'homosexual'],
        'transgender': ['trans', 'transgender', 'transsexual'],
        'bisexual': ['bisexual', 'bi'],
        'heterosexual': ['straight', 'heterosexual'],
        'physical_disability': ['disabled', 'disability', 'wheelchair', 'blind', 'deaf'],
        'psychiatric_or_mental_illness': ['mental', 'mentally ill', 'psychiatric', 'depression', 'anxiety', 'schizophrenia']
    }
    
    # Initialize identity columns
    for identity in identity_keywords.keys():
        df[identity] = 0.0
    
    # Search for keywords (case-insensitive)
    print("Searching for identity keywords...")
    for identity, keywords in tqdm(identity_keywords.items(), desc="Processing identities"):
        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + keyword + r'\b'
            matches = df[text_column].str.lower().str.contains(pattern, regex=True, na=False)
            df.loc[matches, identity] = 1.0
    
    # Count identity mentions
    identity_cols = list(identity_keywords.keys())
    identity_mentions = df[identity_cols].sum()
    
    print(f"\n✓ Identity mentions found:")
    for identity, count in identity_mentions[identity_mentions > 0].sort_values(ascending=False).items():
        print(f"  {identity}: {int(count)}")
    
    return df


def run_rtp_experiment(
    data_dir='.',
    pattern='rtp_part_*_with_perspectiveScore.csv',
    num_parts=30,
    sample_size=5000,
    model_name='roberta-base',
    n_folds=5,
    batch_size=16,
    max_length=256,
    power=3.5,
    merge_first=True
):
    """
    Run the CAFE vs Perspective API comparison experiment.
    
    Args:
        data_dir (str): Directory containing RTP CSV files
        pattern (str): Pattern for CSV file names
        num_parts (int): Expected number of CSV parts
        sample_size (int): Number of samples to process (None = all)
        model_name (str): HuggingFace model name
        n_folds (int): Number of folds in ensemble
        batch_size (int): Batch size for inference
        max_length (int): Maximum sequence length
        power (float): Power for ensemble averaging
        merge_first (bool): Whether to merge parts first or use existing merged file
    """
    print("=" * 80)
    print("CAFE vs Perspective API Experiment on RealToxicityPrompts")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size if sample_size else 'Full dataset'}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading existing merged dataset: 'rtp_merged_full.csv'")
    rtp_df = pd.read_csv('rtp_merged_full.csv')
    print(f"✓ Loaded {len(rtp_df)} samples")
    
    # Sample if requested
    if sample_size and sample_size < len(rtp_df):
        print(f"\nSampling {sample_size} from {len(rtp_df)} total samples...")
        rtp_df = rtp_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"✓ Sampled {len(rtp_df)} samples")
    
    # Check for Perspective scores
    if 'perspective_score' not in rtp_df.columns:
        print("\n⚠ Warning: 'perspective_score' column not found")
        print(f"Available columns: {list(rtp_df.columns)}")
        perspective_col = None
        for col in rtp_df.columns:
            if 'perspective' in col.lower() or 'toxicity' in col.lower():
                perspective_col = col
                print(f"Using column '{col}' as perspective score")
                rtp_df['perspective_score'] = rtp_df[col]
                break
        if perspective_col is None:
            print("⚠ No perspective score found - will only generate CAFE scores")
    else:
        print(f"\n✓ Perspective scores found: mean={rtp_df['perspective_score'].mean():.4f}")
    
    # Create proxy identity labels
    rtp_df = create_rtp_proxy_labels(rtp_df, text_column='full_text')
    
    # Load CAFE ensemble model
    print("\n" + "=" * 80)
    print("Loading CAFE Ensemble Model")
    print("=" * 80)
    
    # Auto-detect available model checkpoints
    roberta_pattern = 'models/roberta_fold*_best.pt'
    deberta_pattern = 'models/deberta_fold*_best.pt'
    
    roberta_models = sorted(glob.glob(roberta_pattern))
    deberta_models = sorted(glob.glob(deberta_pattern))
    
    if roberta_models:
        model_paths = roberta_models
        detected_model = 'roberta-base'
        print(f"✓ Found {len(roberta_models)} RoBERTa model checkpoint(s)")
    elif deberta_models:
        model_paths = deberta_models
        detected_model = 'microsoft/deberta-v3-large'
        print(f"✓ Found {len(deberta_models)} DeBERTa model checkpoint(s)")
    else:
        raise FileNotFoundError(
            "\n❌ No model checkpoints found!\n\n"
            "Please train models first:\n"
            "  1. Run: python train_roberta.py\n"
            "  2. Wait for training to complete (creates models/roberta_fold*_best.pt)\n"
            "  3. Then run this script again\n\n"
            f"Expected files in 'models/' directory:\n"
            f"  - {roberta_pattern}\n"
            f"  - OR {deberta_pattern}"
        )
    
    # Use detected model name if not explicitly specified
    if model_name == 'roberta-base' and detected_model != 'roberta-base':
        model_name = detected_model
        print(f"  Using detected model: {model_name}")
    
    for i, path in enumerate(model_paths):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    ensemble = EnsembleModel(model_paths, model_name=model_name, device=device)
    
    # Get CAFE predictions
    print("\n" + "=" * 80)
    print("Generating CAFE Predictions")
    print("=" * 80)
    
    # Create temporary dataset for CAFE inference
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Create minimal dataframe with required columns
    temp_df = pd.DataFrame({
        'comment_text_cleaned': rtp_df['full_text'],
        'target': [0.0] * len(rtp_df),  # Dummy labels
        'severe_toxicity': [0.0] * len(rtp_df),
        'obscene': [0.0] * len(rtp_df),
        'identity_attack': [0.0] * len(rtp_df),
        'insult': [0.0] * len(rtp_df),
        'threat': [0.0] * len(rtp_df),
        'sexual_explicit': [0.0] * len(rtp_df)
    })
    
    temp_dataset = JigsawDataset(temp_df, tokenizer, max_length=max_length)
    temp_loader = DataLoader(
        temp_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=JigsawDataset.collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    cafe_scores = ensemble.predict(temp_loader, power=power, use_mixed_precision=True)
    rtp_df['cafe_score'] = cafe_scores
    
    print(f"\n✓ CAFE predictions complete")
    print(f"  Mean: {cafe_scores.mean():.4f}")
    print(f"  Std:  {cafe_scores.std():.4f}")
    print(f"  Min:  {cafe_scores.min():.4f}")
    print(f"  Max:  {cafe_scores.max():.4f}")
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    identity_cols = [col for col in rtp_df.columns 
                     if col not in ['full_text', 'cafe_score', 'perspective_score']]
    
    # Prepare output with renamed columns for compatibility
    output_cols = ['full_text']
    output_df = rtp_df[['full_text']].copy()
    
    # Rename CAFE score to match expected format
    output_df['CAFE_score'] = rtp_df['cafe_score']
    
    # Add Perspective score if available
    if 'perspective_score' in rtp_df.columns:
        output_df['Perspective_score'] = rtp_df['perspective_score']
    
    # Add identity columns
    for col in identity_cols:
        output_df[col] = rtp_df[col]
    
    # Save main output file (for generate_results.py)
    main_output = 'rtp_scores.csv'
    output_df.to_csv(main_output, index=False)
    print(f"✓ Main results saved to {main_output}")
    
    # Also save a copy with sample size in filename for reference
    backup_output = f'rtp_cafe_vs_perspective_{len(output_df)}samples.csv'
    output_df.to_csv(backup_output, index=False)
    print(f"✓ Backup saved to {backup_output}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    print(f"\nCAFE scores:")
    print(f"  Mean: {rtp_df['cafe_score'].mean():.4f}")
    print(f"  Std:  {rtp_df['cafe_score'].std():.4f}")
    print(f"  Min:  {rtp_df['cafe_score'].min():.4f}")
    print(f"  Max:  {rtp_df['cafe_score'].max():.4f}")
    
    if 'perspective_score' in rtp_df.columns:
        print(f"\nPerspective scores:")
        print(f"  Mean: {rtp_df['perspective_score'].mean():.4f}")
        print(f"  Std:  {rtp_df['perspective_score'].std():.4f}")
        print(f"  Min:  {rtp_df['perspective_score'].min():.4f}")
        print(f"  Max:  {rtp_df['perspective_score'].max():.4f}")
        
        # Calculate correlation
        correlation = rtp_df[['cafe_score', 'perspective_score']].corr().iloc[0, 1]
        print(f"\nCorrelation (CAFE vs Perspective): {correlation:.4f}")
        
        # Calculate agreement metrics
        threshold = 0.5
        cafe_toxic = (rtp_df['cafe_score'] >= threshold).astype(int)
        perspective_toxic = (rtp_df['perspective_score'] >= threshold).astype(int)
        agreement = (cafe_toxic == perspective_toxic).mean()
        print(f"\nAgreement at threshold {threshold}: {agreement:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Experiment Complete!")
    print("=" * 80)
    print("\nOutput file: 'rtp_merged_full.csv'")
    print("Next steps:")
    print("  1. Run generate_results.py to create figures and tables")
    print("  2. Analyze bias metrics across identity groups")
    
    return rtp_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CAFE vs Perspective experiment on RTP")
    parser.add_argument("--data_dir", type=str, default=".", 
                        help="Directory containing RTP CSV files")
    parser.add_argument("--pattern", type=str, default="rtp_part_*_with_perspectiveScore.csv",
                        help="Pattern for RTP CSV files")
    parser.add_argument("--num_parts", type=int, default=30,
                        help="Expected number of CSV parts")
    parser.add_argument("--sample_size", type=int, default=5000,
                        help="Number of samples to process (None = all)")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Model name")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds in ensemble")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--no_merge", action="store_true",
                        help="Use existing merged file instead of re-merging")
    
    args = parser.parse_args()
    
    results = run_rtp_experiment(
        data_dir=args.data_dir,
        pattern=args.pattern,
        num_parts=args.num_parts,
        sample_size=args.sample_size,
        model_name=args.model_name,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        max_length=args.max_length,
        merge_first=not args.no_merge
    )