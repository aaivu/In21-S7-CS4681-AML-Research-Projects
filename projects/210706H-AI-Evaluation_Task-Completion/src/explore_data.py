"""
Data Exploration Script
Analyze the Jigsaw dataset to understand its characteristics and biases
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_data(filepath='jigsaw_train_preprocessed.parquet'):
    """Load preprocessed data."""
    print(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"⚠ Warning: {filepath} not found.")
        print("Available files in current directory:")
        for f in os.listdir('.'):
            if 'jigsaw' in f.lower() and f.endswith('.parquet'):
                print(f"  - {f}")
        raise FileNotFoundError(f"Please run preprocessing.py first to generate {filepath}")
    
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df):,} samples\n")
    return df


def analyze_toxicity_distribution(df):
    """Analyze the distribution of toxicity scores."""
    print("="*80)
    print("Toxicity Distribution Analysis")
    print("="*80)
    
    # Basic statistics
    print(f"\nTarget Statistics:")
    print(f"  Mean: {df['target'].mean():.4f}")
    print(f"  Median: {df['target'].median():.4f}")
    print(f"  Std: {df['target'].std():.4f}")
    print(f"  Min: {df['target'].min():.4f}")
    print(f"  Max: {df['target'].max():.4f}")
    
    # Binary classification (threshold at 0.5)
    toxic = (df['target'] >= 0.5).sum()
    non_toxic = (df['target'] < 0.5).sum()
    
    print(f"\nBinary Classification (threshold=0.5):")
    print(f"  Toxic: {toxic:,} ({toxic/len(df)*100:.2f}%)")
    print(f"  Non-toxic: {non_toxic:,} ({non_toxic/len(df)*100:.2f}%)")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['target'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    axes[0].set_xlabel('Toxicity Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Toxicity Scores')
    axes[0].legend()
    
    # Density plot
    sns.kdeplot(data=df['target'], ax=axes[1], fill=True)
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    axes[1].set_xlabel('Toxicity Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Density Plot of Toxicity Scores')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('exploration_toxicity_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: exploration_toxicity_distribution.png")
    
    return toxic, non_toxic


def analyze_auxiliary_targets(df):
    """Analyze the distribution of auxiliary toxicity types."""
    print("\n" + "="*80)
    print("Auxiliary Target Analysis")
    print("="*80)
    
    aux_targets = [
        'severe_toxicity', 'obscene', 'identity_attack',
        'insult', 'threat', 'sexual_explicit'
    ]
    
    stats = []
    for target in aux_targets:
        positive = (df[target] >= 0.5).sum()
        pct = positive / len(df) * 100
        stats.append({
            'target': target,
            'count': positive,
            'percentage': pct
        })
        print(f"  {target}: {positive:,} ({pct:.2f}%)")
    
    # Plot
    stats_df = pd.DataFrame(stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(stats_df['target'], stats_df['percentage'], 
                   color=sns.color_palette("viridis", len(stats_df)))
    ax.set_xlabel('Percentage of Positive Samples (%)')
    ax.set_ylabel('Toxicity Type')
    ax.set_title('Distribution of Auxiliary Toxicity Types')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('exploration_auxiliary_targets.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: exploration_auxiliary_targets.png")


def analyze_identity_mentions(df):
    """Analyze identity group mentions."""
    print("\n" + "="*80)
    print("Identity Group Analysis")
    print("="*80)
    
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
    available_identities = [col for col in identity_columns if col in df.columns]
    
    # Count mentions
    identity_counts = {}
    for identity in available_identities:
        count = (df[identity] > 0).sum()
        identity_counts[identity] = count
    
    # Sort by frequency
    sorted_identities = sorted(identity_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 Most Mentioned Identity Groups:")
    for i, (identity, count) in enumerate(sorted_identities[:15], 1):
        pct = count / len(df) * 100
        print(f"  {i:2d}. {identity:40s}: {count:8,} ({pct:5.2f}%)")
    
    # Check samples with no identity mentions
    any_identity = (df[available_identities] > 0).any(axis=1)
    no_identity_count = (~any_identity).sum()
    print(f"\nSamples with NO identity mention: {no_identity_count:,} ({no_identity_count/len(df)*100:.2f}%)")
    
    # Plot top identities
    top_identities = dict(sorted_identities[:15])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    identities = list(top_identities.keys())
    counts = list(top_identities.values())
    percentages = [c / len(df) * 100 for c in counts]
    
    bars = ax.barh(identities, percentages, color=sns.color_palette("coolwarm", len(identities)))
    ax.set_xlabel('Percentage of Samples (%)')
    ax.set_ylabel('Identity Group')
    ax.set_title('Top 15 Identity Groups by Mention Frequency')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('exploration_identity_mentions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: exploration_identity_mentions.png")
    
    return sorted_identities


def analyze_bias_patterns(df, identity_columns=None):
    """Analyze bias patterns: BPSN and BNSP."""
    print("\n" + "="*80)
    print("Bias Pattern Analysis (BPSN/BNSP)")
    print("="*80)
    
    if identity_columns is None:
        identity_columns = [
            'male', 'female', 'transgender', 'black', 'white', 'asian',
            'latino', 'muslim', 'jewish', 'christian', 'homosexual_gay_or_lesbian'
        ]
    
    # Filter to available columns
    available_identities = [col for col in identity_columns if col in df.columns]
    
    # Overall stats
    is_toxic = df['target'] >= 0.5
    has_identity = (df[available_identities] > 0).any(axis=1)
    
    # Count patterns
    background_negative = (~is_toxic) & (~has_identity)
    background_positive = is_toxic & (~has_identity)
    subgroup_negative = (~is_toxic) & has_identity
    subgroup_positive = is_toxic & has_identity
    
    print("\nDataset Composition:")
    print(f"  Background Negative (Non-toxic + No identity): {background_negative.sum():,} ({background_negative.mean()*100:.2f}%)")
    print(f"  Background Positive (Toxic + No identity):     {background_positive.sum():,} ({background_positive.mean()*100:.2f}%)")
    print(f"  Subgroup Negative (Non-toxic + Identity):      {subgroup_negative.sum():,} ({subgroup_negative.mean()*100:.2f}%)")
    print(f"  Subgroup Positive (Toxic + Identity):          {subgroup_positive.sum():,} ({subgroup_positive.mean()*100:.2f}%)")
    
    # BPSN and BNSP samples
    bpsn_samples = subgroup_negative  # Non-toxic + identity
    bnsp_samples = background_positive  # Toxic + no identity
    
    print(f"\nCritical Samples for Fairness:")
    print(f"  BPSN (important for avoiding false positives): {bpsn_samples.sum():,} ({bpsn_samples.mean()*100:.2f}%)")
    print(f"  BNSP (important for avoiding false negatives): {bnsp_samples.sum():,} ({bnsp_samples.mean()*100:.2f}%)")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Background\nNegative', 'Background\nPositive', 
                  'Subgroup\nNegative', 'Subgroup\nPositive']
    counts = [background_negative.sum(), background_positive.sum(),
              subgroup_negative.sum(), subgroup_positive.sum()]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Composition: Background vs Subgroup, Positive vs Negative')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{count:,}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('exploration_bias_patterns.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: exploration_bias_patterns.png")


def analyze_text_length(df):
    """Analyze text length distribution."""
    print("\n" + "="*80)
    print("Text Length Analysis")
    print("="*80)
    
    # Calculate lengths
    df['text_length'] = df['comment_text_cleaned'].str.len()
    df['word_count'] = df['comment_text_cleaned'].str.split().str.len()
    
    print(f"\nCharacter Length:")
    print(f"  Mean: {df['text_length'].mean():.1f}")
    print(f"  Median: {df['text_length'].median():.1f}")
    print(f"  Max: {df['text_length'].max()}")
    
    print(f"\nWord Count:")
    print(f"  Mean: {df['word_count'].mean():.1f}")
    print(f"  Median: {df['word_count'].median():.1f}")
    print(f"  Max: {df['word_count'].max()}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Character length
    axes[0].hist(df['text_length'].clip(upper=1000), bins=50, 
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Character Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Text Length (Characters)')
    axes[0].axvline(x=512, color='red', linestyle='--', linewidth=2, 
                    label='Model max (512 tokens ≈ 2000 chars)')
    axes[0].legend()
    
    # Word count
    axes[1].hist(df['word_count'].clip(upper=200), bins=50,
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Text Length (Words)')
    
    plt.tight_layout()
    plt.savefig('exploration_text_length.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: exploration_text_length.png")


def show_examples(df, n=5):
    """Show example texts from the dataset."""
    print("\n" + "="*80)
    print("Sample Examples")
    print("="*80)
    
    # Show diverse examples
    print("\n1. Non-Toxic Examples:")
    non_toxic = df[df['target'] < 0.3].sample(n=n, random_state=42)
    for i, row in enumerate(non_toxic.itertuples(), 1):
        print(f"\n{i}. Score: {row.target:.4f}")
        print(f"   Text: {row.comment_text_cleaned[:150]}...")
    
    print("\n\n2. Toxic Examples:")
    toxic = df[df['target'] > 0.7].sample(n=n, random_state=42)
    for i, row in enumerate(toxic.itertuples(), 1):
        print(f"\n{i}. Score: {row.target:.4f}")
        print(f"   Text: {row.comment_text_cleaned[:150]}...")
    
    print("\n\n3. Borderline Examples (around 0.5):")
    borderline = df[(df['target'] >= 0.45) & (df['target'] <= 0.55)].sample(n=n, random_state=42)
    for i, row in enumerate(borderline.itertuples(), 1):
        print(f"\n{i}. Score: {row.target:.4f}")
        print(f"   Text: {row.comment_text_cleaned[:150]}...")


def generate_summary_report(df):
    """Generate a comprehensive summary report."""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Toxic samples: {(df['target'] >= 0.5).sum():,} ({(df['target'] >= 0.5).mean()*100:.2f}%)")
    print(f"  Non-toxic samples: {(df['target'] < 0.5).sum():,} ({(df['target'] < 0.5).mean()*100:.2f}%)")
    
    # Identity coverage
    identity_cols = [col for col in df.columns if col in [
        'male', 'female', 'black', 'white', 'muslim', 'christian', 'jewish'
    ]]
    if identity_cols:
        any_identity = (df[identity_cols] > 0).any(axis=1)
        print(f"\n  Samples mentioning identity: {any_identity.sum():,} ({any_identity.mean()*100:.2f}%)")
    
    print(f"\nData Quality:")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate texts: {df['comment_text_cleaned'].duplicated().sum()}")
    
    print("\n✓ Exploration complete!")


def main():
    """Run all exploration analyses."""
    print("="*80)
    print("Jigsaw Dataset Exploration")
    print("="*80)
    
    # Load data
    df = load_data('jigsaw_train_preprocessed.parquet')
    
    # Run analyses
    analyze_toxicity_distribution(df)
    analyze_auxiliary_targets(df)
    analyze_identity_mentions(df)
    analyze_bias_patterns(df)
    analyze_text_length(df)
    show_examples(df, n=3)
    generate_summary_report(df)
    
    print("\n" + "="*80)
    print("All exploration visualizations saved!")
    print("="*80)
    print("\nGenerated files:")
    print("  - exploration_toxicity_distribution.png")
    print("  - exploration_auxiliary_targets.png")
    print("  - exploration_identity_mentions.png")
    print("  - exploration_bias_patterns.png")
    print("  - exploration_text_length.png")


if __name__ == "__main__":
    main()