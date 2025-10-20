"""
Generate Results: Figures and Tables for the Paper
Creates visualizations comparing CAFE and Perspective API
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jigsaw_metrics import calculate_bpsn_auc, calculate_bnsp_auc
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def plot_score_distributions(df, output_file='figure1_score_distributions.png'):
    """
    Generate Figure 1: Density distributions of CAFE vs Perspective scores.
    
    Args:
        df (pd.DataFrame): RTP results with CAFE_score and Perspective_score
        output_file (str): Output filename for the plot
    """
    print("\nGenerating Figure 1: Score Distributions...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot density distributions
    sns.kdeplot(
        data=df['CAFE_score'],
        label='CAFE-Jigsaw',
        linewidth=2.5,
        color='#2E86AB',
        fill=True,
        alpha=0.3,
        ax=ax
    )
    
    sns.kdeplot(
        data=df['Perspective_score'],
        label='Perspective API',
        linewidth=2.5,
        color='#A23B72',
        fill=True,
        alpha=0.3,
        ax=ax
    )
    
    ax.set_xlabel('Toxicity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Toxicity Scores: CAFE vs Perspective API', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, 1)
    
    # Add statistics text box
    stats_text = (
        f"CAFE: μ={df['CAFE_score'].mean():.3f}, σ={df['CAFE_score'].std():.3f}\n"
        f"Perspective: μ={df['Perspective_score'].mean():.3f}, σ={df['Perspective_score'].std():.3f}\n"
        f"Correlation: r={df[['CAFE_score', 'Perspective_score']].corr().iloc[0,1]:.3f}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    
    return fig


def analyze_perspective_bias(df, output_table='table2_perspective_bias.md', 
                             output_figure='figure2_bpsn_by_identity.png'):
    """
    Generate Table 2 and Figure 2: Bias analysis of Perspective API.
    Uses CAFE scores as ground truth to evaluate Perspective.
    
    Args:
        df (pd.DataFrame): RTP results with scores and identity labels
        output_table (str): Output filename for markdown table
        output_figure (str): Output filename for bar chart
        
    Returns:
        pd.DataFrame: Bias metrics by identity group
    """
    print("\nGenerating Table 2 and Figure 2: Perspective API Bias Analysis...")
    
    # Identity columns
    identity_columns = [
        'male', 'female', 'transgender', 'heterosexual', 'homosexual_gay_or_lesbian',
        'bisexual', 'christian', 'jewish', 'muslim', 'black', 'white', 'asian',
        'latino', 'physical_disability', 'psychiatric_or_mental_illness'
    ]
    
    # Filter to available columns
    available_identities = [col for col in identity_columns if col in df.columns]
    
    results = []
    
    # Calculate bias metrics for each identity
    for identity in available_identities:
        subgroup_size = (df[identity] > 0).sum()
        
        if subgroup_size < 10:  # Skip small subgroups
            continue
        
        # Use CAFE scores as "ground truth" labels (binarize at 0.5)
        cafe_labels = (df['CAFE_score'] >= 0.5).astype(float)
        
        # Calculate BPSN and BNSP for Perspective API
        bpsn_auc = calculate_bpsn_auc(
            cafe_labels,
            df['Perspective_score'],
            df[identity]
        )
        
        bnsp_auc = calculate_bnsp_auc(
            cafe_labels,
            df['Perspective_score'],
            df[identity]
        )
        
        results.append({
            'Identity': identity.replace('_', ' ').title(),
            'Subgroup Size': int(subgroup_size),
            'BPSN AUC': bpsn_auc,
            'BNSP AUC': bnsp_auc,
            'Bias Score': abs(0.5 - bpsn_auc) + abs(0.5 - bnsp_auc)  # Combined bias metric
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Bias Score', ascending=False)
    
    # Save Table 2 (Markdown format)
    with open(output_table, 'w') as f:
        f.write("# Table 2: Perspective API Bias Metrics by Identity Group\n\n")
        f.write("*Using CAFE-Jigsaw scores as ground truth*\n\n")
        
        # Format table
        table_str = results_df.to_markdown(index=False, floatfmt='.4f')
        f.write(table_str)
        
        f.write("\n\n## Interpretation\n")
        f.write("- **BPSN AUC**: Lower values indicate the model incorrectly associates the identity with toxicity\n")
        f.write("- **BNSP AUC**: Lower values indicate the model fails to detect toxicity when identity is mentioned\n")
        f.write("- **Bias Score**: Combined metric (higher = more biased)\n")
    
    print(f"✓ Saved table to {output_table}")
    
    # Generate Figure 2: BPSN AUC bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by BPSN for visualization
    plot_df = results_df.sort_values('BPSN AUC')
    
    # Color bars by bias level (red for lower AUC = more bias)
    colors = plt.cm.RdYlGn(plot_df['BPSN AUC'])
    
    bars = ax.barh(plot_df['Identity'], plot_df['BPSN AUC'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add reference line at 0.5 (no bias)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='No Bias (0.5)')
    
    ax.set_xlabel('BPSN AUC', fontsize=12)
    ax.set_ylabel('Identity Group', fontsize=12)
    ax.set_title('Perspective API Bias: BPSN AUC by Identity Group', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(row['BPSN AUC'] + 0.02, i, f"{row['BPSN AUC']:.3f}",
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_figure, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to {output_figure}")
    
    # Print summary
    print("\n" + "="*80)
    print("Table 2: Perspective API Bias Metrics")
    print("="*80)
    print(results_df.to_string(index=False))
    
    return results_df


def find_disagreements(df, top_n=10, output_file='table3_disagreement_examples.csv'):
    """
    Generate Table 3: Examples where CAFE and Perspective strongly disagree.
    
    Args:
        df (pd.DataFrame): RTP results with scores
        top_n (int): Number of top disagreement examples
        output_file (str): Output CSV filename
        
    Returns:
        pd.DataFrame: Top disagreement examples
    """
    print(f"\nGenerating Table 3: Top {top_n} Disagreement Examples...")
    
    # Calculate absolute difference
    df['score_diff'] = abs(df['CAFE_score'] - df['Perspective_score'])
    
    # Get top disagreements
    top_disagreements = df.nlargest(top_n, 'score_diff')
    
    # Select relevant columns
    output_cols = [
        'full_text', 'CAFE_score', 'Perspective_score', 'score_diff',
        'male', 'female', 'black', 'white', 'asian', 'muslim', 'jewish', 'christian'
    ]
    
    # Filter to available columns
    available_cols = [col for col in output_cols if col in top_disagreements.columns]
    disagreement_table = top_disagreements[available_cols].copy()
    
    # Format text for readability
    disagreement_table['full_text'] = disagreement_table['full_text'].str[:200] + '...'
    
    # Save to CSV
    disagreement_table.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    
    # Print for manual review
    print("\n" + "="*80)
    print("Table 3: Examples of Maximum Disagreement")
    print("="*80)
    
    for idx, row in disagreement_table.iterrows():
        print(f"\n{'-'*80}")
        print(f"Text: {row['full_text'][:150]}...")
        print(f"CAFE Score: {row['CAFE_score']:.4f}")
        print(f"Perspective Score: {row['Perspective_score']:.4f}")
        print(f"Difference: {row['score_diff']:.4f}")
        
        # Show which identities are mentioned
        identity_cols = [col for col in row.index if col not in ['full_text', 'CAFE_score', 'Perspective_score', 'score_diff']]
        mentioned = [col for col in identity_cols if row[col] > 0]
        if mentioned:
            print(f"Identities mentioned: {', '.join(mentioned)}")
    
    return disagreement_table


def generate_all_results():
    """
    Main function to generate all figures and tables.
    """
    print("="*80)
    print("Generating Results for Paper")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    
    # Check if files exist
    if not pd.io.common.file_exists('rtp_scores.csv'):
        raise FileNotFoundError(
            "rtp_scores.csv not found. Please run run_rtp_experiment.py first."
        )
    
    rtp_df = pd.read_csv('rtp_scores.csv')
    print(f"Loaded {len(rtp_df)} RTP samples")
    
    # Generate Figure 1
    fig1 = plot_score_distributions(rtp_df)
    
    # Generate Table 2 and Figure 2
    bias_results = analyze_perspective_bias(rtp_df)
    
    # Generate Table 3
    disagreements = find_disagreements(rtp_df, top_n=10)
    
    print("\n" + "="*80)
    print("All Results Generated Successfully!")
    print("="*80)
    print("\nOutput files:")
    print("  - figure1_score_distributions.png")
    print("  - table2_perspective_bias.md")
    print("  - figure2_bpsn_by_identity.png")
    print("  - table3_disagreement_examples.csv")
    
    # Additional summary statistics
    print("\n" + "="*80)
    print("Additional Statistics")
    print("="*80)
    
    print(f"\nMean BPSN AUC: {bias_results['BPSN AUC'].mean():.4f}")
    print(f"Mean BNSP AUC: {bias_results['BNSP AUC'].mean():.4f}")
    print(f"Mean Bias Score: {bias_results['Bias Score'].mean():.4f}")
    
    print(f"\nMost biased identity (highest bias score): {bias_results.iloc[0]['Identity']}")
    print(f"Least biased identity (lowest bias score): {bias_results.iloc[-1]['Identity']}")


if __name__ == "__main__":
    generate_all_results()