#!/usr/bin/env python3
"""
Publication-Ready Plot Generator for OmniQ Framework

This script generates high-quality, publication-ready figures optimized for
academic papers, conference presentations, and research publications.
Includes both combined multi-panel figures and individual focused plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PublicationPlotGenerator:
    def __init__(self, csv_path="results/summary.csv", output_dir="plots/publication_plots"):
        """Initialize the publication plot generator."""
        self.csv_path = csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
        
        # Set publication style
        self.setup_publication_style()
        
        # Define academic paper appropriate colors (blues and grays)
        self.pub_colors = {
            'swin_tiny_2d_temporalavg': '#1f4e79',  # Dark blue
            'omniq_transformer': '#4472c4',         # Medium blue
            'omniq_mamba': '#8db4e2'                # Light blue
        }
    
    def prepare_data(self):
        """Prepare data for publication plots."""
        # Create simplified model names for publication
        self.df['Model'] = self.df['model'].apply(self.simplify_model_name)
        
        # Calculate efficiency metrics
        self.df['param_efficiency'] = self.df['top1'] / self.df['params_M']
        self.df['memory_efficiency'] = self.df['top1'] / self.df['peak_vram_GB']
        self.df['speed_efficiency'] = self.df['top1'] / self.df['avg_latency_ms']
        
    def simplify_model_name(self, model_name):
        """Simplify model names for publication."""
        name_map = {
            'swin_tiny_2d_temporalavg': 'Swin-Tiny',
            'omniq_transformer': 'OmniQ-Transformer',
            'omniq_mamba': 'OmniQ-Mamba'
        }
        return name_map.get(model_name, model_name)
    
    def setup_publication_style(self):
        """Set up publication-quality matplotlib style."""
        plt.style.use('default')

        # Publication parameters
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'serif'],
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.edgecolor': 'black',
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_main_results_figure(self):
        """Generate the main results figure for publication."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('OmniQ Framework: Comprehensive Performance Analysis', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        models = self.df['Model']
        colors = [self.pub_colors.get(model, 'gray') for model in self.df['model']]
        
        # 1. Model Accuracy Comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(models, self.df['top1'], color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Top-1 Accuracy (%)')
        ax1.set_title('(a) Classification Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, self.df['top1']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Parameter Efficiency
        ax2 = axes[0, 1]
        param_eff = self.df['top1'] / self.df['params_M']
        bars = ax2.bar(models, param_eff, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Accuracy per Million Parameters')
        ax2.set_title('(b) Parameter Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Memory Usage Analysis
        ax3 = axes[0, 2]
        scatter = ax3.scatter(self.df['peak_vram_GB'], self.df['top1'], 
                             c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        for i, model in enumerate(models):
            ax3.annotate(model, (self.df['peak_vram_GB'].iloc[i], self.df['top1'].iloc[i]),
                        xytext=(3, 3), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Peak Memory (GB)')
        ax3.set_ylabel('Top-1 Accuracy (%)')
        ax3.set_title('(c) Memory Efficiency')
        
        # 4. Inference Latency
        ax4 = axes[1, 0]
        bars = ax4.bar(models, self.df['avg_latency_ms'], color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax4.set_ylabel('Inference Latency (ms)')
        ax4.set_title('(d) Inference Speed')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Efficiency Frontier
        ax5 = axes[1, 1]
        # Normalize metrics for comparison
        norm_acc = self.df['top1'] / self.df['top1'].max()
        norm_params = 1 - (self.df['params_M'] - self.df['params_M'].min()) / \
                     (self.df['params_M'].max() - self.df['params_M'].min())
        
        ax5.scatter(norm_params, norm_acc, c=colors, s=100, alpha=0.8,
                   edgecolors='black', linewidth=0.5)
        
        for i, model in enumerate(models):
            ax5.annotate(model, (norm_params.iloc[i], norm_acc.iloc[i]),
                        xytext=(3, 3), textcoords='offset points', fontsize=9)
        
        ax5.set_xlabel('Parameter Efficiency (normalized)')
        ax5.set_ylabel('Accuracy (normalized)')
        ax5.set_title('(e) Efficiency Frontier')
        
        # 6. Resource Utilization Summary
        ax6 = axes[1, 2]
        
        # Create a summary score
        summary_scores = []
        for _, row in self.df.iterrows():
            score = row['top1'] / (row['params_M'] + row['peak_vram_GB'] + row['avg_latency_ms']/10)
            summary_scores.append(score)
        
        bars = ax6.bar(models, summary_scores, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax6.set_ylabel('Overall Efficiency Score')
        ax6.set_title('(f) Resource Utilization')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'main_results_figure.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'main_results_figure.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_efficiency_frontier(self):
        """Generate efficiency frontier analysis."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create efficiency metrics
        accuracy = self.df['top1']
        efficiency = self.df['top1'] / (self.df['params_M'] + self.df['peak_vram_GB'])
        
        colors = [self.pub_colors.get(model, 'gray') for model in self.df['model']]
        
        scatter = ax.scatter(self.df['params_M'], accuracy, 
                           c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add model labels
        for i, model in enumerate(self.df['Model']):
            ax.annotate(model, (self.df['params_M'].iloc[i], accuracy.iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Model Parameters (Millions)', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title('Performance vs Model Complexity Trade-off', fontsize=14, fontweight='bold')
        
        # Add efficiency lines
        x_range = np.linspace(self.df['params_M'].min(), self.df['params_M'].max(), 100)
        for eff_level in [1, 2, 3]:
            y_line = eff_level * x_range
            y_line = y_line[y_line <= accuracy.max()]
            x_line = x_range[:len(y_line)]
            ax.plot(x_line, y_line, '--', alpha=0.5, color='gray', linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_frontier.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'efficiency_frontier.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_fusion_comparison(self):
        """Generate fusion mechanism comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Fusion Mechanism Analysis', fontsize=16, fontweight='bold')
        
        models = self.df['Model']
        colors = [self.pub_colors.get(model, 'gray') for model in self.df['model']]
        
        # 1. Accuracy Comparison
        ax1 = axes[0]
        bars = ax1.bar(models, self.df['top1'], color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Top-1 Accuracy (%)')
        ax1.set_title('(a) Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Computational Complexity
        ax2 = axes[1]
        bars = ax2.bar(models, self.df['params_M'], color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Parameters (Millions)')
        ax2.set_title('(b) Model Complexity')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Efficiency Score
        ax3 = axes[2]
        efficiency_scores = self.df['top1'] / self.df['params_M']
        bars = ax3.bar(models, efficiency_scores, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        ax3.set_ylabel('Accuracy per Parameter (×10⁻⁶)')
        ax3.set_title('(c) Parameter Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fusion_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fusion_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def generate_publication_table(self):
        """Generate publication-ready table."""
        # Create publication table
        pub_table = self.df[['Model', 'top1', 'top5', 'params_M', 'peak_vram_GB', 'avg_latency_ms']].copy()
        pub_table.columns = ['Model', 'Top-1 (%)', 'Top-5 (%)', 'Params (M)', 'Memory (GB)', 'Latency (ms)']
        
        # Round values for publication
        pub_table['Top-1 (%)'] = pub_table['Top-1 (%)'].round(1)
        pub_table['Top-5 (%)'] = pub_table['Top-5 (%)'].round(1)
        pub_table['Params (M)'] = pub_table['Params (M)'].round(1)
        pub_table['Memory (GB)'] = pub_table['Memory (GB)'].round(1)
        pub_table['Latency (ms)'] = pub_table['Latency (ms)'].round(1)
        
        # Save as CSV
        pub_table.to_csv(self.output_dir / 'results_table.csv', index=False)
        
        # Generate LaTeX table
        latex_table = pub_table.to_latex(index=False, float_format='%.1f',
                                        caption='Experimental Results on UCF101 Dataset',
                                        label='tab:results')
        
        # Save LaTeX table
        with open(self.output_dir / 'results_table.tex', 'w') as f:
            f.write(latex_table)
        
        print("Publication table saved as:")
        print(f"  - CSV: {self.output_dir / 'results_table.csv'}")
        print(f"  - LaTeX: {self.output_dir / 'results_table.tex'}")
    
    def save_individual_subplots(self):
        """Save individual plots from the main results figure."""
        print("5. Saving Individual Subplots...")
        
        models = self.df['Model']
        colors = [self.pub_colors.get(model, 'gray') for model in self.df['model']]
        
        # Individual accuracy plot
        fig, ax = plt.subplots(figsize=(8, 6))
        accuracies = self.df['top1']
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('Classification Accuracy Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_accuracy.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_accuracy.pdf', bbox_inches='tight')
        plt.close()
        
        # Individual parameter efficiency plot
        fig, ax = plt.subplots(figsize=(8, 6))
        param_eff = self.df['top1'] / self.df['params_M']
        bars = ax.bar(models, param_eff, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Accuracy per Million Parameters')
        ax.set_title('Parameter Efficiency Analysis')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_param_efficiency.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_param_efficiency.pdf', bbox_inches='tight')
        plt.close()
        
        # Individual memory vs accuracy plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(self.df['peak_vram_GB'], self.df['top1'], 
                           c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        for i, model in enumerate(models):
            ax.annotate(model, (self.df['peak_vram_GB'].iloc[i], self.df['top1'].iloc[i]),
                       xytext=(3, 3), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Peak Memory (GB)')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title('Memory Efficiency Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_memory_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_memory_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        # Individual latency analysis plot
        fig, ax = plt.subplots(figsize=(8, 6))
        latencies = self.df['avg_latency_ms']
        bars = ax.bar(models, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Inference Latency (ms)')
        ax.set_title('Inference Speed Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_latency.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_latency.pdf', bbox_inches='tight')
        plt.close()

    def generate_all_publication_plots(self):
        """Generate all publication-ready plots."""
        print("Generating Publication-Ready Plots for OmniQ Framework...")
        print("="*60)
        
        print("1. Main Results Figure...")
        self.plot_main_results_figure()
        
        print("2. Efficiency Frontier Analysis...")
        self.plot_efficiency_frontier()
        
        print("3. Fusion Mechanism Comparison...")
        self.plot_fusion_comparison()
        
        print("4. Publication Table...")
        self.generate_publication_table()
        
        print("5. Individual Subplot Figures...")
        self.save_individual_subplots()
        
        print(f"\nAll publication plots saved to: {self.output_dir}")
        print("Files generated:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")
        
        print("\nRecommended usage:")
        print("  - main_results_figure.pdf: Main paper figure")
        print("  - efficiency_frontier.pdf: Performance analysis")
        print("  - fusion_comparison.pdf: Architecture comparison")
        print("  - individual_*.pdf: Individual subplot figures")
        print("  - results_table.tex: LaTeX table for paper")

def main():
    """Main function to generate publication plots."""
    generator = PublicationPlotGenerator()
    generator.generate_all_publication_plots()

if __name__ == "__main__":
    main()
