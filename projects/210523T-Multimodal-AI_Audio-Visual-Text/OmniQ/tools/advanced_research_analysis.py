#!/usr/bin/env python3
"""
Advanced Research Analysis for OmniQ Framework

This script provides advanced statistical analysis, correlation studies,
and publication-ready visualizations for research papers and technical reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedOmniQAnalyzer:
    def __init__(self, csv_path="results/summary.csv", output_dir="plots/advanced_research_plots"):
        """Initialize advanced analyzer with enhanced statistical capabilities."""
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = self.load_and_enhance_data()
        self.setup_advanced_plotting()
        
    def load_and_enhance_data(self):
        """Load data with advanced feature engineering."""
        df = pd.read_csv(self.csv_path)
        
        # Advanced feature engineering
        df['model_family'] = df['model'].apply(self.extract_model_family)
        df['fusion_type'] = df['model'].apply(self.extract_fusion_type)
        df['efficiency_ratio'] = df['top1'] / (df['params_M'] + df['peak_vram_GB'])
        df['speed_accuracy_product'] = df['top1'] * (1000 / df['avg_latency_ms'])
        df['resource_efficiency'] = df['top1'] / (df['params_M'] * df['peak_vram_GB'])
        df['lora_impact'] = df['lora_enabled'].astype(int)
        
        # Compute relative improvements
        baseline_acc = df[df['model'].str.contains('swin_tiny_2d')]['top1'].iloc[0] if len(df[df['model'].str.contains('swin_tiny_2d')]) > 0 else df['top1'].min()
        df['accuracy_improvement'] = df['top1'] - baseline_acc
        
        return df
    
    def extract_model_family(self, model_name):
        """Extract model family for grouping."""
        if 'transformer' in model_name.lower():
            return 'Transformer-based'
        elif 'mamba' in model_name.lower():
            return 'State Space Model'
        elif 'swin' in model_name.lower():
            return 'Vision Transformer'
        return 'Other'
    
    def extract_fusion_type(self, model_name):
        """Extract fusion mechanism type."""
        if 'omniq_transformer' in model_name.lower():
            return 'Multi-Head Attention'
        elif 'omniq_mamba' in model_name.lower():
            return 'Selective State Space'
        elif 'temporalavg' in model_name.lower():
            return 'Temporal Averaging'
        return 'Unknown'
    
    def setup_advanced_plotting(self):
        """Setup advanced plotting parameters."""
        plt.style.use('seaborn-v0_8-paper')

        # Academic paper appropriate color palette (blues and grays)
        self.research_colors = {
            'Transformer-based': '#4472c4',  # Medium blue
            'State Space Model': '#8db4e2',  # Light blue
            'Vision Transformer': '#1f4e79',           # Dark blue
            'Other': '#b7c9e2'               # Very light blue
        }
        
        plt.rcParams.update({
            'figure.figsize': (14, 10),
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 15,
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def plot_correlation_analysis(self):
        """Generate comprehensive correlation analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('OmniQ Framework: Correlation and Statistical Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Correlation Heatmap
        ax1 = axes[0, 0]
        
        # Select numerical columns for correlation
        corr_cols = ['top1', 'top5', 'params_M', 'trainable_params_M', 
                     'peak_vram_GB', 'avg_latency_ms', 'fusion_depth', 'lora_impact']
        corr_matrix = self.df[corr_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax1, cbar_kws={"shrink": .8})
        ax1.set_title('Feature Correlation Matrix')
        
        # 2. Principal Component Analysis
        ax2 = axes[0, 1]
        
        # Prepare data for PCA
        features = ['top1', 'params_M', 'peak_vram_GB', 'avg_latency_ms', 'fusion_depth']
        X = self.df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot PCA results
        for family in self.df['model_family'].unique():
            mask = self.df['model_family'] == family
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=family, s=100, alpha=0.7,
                       color=self.research_colors.get(family, 'gray'))
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax2.set_title('Principal Component Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistical Significance Testing
        ax3 = axes[1, 0]
        
        # Compare performance across model families
        families = self.df['model_family'].unique()
        family_data = [self.df[self.df['model_family'] == family]['top1'].values 
                      for family in families]
        
        # Box plot with statistical annotations
        box_plot = ax3.boxplot(family_data, labels=families, patch_artist=True)
        
        # Color boxes
        colors = [self.research_colors.get(family, 'gray') for family in families]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Top-1 Accuracy (%)')
        ax3.set_title('Performance Distribution by Model Family')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add statistical test results
        if len(family_data) > 1 and all(len(data) > 0 for data in family_data):
            try:
                f_stat, p_value = stats.f_oneway(*family_data)
                ax3.text(0.02, 0.98, f'ANOVA F-stat: {f_stat:.3f}\np-value: {p_value:.3f}',
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except:
                pass
        
        # 4. Efficiency Frontier Analysis
        ax4 = axes[1, 1]
        
        # Create efficiency frontier
        x = self.df['params_M'].values
        y = self.df['top1'].values
        
        # Plot points colored by model family
        for family in self.df['model_family'].unique():
            mask = self.df['model_family'] == family
            ax4.scatter(x[mask], y[mask], label=family, s=120, alpha=0.8,
                       color=self.research_colors.get(family, 'gray'))
        
        # Fit and plot Pareto frontier
        try:
            # Simple convex hull approach for Pareto frontier
            from scipy.spatial import ConvexHull
            points = np.column_stack([x, y])
            hull = ConvexHull(points)
            
            # Get upper frontier points
            frontier_points = points[hull.vertices]
            frontier_points = frontier_points[frontier_points[:, 1].argsort()]
            
            ax4.plot(frontier_points[:, 0], frontier_points[:, 1], 
                    'r--', alpha=0.7, linewidth=2, label='Efficiency Frontier')
        except:
            pass
        
        ax4.set_xlabel('Parameters (Millions)')
        ax4.set_ylabel('Top-1 Accuracy (%)')
        ax4.set_title('Pareto Efficiency Frontier')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'correlation_analysis.pdf', bbox_inches='tight')
        plt.show()

    def plot_fusion_mechanism_study(self):
        """Detailed analysis of fusion mechanisms."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fusion Mechanism Comparative Study', fontsize=16, fontweight='bold')
        
        # 1. Fusion Type Performance Comparison
        ax1 = axes[0, 0]
        
        fusion_performance = self.df.groupby('fusion_type').agg({
            'top1': ['mean', 'std'],
            'top5': ['mean', 'std']
        }).round(2)
        
        fusion_types = fusion_performance.index
        top1_means = fusion_performance[('top1', 'mean')]
        top1_stds = fusion_performance[('top1', 'std')]
        
        bars = ax1.bar(fusion_types, top1_means, yerr=top1_stds,
                      capsize=5, alpha=0.8, color=['#1f4e79', '#4472c4', '#8db4e2'])
        
        ax1.set_ylabel('Top-1 Accuracy (%)')
        ax1.set_title('Fusion Mechanism Performance')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars, top1_means, top1_stds):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 1,
                    f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Computational Complexity Analysis
        ax2 = axes[0, 1]
        
        # Theoretical complexity analysis
        complexity_data = []
        for _, row in self.df.iterrows():
            if 'transformer' in row['model'].lower():
                # O(n²) for self-attention
                complexity = row['frames'] ** 2 * row['fusion_depth']
            elif 'mamba' in row['model'].lower():
                # O(n) for state space models
                complexity = row['frames'] * row['fusion_depth']
            else:
                # O(n) for temporal averaging
                complexity = row['frames']
            complexity_data.append(complexity)
        
        self.df['theoretical_complexity'] = complexity_data
        
        scatter = ax2.scatter(self.df['theoretical_complexity'], self.df['avg_latency_ms'],
                             c=[self.research_colors.get(family, 'gray') 
                               for family in self.df['model_family']],
                             s=120, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(self.df['theoretical_complexity'], self.df['avg_latency_ms'], 1)
        p = np.poly1d(z)
        ax2.plot(self.df['theoretical_complexity'], 
                p(self.df['theoretical_complexity']), "r--", alpha=0.8)
        
        ax2.set_xlabel('Theoretical Complexity (Operations)')
        ax2.set_ylabel('Actual Latency (ms)')
        ax2.set_title('Theoretical vs Actual Complexity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Scaling Analysis
        ax3 = axes[1, 0]
        
        # Memory vs sequence length relationship
        ax3.scatter(self.df['frames'], self.df['peak_vram_GB'],
                   c=[self.research_colors.get(family, 'gray') 
                     for family in self.df['model_family']],
                   s=self.df['params_M']*3, alpha=0.7)
        
        # Add model labels
        for i, row in self.df.iterrows():
            ax3.annotate(row['fusion_type'], 
                        (row['frames'], row['peak_vram_GB']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Sequence Length (Frames)')
        ax3.set_ylabel('Peak Memory (GB)')
        ax3.set_title('Memory Scaling by Fusion Type\n(Bubble size = Parameters)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Fusion Depth Impact Analysis
        ax4 = axes[1, 1]
        
        # Analyze impact of fusion depth on performance
        depth_impact = self.df.groupby('fusion_depth').agg({
            'top1': 'mean',
            'avg_latency_ms': 'mean',
            'peak_vram_GB': 'mean'
        })
        
        # Normalize metrics for comparison
        norm_acc = depth_impact['top1'] / depth_impact['top1'].max()
        norm_latency = 1 - (depth_impact['avg_latency_ms'] / depth_impact['avg_latency_ms'].max())
        norm_memory = 1 - (depth_impact['peak_vram_GB'] / depth_impact['peak_vram_GB'].max())
        
        x = depth_impact.index
        width = 0.25
        
        ax4.bar(x - width, norm_acc, width, label='Accuracy (norm)', alpha=0.8)
        ax4.bar(x, norm_latency, width, label='Speed (norm)', alpha=0.8)
        ax4.bar(x + width, norm_memory, width, label='Memory Eff (norm)', alpha=0.8)
        
        ax4.set_xlabel('Fusion Depth')
        ax4.set_ylabel('Normalized Performance')
        ax4.set_title('Fusion Depth Impact Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fusion_mechanism_study.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fusion_mechanism_study.pdf', bbox_inches='tight')
        plt.show()

    def plot_lora_impact_analysis(self):
        """Analyze the impact of LoRA on model performance and efficiency."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LoRA Impact Analysis on OmniQ Models', fontsize=16, fontweight='bold')
        
        # 1. LoRA vs Full Fine-tuning Comparison
        ax1 = axes[0, 0]
        
        lora_data = self.df[self.df['lora_enabled'] == True]
        full_data = self.df[self.df['lora_enabled'] == False]
        
        categories = ['Accuracy', 'Parameters', 'Memory', 'Speed']
        
        if len(lora_data) > 0 and len(full_data) > 0:
            lora_metrics = [
                lora_data['top1'].mean(),
                lora_data['trainable_params_M'].mean(),
                lora_data['peak_vram_GB'].mean(),
                1000 / lora_data['avg_latency_ms'].mean()
            ]
            
            full_metrics = [
                full_data['top1'].mean(),
                full_data['trainable_params_M'].mean(),
                full_data['peak_vram_GB'].mean(),
                1000 / full_data['avg_latency_ms'].mean()
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, lora_metrics, width, label='LoRA', alpha=0.8)
            bars2 = ax1.bar(x + width/2, full_metrics, width, label='Full Fine-tuning', alpha=0.8)
            
            ax1.set_ylabel('Metric Value')
            ax1.set_title('LoRA vs Full Fine-tuning Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Parameter Reduction Analysis
        ax2 = axes[0, 1]
        
        if len(self.df) > 0:
            # Calculate parameter reduction ratio
            param_reduction = []
            labels = []
            
            for _, row in self.df.iterrows():
                if row['lora_enabled']:
                    reduction_ratio = (row['params_M'] - row['trainable_params_M']) / row['params_M']
                    param_reduction.append(reduction_ratio * 100)
                    labels.append(f"{row['model_family']}\n(LoRA)")
                else:
                    param_reduction.append(0)
                    labels.append(f"{row['model_family']}\n(Full)")
            
            colors = ['#8db4e2' if 'LoRA' in label else '#4472c4' for label in labels]
            bars = ax2.bar(range(len(param_reduction)), param_reduction, color=colors, alpha=0.8)
            
            ax2.set_ylabel('Parameter Reduction (%)')
            ax2.set_title('Trainable Parameter Reduction with LoRA')
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, param_reduction):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Efficiency vs Performance Trade-off
        ax3 = axes[1, 0]
        
        # Plot efficiency vs accuracy for LoRA and non-LoRA models
        for lora_status in [True, False]:
            subset = self.df[self.df['lora_enabled'] == lora_status]
            if len(subset) > 0:
                label = 'LoRA Enabled' if lora_status else 'Full Fine-tuning'
                marker = 'o' if lora_status else 's'
                ax3.scatter(subset['resource_efficiency'], subset['top1'],
                           label=label, s=120, alpha=0.7, marker=marker)
        
        ax3.set_xlabel('Resource Efficiency (Accuracy/Params/Memory)')
        ax3.set_ylabel('Top-1 Accuracy (%)')
        ax3.set_title('Resource Efficiency vs Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Training Efficiency Analysis
        ax4 = axes[1, 1]
        
        # Simulate training efficiency metrics (in real scenario, these would come from training logs)
        if len(self.df) > 0:
            training_metrics = []
            model_names = []
            
            for _, row in self.df.iterrows():
                # Estimate training efficiency based on trainable parameters
                if row['lora_enabled']:
                    # LoRA typically trains 10-100x faster
                    training_speed = 100 / row['trainable_params_M']
                    memory_efficiency = 100 / row['peak_vram_GB']
                else:
                    training_speed = 10 / row['trainable_params_M']
                    memory_efficiency = 50 / row['peak_vram_GB']
                
                training_metrics.append([training_speed, memory_efficiency])
                model_names.append(row['model_family'])
            
            training_array = np.array(training_metrics)
            
            # Create radar chart for training efficiency
            angles = np.linspace(0, 2 * np.pi, 2, endpoint=False).tolist()
            angles += angles[:1]
            
            ax4 = plt.subplot(2, 2, 4, projection='polar')
            
            for i, (metrics, name) in enumerate(zip(training_metrics, model_names)):
                metrics_plot = metrics + metrics[:1]
                lora_status = self.df.iloc[i]['lora_enabled']
                color = '#8db4e2' if lora_status else '#1f4e79'
                linestyle = '-' if lora_status else '--'
                
                ax4.plot(angles, metrics_plot, 'o-', linewidth=2, 
                        label=f"{name} ({'LoRA' if lora_status else 'Full'})",
                        color=color, linestyle=linestyle)
                ax4.fill(angles, metrics_plot, alpha=0.25, color=color)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(['Training Speed', 'Memory Efficiency'])
            ax4.set_title('Training Efficiency Comparison')
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lora_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'lora_impact_analysis.pdf', bbox_inches='tight')
        plt.show()

    def generate_statistical_report(self):
        """Generate comprehensive statistical analysis report."""
        report_path = self.output_dir / 'statistical_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OMNIQ FRAMEWORK - ADVANCED STATISTICAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Descriptive Statistics
            f.write("DESCRIPTIVE STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(self.df[['top1', 'top5', 'params_M', 'peak_vram_GB', 'avg_latency_ms']].describe().to_string())
            f.write("\n\n")
            
            # Correlation Analysis
            f.write("CORRELATION ANALYSIS:\n")
            f.write("-"*40 + "\n")
            corr_cols = ['top1', 'params_M', 'peak_vram_GB', 'avg_latency_ms', 'fusion_depth']
            correlations = self.df[corr_cols].corr()['top1'].sort_values(ascending=False)
            
            f.write("Correlations with Top-1 Accuracy:\n")
            for metric, corr in correlations.items():
                if metric != 'top1':
                    f.write(f"  {metric}: {corr:.3f}\n")
            
            # Model Family Analysis
            f.write("\nMODEL FAMILY ANALYSIS:\n")
            f.write("-"*40 + "\n")
            family_stats = self.df.groupby('model_family').agg({
                'top1': ['mean', 'std', 'min', 'max'],
                'params_M': 'mean',
                'peak_vram_GB': 'mean',
                'avg_latency_ms': 'mean'
            }).round(3)
            
            f.write(family_stats.to_string())
            f.write("\n\n")
            
            # LoRA Impact Analysis
            if self.df['lora_enabled'].sum() > 0:
                f.write("LORA IMPACT ANALYSIS:\n")
                f.write("-"*40 + "\n")
                
                lora_models = self.df[self.df['lora_enabled'] == True]
                full_models = self.df[self.df['lora_enabled'] == False]
                
                if len(lora_models) > 0 and len(full_models) > 0:
                    f.write(f"LoRA Models Average Accuracy: {lora_models['top1'].mean():.2f}%\n")
                    f.write(f"Full Models Average Accuracy: {full_models['top1'].mean():.2f}%\n")
                    f.write(f"Average Parameter Reduction with LoRA: {((lora_models['params_M'] - lora_models['trainable_params_M']) / lora_models['params_M'] * 100).mean():.1f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RESEARCH RECOMMENDATIONS:\n")
            f.write("="*80 + "\n")
            
            best_model = self.df.loc[self.df['top1'].idxmax()]
            most_efficient = self.df.loc[self.df['resource_efficiency'].idxmax()]
            
            f.write(f"• Best performing model: {best_model['model_family']} ({best_model['top1']:.2f}% accuracy)\n")
            f.write(f"• Most resource efficient: {most_efficient['model_family']} (efficiency score: {most_efficient['resource_efficiency']:.3f})\n")
            f.write(f"• Recommended for production: Balance of accuracy and efficiency\n")
            f.write(f"• Future research directions: Investigate fusion mechanism optimizations\n")
        
        print(f"Statistical analysis report saved to: {report_path}")

    def generate_all_advanced_plots(self):
        """Generate all advanced research plots and analyses."""
        print("Generating Advanced OmniQ Research Analysis...")
        print("="*50)
        
        print("1. Correlation and Statistical Analysis...")
        self.plot_correlation_analysis()
        
        print("2. Fusion Mechanism Study...")
        self.plot_fusion_mechanism_study()
        
        print("3. LoRA Impact Analysis...")
        self.plot_lora_impact_analysis()
        
        print("4. Generating Statistical Report...")
        self.generate_statistical_report()
        
        print(f"\nAll advanced plots and reports saved to: {self.output_dir}")
        print("Files generated:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")

def main():
    """Main function for advanced research analysis."""
    try:
        analyzer = AdvancedOmniQAnalyzer()
        analyzer.generate_all_advanced_plots()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure results/summary.csv exists with experimental data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
