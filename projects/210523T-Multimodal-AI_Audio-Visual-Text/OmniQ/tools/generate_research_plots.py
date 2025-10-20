#!/usr/bin/env python3
"""
Technical Research Directive Plots Generator for OmniQ Framework

This script generates comprehensive research-quality plots based on experimental results
from results/summary.csv for technical analysis and publication purposes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots with academic colors
plt.style.use('seaborn-v0_8-whitegrid')
# Academic paper appropriate color palette (blues and grays)
academic_colors = ['#1f4e79', '#4472c4', '#8db4e2', '#b7c9e2', '#d5e3f0']
sns.set_palette(academic_colors)

class OmniQResearchPlotter:
    def __init__(self, csv_path="results/summary.csv", output_dir="plots/research_plots"):
        """Initialize the research plotter with data and output directory."""
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load and preprocess data
        self.df = self.load_data()
        self.setup_plotting()
        
    def load_data(self):
        """Load and preprocess the experimental results."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        
        # Clean and enhance data
        df['model_type'] = df['model'].apply(self.categorize_model)
        df['efficiency_score'] = df['top1'] / (df['params_M'] * df['avg_latency_ms'] / 1000)
        df['memory_efficiency'] = df['top1'] / df['peak_vram_GB']
        df['param_efficiency'] = df['top1'] / df['params_M']
        
        return df
    
    def categorize_model(self, model_name):
        """Categorize models for better visualization."""
        if 'transformer' in model_name.lower():
            return 'Transformer'
        elif 'mamba' in model_name.lower():
            return 'Mamba SSM'
        elif 'swin' in model_name.lower():
            return 'Baseline'
        else:
            return 'Other'
    
    def setup_plotting(self):
        """Setup matplotlib parameters for publication quality."""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 8
        })

    def plot_performance_comparison(self):
        """Generate comprehensive performance comparison plots."""
        # Create combined figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('OmniQ Framework: Performance Analysis', fontsize=18, fontweight='bold')

        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        models = self.df['model_type']
        x_pos = np.arange(len(models))

        bars1 = ax1.bar(x_pos - 0.2, self.df['top1'], 0.4, label='Top-1 Accuracy', alpha=0.8)
        bars2 = ax1.bar(x_pos + 0.2, self.df['top5'], 0.4, label='Top-5 Accuracy', alpha=0.8)

        ax1.set_xlabel('Model Architecture')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Classification Accuracy Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Save individual accuracy plot
        fig_acc = plt.figure(figsize=(10, 6))
        ax_acc = fig_acc.add_subplot(111)
        bars1_ind = ax_acc.bar(x_pos - 0.2, self.df['top1'], 0.4, label='Top-1 Accuracy', alpha=0.8)
        bars2_ind = ax_acc.bar(x_pos + 0.2, self.df['top5'], 0.4, label='Top-5 Accuracy', alpha=0.8)
        ax_acc.set_xlabel('Model Architecture')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title('Classification Accuracy Comparison')
        ax_acc.set_xticks(x_pos)
        ax_acc.set_xticklabels(models, rotation=45)
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        for bar in bars1_ind:
            height = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_accuracy_comparison.pdf', bbox_inches='tight')
        plt.close(fig_acc)
        
        # 2. Parameter Efficiency
        ax2 = axes[0, 1]
        scatter = ax2.scatter(self.df['params_M'], self.df['top1'],
                             c=self.df.index, s=200, alpha=0.7, cmap='viridis')

        for i, model in enumerate(self.df['model_type']):
            ax2.annotate(model, (self.df['params_M'].iloc[i], self.df['top1'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)

        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_ylabel('Top-1 Accuracy (%)')
        ax2.set_title('Parameter Efficiency Analysis')
        ax2.grid(True, alpha=0.3)

        # Save individual parameter efficiency plot
        fig_param = plt.figure(figsize=(10, 6))
        ax_param = fig_param.add_subplot(111)
        scatter_ind = ax_param.scatter(self.df['params_M'], self.df['top1'],
                                      c=self.df.index, s=200, alpha=0.7, cmap='viridis')
        for i, model in enumerate(self.df['model_type']):
            ax_param.annotate(model, (self.df['params_M'].iloc[i], self.df['top1'].iloc[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax_param.set_xlabel('Parameters (Millions)')
        ax_param.set_ylabel('Top-1 Accuracy (%)')
        ax_param.set_title('Parameter Efficiency Analysis')
        ax_param.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_parameter_efficiency.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_parameter_efficiency.pdf', bbox_inches='tight')
        plt.close(fig_param)
        
        # 3. Memory vs Performance
        ax3 = axes[1, 0]
        bubble_sizes = self.df['avg_latency_ms'] * 2  # Scale for visibility
        scatter2 = ax3.scatter(self.df['peak_vram_GB'], self.df['top1'],
                              s=bubble_sizes, alpha=0.6, c=self.df.index, cmap='plasma')

        ax3.set_xlabel('Peak VRAM (GB)')
        ax3.set_ylabel('Top-1 Accuracy (%)')
        ax3.set_title('Memory Efficiency (Bubble size = Latency)')
        ax3.grid(True, alpha=0.3)

        # Save individual memory efficiency plot
        fig_mem = plt.figure(figsize=(10, 6))
        ax_mem = fig_mem.add_subplot(111)
        scatter2_ind = ax_mem.scatter(self.df['peak_vram_GB'], self.df['top1'],
                                     s=bubble_sizes, alpha=0.6, c=self.df.index, cmap='plasma')
        for i, model in enumerate(self.df['model_type']):
            ax_mem.annotate(model, (self.df['peak_vram_GB'].iloc[i], self.df['top1'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax_mem.set_xlabel('Peak VRAM (GB)')
        ax_mem.set_ylabel('Top-1 Accuracy (%)')
        ax_mem.set_title('Memory Efficiency Analysis (Bubble size = Latency)')
        ax_mem.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_memory_efficiency.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_memory_efficiency.pdf', bbox_inches='tight')
        plt.close(fig_mem)
        
        # 4. Efficiency Score Comparison
        ax4 = axes[1, 1]

        # Calculate efficiency scores
        efficiency_scores = []
        model_names = []

        for _, row in self.df.iterrows():
            # Simple efficiency score: accuracy per resource unit
            efficiency = row['top1'] / (row['params_M'] + row['peak_vram_GB'] + row['avg_latency_ms']/100)
            efficiency_scores.append(efficiency)
            model_names.append(row['model_type'])

        # Create bar chart
        colors = academic_colors[:len(efficiency_scores)]
        bars = ax4.bar(model_names, efficiency_scores, color=colors, alpha=0.8)

        ax4.set_ylabel('Efficiency Score')
        ax4.set_title('Overall Efficiency Comparison')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, score in zip(bars, efficiency_scores):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Save individual efficiency score plot
        fig_eff = plt.figure(figsize=(10, 6))
        ax_eff = fig_eff.add_subplot(111)
        bars_ind = ax_eff.bar(model_names, efficiency_scores, color=colors, alpha=0.8)
        ax_eff.set_ylabel('Efficiency Score')
        ax_eff.set_title('Overall Efficiency Comparison')
        ax_eff.tick_params(axis='x', rotation=45)
        ax_eff.grid(True, alpha=0.3, axis='y')
        for bar, score in zip(bars_ind, efficiency_scores):
            ax_eff.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_efficiency_score.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'individual_efficiency_score.pdf', bbox_inches='tight')
        plt.close(fig_eff)

        # Save combined plot
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_comparison.pdf', bbox_inches='tight')
        plt.show()

    def plot_architectural_analysis(self):
        """Generate architectural comparison and analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('OmniQ Framework: Architectural Analysis', fontsize=18, fontweight='bold')
        
        # 1. Model Complexity vs Performance
        ax1 = axes[0, 0]
        
        # Create complexity score based on parameters and fusion depth
        complexity_score = self.df['params_M'] * (self.df['fusion_depth'] + 1)
        
        colors = ['#8db4e2' if lora else '#1f4e79' for lora in self.df['lora_enabled']]
        scatter = ax1.scatter(complexity_score, self.df['top1'], c=colors, s=150, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(complexity_score, self.df['top1'], 1)
        p = np.poly1d(z)
        ax1.plot(complexity_score, p(complexity_score), "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Model Complexity Score (Params × Fusion Depth)')
        ax1.set_ylabel('Top-1 Accuracy (%)')
        ax1.set_title('Complexity vs Performance Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # Custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f4e79', label='Without LoRA'),
                          Patch(facecolor='#8db4e2', label='With LoRA')]
        ax1.legend(handles=legend_elements)
        
        # 2. Fusion Architecture Comparison
        ax2 = axes[0, 1]
        
        fusion_types = []
        for _, row in self.df.iterrows():
            if row['model_type'] == 'Transformer':
                fusion_types.append('Transformer\nFusion')
            elif row['model_type'] == 'Mamba SSM':
                fusion_types.append('Mamba\nFusion')
            else:
                fusion_types.append('Temporal\nAverage')
        
        bars = ax2.bar(fusion_types, self.df['top1'], color=academic_colors[:len(fusion_types)])
        ax2.set_ylabel('Top-1 Accuracy (%)')
        ax2.set_title('Fusion Architecture Performance')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, self.df['top1']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Efficiency Frontier Analysis
        ax3 = axes[1, 0]
        
        # Plot Pareto frontier for accuracy vs efficiency
        ax3.scatter(self.df['avg_latency_ms'], self.df['top1'], 
                   s=self.df['params_M']*5, alpha=0.7, c=self.df.index, cmap='viridis')
        
        # Annotate points
        for i, model in enumerate(self.df['model_type']):
            ax3.annotate(f"{model}\n({self.df['params_M'].iloc[i]:.1f}M)", 
                        (self.df['avg_latency_ms'].iloc[i], self.df['top1'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Average Latency (ms)')
        ax3.set_ylabel('Top-1 Accuracy (%)')
        ax3.set_title('Accuracy vs Latency Trade-off\n(Bubble size = Parameters)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Resource Utilization Heatmap
        ax4 = axes[1, 1]
        
        # Create normalized heatmap data
        metrics_data = []
        metric_names = ['Accuracy', 'Parameters', 'Memory', 'Latency']
        
        for _, row in self.df.iterrows():
            metrics_data.append([
                row['top1'],
                row['params_M'],
                row['peak_vram_GB'],
                row['avg_latency_ms']
            ])
        
        # Normalize each metric
        metrics_array = np.array(metrics_data)
        for i in range(metrics_array.shape[1]):
            col = metrics_array[:, i]
            if i == 0:  # Accuracy - higher is better
                metrics_array[:, i] = (col - col.min()) / (col.max() - col.min())
            else:  # Others - lower is better
                metrics_array[:, i] = 1 - ((col - col.min()) / (col.max() - col.min()))
        
        im = ax4.imshow(metrics_array.T, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax4.set_xticks(range(len(self.df)))
        ax4.set_xticklabels([f"{model}\n({lora})" for model, lora in 
                            zip(self.df['model_type'], 
                                ['LoRA' if x else 'Full' for x in self.df['lora_enabled']])], 
                           rotation=45)
        ax4.set_yticks(range(len(metric_names)))
        ax4.set_yticklabels(metric_names)
        ax4.set_title('Normalized Resource Utilization\n(Green=Better, Red=Worse)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Normalized Performance')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architectural_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'architectural_analysis.pdf', bbox_inches='tight')
        plt.show()

    def plot_efficiency_analysis(self):
        """Generate detailed efficiency analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('OmniQ Framework: Efficiency Analysis', fontsize=18, fontweight='bold')
        
        # 1. Parameter Efficiency Ranking
        ax1 = axes[0, 0]
        
        param_eff_sorted = self.df.sort_values('param_efficiency', ascending=True)
        bars = ax1.barh(range(len(param_eff_sorted)), param_eff_sorted['param_efficiency'],
                       color=academic_colors[:len(param_eff_sorted)])
        
        ax1.set_yticks(range(len(param_eff_sorted)))
        ax1.set_yticklabels(param_eff_sorted['model_type'])
        ax1.set_xlabel('Parameter Efficiency (Accuracy/Million Parameters)')
        ax1.set_title('Parameter Efficiency Ranking')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, param_eff_sorted['param_efficiency'])):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{val:.2f}', ha='left', va='center', fontweight='bold')
        
        # 2. Memory Efficiency Analysis
        ax2 = axes[0, 1]
        
        mem_eff_sorted = self.df.sort_values('memory_efficiency', ascending=True)
        bars2 = ax2.barh(range(len(mem_eff_sorted)), mem_eff_sorted['memory_efficiency'],
                        color=academic_colors[:len(mem_eff_sorted)])
        
        ax2.set_yticks(range(len(mem_eff_sorted)))
        ax2.set_yticklabels(mem_eff_sorted['model_type'])
        ax2.set_xlabel('Memory Efficiency (Accuracy/GB VRAM)')
        ax2.set_title('Memory Efficiency Ranking')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, mem_eff_sorted['memory_efficiency'])):
            ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}', ha='left', va='center', fontweight='bold')
        
        # 3. Speed vs Accuracy Trade-off
        ax3 = axes[1, 0]
        
        # Create speed score (inverse of latency)
        speed_score = 1000 / self.df['avg_latency_ms']  # FPS equivalent
        
        scatter = ax3.scatter(speed_score, self.df['top1'], 
                             s=200, alpha=0.7, c=self.df.index, cmap='viridis')
        
        # Add model labels
        for i, model in enumerate(self.df['model_type']):
            ax3.annotate(model, (speed_score.iloc[i], self.df['top1'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel('Processing Speed (FPS equivalent)')
        ax3.set_ylabel('Top-1 Accuracy (%)')
        ax3.set_title('Speed vs Accuracy Trade-off')
        ax3.grid(True, alpha=0.3)
        
        # 4. Overall Efficiency Score
        ax4 = axes[1, 1]
        
        # Calculate comprehensive efficiency score
        # Normalize all metrics and combine
        norm_acc = self.df['top1'] / 100
        norm_param_eff = (self.df['param_efficiency'] - self.df['param_efficiency'].min()) / \
                        (self.df['param_efficiency'].max() - self.df['param_efficiency'].min())
        norm_mem_eff = (self.df['memory_efficiency'] - self.df['memory_efficiency'].min()) / \
                      (self.df['memory_efficiency'].max() - self.df['memory_efficiency'].min())
        norm_speed = (speed_score - speed_score.min()) / (speed_score.max() - speed_score.min())
        
        # Weighted combination (you can adjust weights based on importance)
        overall_efficiency = (0.4 * norm_acc + 0.2 * norm_param_eff + 
                             0.2 * norm_mem_eff + 0.2 * norm_speed)
        
        colors = academic_colors[:len(self.df)]
        bars = ax4.bar(self.df['model_type'], overall_efficiency, color=colors, alpha=0.8)
        
        ax4.set_ylabel('Overall Efficiency Score')
        ax4.set_title('Comprehensive Efficiency Ranking')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels and ranking
        for i, (bar, val) in enumerate(zip(bars, overall_efficiency)):
            ax4.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                    f'{val:.3f}\n(#{i+1})', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'efficiency_analysis.pdf', bbox_inches='tight')
        plt.show()

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        report_path = self.output_dir / 'research_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OmniQ FRAMEWORK - TECHNICAL RESEARCH SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXPERIMENTAL OVERVIEW:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Models Evaluated: {len(self.df)}\n")
            f.write(f"Dataset: {self.df['dataset'].iloc[0]}\n")
            f.write(f"Input Configuration: {self.df['frames'].iloc[0]} frames, {self.df['size'].iloc[0]}x{self.df['size'].iloc[0]} resolution\n\n")
            
            f.write("PERFORMANCE RANKINGS:\n")
            f.write("-"*40 + "\n")
            
            # Top-1 Accuracy ranking
            acc_ranking = self.df.sort_values('top1', ascending=False)
            f.write("Top-1 Accuracy Ranking:\n")
            for i, (_, row) in enumerate(acc_ranking.iterrows(), 1):
                f.write(f"  {i}. {row['model_type']}: {row['top1']:.2f}%\n")
            
            f.write("\nParameter Efficiency Ranking:\n")
            param_ranking = self.df.sort_values('param_efficiency', ascending=False)
            for i, (_, row) in enumerate(param_ranking.iterrows(), 1):
                f.write(f"  {i}. {row['model_type']}: {row['param_efficiency']:.3f} acc/M-params\n")
            
            f.write("\nMemory Efficiency Ranking:\n")
            mem_ranking = self.df.sort_values('memory_efficiency', ascending=False)
            for i, (_, row) in enumerate(mem_ranking.iterrows(), 1):
                f.write(f"  {i}. {row['model_type']}: {row['memory_efficiency']:.2f} acc/GB\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("="*80 + "\n")
            
            best_acc = self.df.loc[self.df['top1'].idxmax()]
            best_param_eff = self.df.loc[self.df['param_efficiency'].idxmax()]
            best_mem_eff = self.df.loc[self.df['memory_efficiency'].idxmax()]
            
            f.write(f"• Best Overall Accuracy: {best_acc['model_type']} ({best_acc['top1']:.2f}%)\n")
            f.write(f"• Most Parameter Efficient: {best_param_eff['model_type']} ({best_param_eff['param_efficiency']:.3f})\n")
            f.write(f"• Most Memory Efficient: {best_mem_eff['model_type']} ({best_mem_eff['memory_efficiency']:.2f})\n")
            
            f.write(f"\n• LoRA Impact: {self.df['lora_enabled'].sum()}/{len(self.df)} models use LoRA\n")
            f.write(f"• Average Parameter Count: {self.df['params_M'].mean():.1f}M parameters\n")
            f.write(f"• Average Memory Usage: {self.df['peak_vram_GB'].mean():.2f}GB VRAM\n")
            f.write(f"• Average Inference Speed: {self.df['avg_latency_ms'].mean():.1f}ms per sample\n")
            
        print(f"Summary report saved to: {report_path}")

    def generate_all_plots(self):
        """Generate all research plots and summary."""
        print("Generating OmniQ Research Directive Plots...")
        print("="*50)
        
        print("1. Performance Comparison Analysis...")
        self.plot_performance_comparison()
        
        print("2. Architectural Analysis...")
        self.plot_architectural_analysis()
        
        print("3. Efficiency Analysis...")
        self.plot_efficiency_analysis()
        
        print("4. Generating Summary Report...")
        self.generate_summary_report()
        
        print(f"\nAll plots and reports saved to: {self.output_dir}")
        print("Files generated:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")

def main():
    """Main function to run the research plotting pipeline."""
    try:
        plotter = OmniQResearchPlotter()
        plotter.generate_all_plots()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure results/summary.csv exists with experimental data.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
