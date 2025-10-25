import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def analyze_resnet_results():
    """Analyze and visualize ResNet-50 results"""
    with open('results/resnet50_results.json', 'r') as f:
        results = json.load(f)
    
    baseline = results['baseline']
    optimized = results['optimized']
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training accuracy comparison
    axes[0, 0].plot(baseline['train_acc'], label='Baseline', linewidth=2)
    axes[0, 0].plot(optimized['train_acc'], label='Optimized', linewidth=2)
    axes[0, 0].set_title('Training Accuracy Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Test accuracy comparison
    axes[0, 1].plot(baseline['test_acc'], label='Baseline', linewidth=2)
    axes[0, 1].plot(optimized['test_acc'], label='Optimized', linewidth=2)
    axes[0, 1].set_title('Test Accuracy Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Epoch time comparison
    axes[1, 0].plot(baseline['epoch_times'], label='Baseline', linewidth=2)
    axes[1, 0].plot(optimized['epoch_times'], label='Optimized', linewidth=2)
    axes[1, 0].set_title('Epoch Time Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Memory usage comparison
    axes[1, 1].plot(baseline['memory_usage'], label='Baseline', linewidth=2)
    axes[1, 1].plot(optimized['memory_usage'], label='Optimized', linewidth=2)
    axes[1, 1].set_title('Memory Usage Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Memory (GB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/resnet50_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate efficiency improvements
    baseline_avg_time = sum(baseline['epoch_times']) / len(baseline['epoch_times'])
    optimized_avg_time = sum(optimized['epoch_times']) / len(optimized['epoch_times'])
    time_improvement = (baseline_avg_time - optimized_avg_time) / baseline_avg_time * 100
    
    baseline_avg_memory = sum(baseline['memory_usage']) / len(baseline['memory_usage'])
    optimized_avg_memory = sum(optimized['memory_usage']) / len(optimized['memory_usage'])
    memory_improvement = (baseline_avg_memory - optimized_avg_memory) / baseline_avg_memory * 100
    
    final_accuracy_baseline = baseline['test_acc'][-1]
    final_accuracy_optimized = optimized['test_acc'][-1]
    
    print(f"ResNet-50 Results:")
    print(f"Time Improvement: {time_improvement:.2f}%")
    print(f"Memory Improvement: {memory_improvement:.2f}%")
    print(f"Final Accuracy - Baseline: {final_accuracy_baseline:.2f}%")
    print(f"Final Accuracy - Optimized: {final_accuracy_optimized:.2f}%")
    
    return {
        'time_improvement': time_improvement,
        'memory_improvement': memory_improvement,
        'final_accuracy_baseline': final_accuracy_baseline,
        'final_accuracy_optimized': final_accuracy_optimized
    }

def analyze_gpt2_results():
    """Analyze and visualize GPT-2 results"""
    with open('results/gpt2_results.json', 'r') as f:
        results = json.load(f)
    
    baseline = results['baseline']
    optimized = results['optimized']
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    axes[0, 0].plot(baseline['train_loss'], label='Baseline', linewidth=2)
    axes[0, 0].plot(optimized['train_loss'], label='Optimized', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Perplexity comparison
    axes[0, 1].plot(baseline['perplexity'], label='Baseline', linewidth=2)
    axes[0, 1].plot(optimized['perplexity'], label='Optimized', linewidth=2)
    axes[0, 1].set_title('Perplexity Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Epoch time comparison
    axes[1, 0].plot(baseline['epoch_times'], label='Baseline', linewidth=2)
    axes[1, 0].plot(optimized['epoch_times'], label='Optimized', linewidth=2)
    axes[1, 0].set_title('Epoch Time Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Memory usage comparison
    axes[1, 1].plot(baseline['memory_usage'], label='Baseline', linewidth=2)
    axes[1, 1].plot(optimized['memory_usage'], label='Optimized', linewidth=2)
    axes[1, 1].set_title('Memory Usage Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Memory (GB)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/gpt2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate efficiency improvements
    baseline_avg_time = sum(baseline['epoch_times']) / len(baseline['epoch_times'])
    optimized_avg_time = sum(optimized['epoch_times']) / len(optimized['epoch_times'])
    time_improvement = (baseline_avg_time - optimized_avg_time) / baseline_avg_time * 100
    
    baseline_avg_memory = sum(baseline['memory_usage']) / len(baseline['memory_usage'])
    optimized_avg_memory = sum(optimized['memory_usage']) / len(optimized['memory_usage'])
    memory_improvement = (baseline_avg_memory - optimized_avg_memory) / baseline_avg_memory * 100
    
    final_perplexity_baseline = baseline['perplexity'][-1]
    final_perplexity_optimized = optimized['perplexity'][-1]
    
    print(f"GPT-2 Results:")
    print(f"Time Improvement: {time_improvement:.2f}%")
    print(f"Memory Improvement: {memory_improvement:.2f}%")
    print(f"Final Perplexity - Baseline: {final_perplexity_baseline:.2f}")
    print(f"Final Perplexity - Optimized: {final_perplexity_optimized:.2f}")
    
    return {
        'time_improvement': time_improvement,
        'memory_improvement': memory_improvement,
        'final_perplexity_baseline': final_perplexity_baseline,
        'final_perplexity_optimized': final_perplexity_optimized
    }

if __name__ == "__main__":
    # Create plots directory
    Path("results/plots").mkdir(exist_ok=True)
    
    print("Analyzing ResNet-50 Results...")
    resnet_analysis = analyze_resnet_results()
    
    print("\nAnalyzing GPT-2 Results...")
    gpt2_analysis = analyze_gpt2_results()
    
    # Create summary table
    summary_data = {
        'Model': ['ResNet-50', 'GPT-2'],
        'Time Improvement (%)': [resnet_analysis['time_improvement'], gpt2_analysis['time_improvement']],
        'Memory Improvement (%)': [resnet_analysis['memory_improvement'], gpt2_analysis['memory_improvement']],
        'Baseline Metric': [resnet_analysis['final_accuracy_baseline'], gpt2_analysis['final_perplexity_baseline']],
        'Optimized Metric': [resnet_analysis['final_accuracy_optimized'], gpt2_analysis['final_perplexity_optimized']]
    }
    
    df = pd.DataFrame(summary_data)
    df.to_csv('results/tables/summary_results.csv', index=False)
    print("\nSummary Results:")
    print(df)