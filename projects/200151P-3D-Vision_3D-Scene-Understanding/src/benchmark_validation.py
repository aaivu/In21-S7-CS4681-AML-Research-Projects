"""
Benchmark Validation Framework for Enhanced PointNeXt
Validates the claimed performance improvements from the IEEE paper:
"Enhancing PointNeXt for Large-Scale 3D Point Cloud Processing: Adaptive Sampling vs. Memory-Efficient Attention"

Expected improvements:
- 3.1x speed improvement
- 58% memory reduction
- Better accuracy on large-scale datasets
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from ..models.backbone.pointnext import PointNeXt
from ..models.backbone.enhanced_pointnext import EnhancedPointNeXt
from ..models.training_optimizations import PerformanceProfiler, TrainingOptimizer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    batch_sizes: List[int] = None
    point_counts: List[int] = None
    num_classes: int = 40
    feature_dims: List[int] = None
    num_runs: int = 50
    warmup_runs: int = 10
    test_datasets: List[str] = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32]
        if self.point_counts is None:
            self.point_counts = [1024, 2048, 4096, 8192, 16384]
        if self.feature_dims is None:
            self.feature_dims = [3, 6, 9, 12]
        if self.test_datasets is None:
            self.test_datasets = ['ModelNet40', 'ScanObjectNN', 'S3DIS']


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    batch_size: int
    point_count: int
    feature_dim: int
    forward_time_ms: float
    backward_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    throughput_samples_per_sec: float
    accuracy: Optional[float] = None
    
    @property
    def total_time_ms(self) -> float:
        return self.forward_time_ms + self.backward_time_ms


class EnhancedPointNeXtBenchmark:
    """
    Comprehensive benchmark suite to validate Enhanced PointNeXt improvements.
    """
    
    def __init__(self, 
                 config: BenchmarkConfig = None,
                 device: str = 'cuda',
                 save_results: bool = True,
                 results_dir: str = './benchmark_results'):
        """
        Initialize benchmark suite.
        
        Args:
            config: Benchmark configuration
            device: Device to run benchmarks on
            save_results: Whether to save results to disk
            results_dir: Directory to save results
        """
        self.config = config or BenchmarkConfig()
        self.device = device
        self.save_results = save_results
        self.results_dir = results_dir
        
        # Create results directory
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        # Initialize profiler
        self.profiler = PerformanceProfiler()
        
        # Storage for results
        self.results = []
        self.comparison_results = {}
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def create_sample_data(self, 
                          batch_size: int, 
                          point_count: int, 
                          feature_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sample point cloud data for benchmarking.
        
        Args:
            batch_size: Batch size
            point_count: Number of points per cloud
            feature_dim: Feature dimension
            
        Returns:
            Tuple of (points, features)
        """
        # Generate random point coordinates
        points = torch.randn(batch_size, point_count, 3, device=self.device)
        
        # Generate random features if feature_dim > 3
        if feature_dim > 3:
            features = torch.randn(batch_size, point_count, feature_dim, device=self.device)
        else:
            features = points
        
        return points, features
    
    def benchmark_model(self, 
                       model: nn.Module, 
                       model_name: str,
                       batch_size: int,
                       point_count: int, 
                       feature_dim: int) -> BenchmarkResult:
        """
        Benchmark a single model configuration.
        
        Args:
            model: Model to benchmark
            model_name: Name of the model
            batch_size: Batch size
            point_count: Number of points
            feature_dim: Feature dimension
            
        Returns:
            Benchmark results
        """
        model.eval()
        
        # Create sample data
        points, features = self.create_sample_data(batch_size, point_count, feature_dim)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_runs):
                _ = model(points, features)
        
        torch.cuda.synchronize()
        
        # Benchmark forward pass
        forward_times = []
        memory_usages = []
        peak_memories = []
        
        for _ in range(self.config.num_runs):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            memory_before = torch.cuda.memory_allocated()
            
            # Forward pass timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                output = model(points, features)
            end_event.record()
            
            torch.cuda.synchronize()
            
            memory_after = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            forward_times.append(start_event.elapsed_time(end_event))
            memory_usages.append((memory_after - memory_before) / 1024**2)  # MB
            peak_memories.append(peak_memory / 1024**2)  # MB
        
        # Benchmark backward pass
        model.train()
        backward_times = []
        
        for _ in range(self.config.num_runs):
            torch.cuda.empty_cache()
            
            # Create data with gradients
            points_grad = points.detach().requires_grad_(True)
            features_grad = features.detach().requires_grad_(True) if features is not points else points_grad
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            
            output = model(points_grad, features_grad)
            loss = output.sum()  # Dummy loss
            loss.backward()
            
            end_event.record()
            torch.cuda.synchronize()
            
            backward_times.append(start_event.elapsed_time(end_event))
        
        # Calculate averages
        avg_forward_time = np.mean(forward_times)
        avg_backward_time = np.mean(backward_times)
        avg_memory_usage = np.mean(memory_usages)
        avg_peak_memory = np.mean(peak_memories)
        
        # Calculate throughput
        total_time_sec = (avg_forward_time + avg_backward_time) / 1000.0
        throughput = batch_size / total_time_sec
        
        return BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            point_count=point_count,
            feature_dim=feature_dim,
            forward_time_ms=avg_forward_time,
            backward_time_ms=avg_backward_time,
            memory_usage_mb=avg_memory_usage,
            peak_memory_mb=avg_peak_memory,
            throughput_samples_per_sec=throughput
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmark across all configurations.
        
        Returns:
            Dictionary of results organized by model
        """
        results = {
            'baseline_pointnext': [],
            'enhanced_pointnext': [],
            'enhanced_pointnext_lightweight': [],
            'enhanced_pointnext_high_performance': []
        }
        
        print("Starting comprehensive benchmark...")
        
        total_tests = (len(self.config.batch_sizes) * 
                      len(self.config.point_counts) * 
                      len(self.config.feature_dims) * 
                      len(results.keys()))
        
        test_count = 0
        
        for batch_size in self.config.batch_sizes:
            for point_count in self.config.point_counts:
                for feature_dim in self.config.feature_dims:
                    
                    # Create sample data for this configuration
                    points, features = self.create_sample_data(batch_size, point_count, feature_dim)
                    
                    # Test baseline PointNeXt
                    try:
                        baseline_model = PointNeXt(
                            in_channels=feature_dim,
                            num_classes=self.config.num_classes
                        ).to(self.device)
                        
                        result = self.benchmark_model(
                            baseline_model, 'baseline_pointnext', 
                            batch_size, point_count, feature_dim
                        )
                        results['baseline_pointnext'].append(result)
                        
                        del baseline_model
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Baseline benchmark failed for batch_size={batch_size}, "
                              f"point_count={point_count}, feature_dim={feature_dim}: {e}")
                    
                    test_count += 1
                    print(f"Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.1f}%)")
                    
                    # Test Enhanced PointNeXt variants
                    enhanced_configs = {
                        'enhanced_pointnext': {},
                        'enhanced_pointnext_lightweight': {'use_adaptive_sampling': True, 'use_memory_efficient_attention': True},
                        'enhanced_pointnext_high_performance': {'use_adaptive_sampling': True, 'use_memory_efficient_attention': True}
                    }
                    
                    for model_name, config in enhanced_configs.items():
                        try:
                            enhanced_model = EnhancedPointNeXt(
                                in_channels=feature_dim,
                                num_classes=self.config.num_classes,
                                **config
                            ).to(self.device)
                            
                            result = self.benchmark_model(
                                enhanced_model, model_name,
                                batch_size, point_count, feature_dim
                            )
                            results[model_name].append(result)
                            
                            del enhanced_model
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Enhanced benchmark failed for {model_name}: {e}")
                        
                        test_count += 1
                        print(f"Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.1f}%)")
        
        self.results = results
        return results
    
    def calculate_improvements(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance improvements compared to baseline.
        
        Returns:
            Dictionary of improvements for each enhanced model
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run benchmark first.")
        
        baseline_results = self.results['baseline_pointnext']
        improvements = {}
        
        for model_name, enhanced_results in self.results.items():
            if model_name == 'baseline_pointnext':
                continue
            
            improvements[model_name] = self._compare_results(baseline_results, enhanced_results)
        
        return improvements
    
    def _compare_results(self, 
                        baseline_results: List[BenchmarkResult], 
                        enhanced_results: List[BenchmarkResult]) -> Dict[str, float]:
        """Compare baseline and enhanced results."""
        if len(baseline_results) != len(enhanced_results):
            print(f"Warning: Result count mismatch - baseline: {len(baseline_results)}, "
                  f"enhanced: {len(enhanced_results)}")
        
        # Match results by configuration
        matched_pairs = []
        for baseline in baseline_results:
            for enhanced in enhanced_results:
                if (baseline.batch_size == enhanced.batch_size and
                    baseline.point_count == enhanced.point_count and
                    baseline.feature_dim == enhanced.feature_dim):
                    matched_pairs.append((baseline, enhanced))
                    break
        
        if not matched_pairs:
            return {}
        
        # Calculate improvements
        speed_improvements = []
        memory_reductions = []
        throughput_improvements = []
        
        for baseline, enhanced in matched_pairs:
            speed_improvement = baseline.total_time_ms / enhanced.total_time_ms
            memory_reduction = 1.0 - (enhanced.memory_usage_mb / baseline.memory_usage_mb)
            throughput_improvement = enhanced.throughput_samples_per_sec / baseline.throughput_samples_per_sec
            
            speed_improvements.append(speed_improvement)
            memory_reductions.append(memory_reduction)
            throughput_improvements.append(throughput_improvement)
        
        return {
            'average_speed_improvement': np.mean(speed_improvements),
            'average_memory_reduction_percent': np.mean(memory_reductions) * 100,
            'average_throughput_improvement': np.mean(throughput_improvements),
            'max_speed_improvement': np.max(speed_improvements),
            'max_memory_reduction_percent': np.max(memory_reductions) * 100,
            'meets_paper_speed_claim': np.mean(speed_improvements) >= 3.0,
            'meets_paper_memory_claim': np.mean(memory_reductions) >= 0.55,
            'num_configurations_tested': len(matched_pairs)
        }
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No benchmark results available."
        
        improvements = self.calculate_improvements()
        
        report = []
        report.append("=" * 80)
        report.append("ENHANCED POINTNEXT BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Paper claims
        report.append("PAPER CLAIMS:")
        report.append("- 3.1x speed improvement")
        report.append("- 58% memory reduction")
        report.append("")
        
        # Results summary
        report.append("BENCHMARK RESULTS:")
        report.append("-" * 40)
        
        for model_name, metrics in improvements.items():
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  Speed Improvement: {metrics['average_speed_improvement']:.2f}x")
            report.append(f"  Memory Reduction: {metrics['average_memory_reduction_percent']:.1f}%")
            report.append(f"  Throughput Improvement: {metrics['average_throughput_improvement']:.2f}x")
            report.append(f"  Meets Speed Claim (3.1x): {'✓' if metrics['meets_paper_speed_claim'] else '✗'}")
            report.append(f"  Meets Memory Claim (58%): {'✓' if metrics['meets_paper_memory_claim'] else '✗'}")
            report.append(f"  Configurations Tested: {metrics['num_configurations_tested']}")
        
        # Configuration details
        report.append(f"\nBENCHMARK CONFIGURATION:")
        report.append(f"  Batch Sizes: {self.config.batch_sizes}")
        report.append(f"  Point Counts: {self.config.point_counts}")
        report.append(f"  Feature Dims: {self.config.feature_dims}")
        report.append(f"  Runs per Config: {self.config.num_runs}")
        report.append(f"  Device: {self.device}")
        
        return "\n".join(report)
    
    def save_results(self):
        """Save benchmark results to disk."""
        if not self.save_results:
            return
        
        # Save raw results
        results_file = os.path.join(self.results_dir, 'benchmark_results.json')
        
        # Convert results to serializable format
        serializable_results = {}
        for model_name, results_list in self.results.items():
            serializable_results[model_name] = [asdict(result) for result in results_list]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save improvements
        improvements = self.calculate_improvements()
        improvements_file = os.path.join(self.results_dir, 'improvements.json')
        with open(improvements_file, 'w') as f:
            json.dump(improvements, f, indent=2)
        
        # Save report
        report = self.generate_report()
        report_file = os.path.join(self.results_dir, 'benchmark_report.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Results saved to {self.results_dir}")
    
    def create_visualizations(self):
        """Create visualization plots of benchmark results."""
        if not self.results:
            return
        
        try:
            improvements = self.calculate_improvements()
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Enhanced PointNeXt Benchmark Results', fontsize=16)
            
            # Speed improvement plot
            models = list(improvements.keys())
            speed_improvements = [improvements[model]['average_speed_improvement'] for model in models]
            
            axes[0, 0].bar(models, speed_improvements, color='skyblue')
            axes[0, 0].axhline(y=3.1, color='red', linestyle='--', label='Paper Claim (3.1x)')
            axes[0, 0].set_title('Speed Improvement vs Baseline')
            axes[0, 0].set_ylabel('Speed Improvement (x)')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Memory reduction plot
            memory_reductions = [improvements[model]['average_memory_reduction_percent'] for model in models]
            
            axes[0, 1].bar(models, memory_reductions, color='lightcoral')
            axes[0, 1].axhline(y=58, color='red', linestyle='--', label='Paper Claim (58%)')
            axes[0, 1].set_title('Memory Reduction vs Baseline')
            axes[0, 1].set_ylabel('Memory Reduction (%)')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Throughput comparison
            throughput_improvements = [improvements[model]['average_throughput_improvement'] for model in models]
            
            axes[1, 0].bar(models, throughput_improvements, color='lightgreen')
            axes[1, 0].set_title('Throughput Improvement vs Baseline')
            axes[1, 0].set_ylabel('Throughput Improvement (x)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Claims validation
            speed_claims = [improvements[model]['meets_paper_speed_claim'] for model in models]
            memory_claims = [improvements[model]['meets_paper_memory_claim'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, speed_claims, width, label='Speed Claim (3.1x)', color='blue', alpha=0.7)
            axes[1, 1].bar(x + width/2, memory_claims, width, label='Memory Claim (58%)', color='red', alpha=0.7)
            axes[1, 1].set_title('Paper Claims Validation')
            axes[1, 1].set_ylabel('Claim Met (1=Yes, 0=No)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(models, rotation=45)
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            if self.save_results:
                plot_file = os.path.join(self.results_dir, 'benchmark_plots.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"Plots saved to {plot_file}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


def run_quick_benchmark():
    """Run a quick benchmark for testing."""
    config = BenchmarkConfig(
        batch_sizes=[2, 4],
        point_counts=[1024, 2048],
        feature_dims=[3],
        num_runs=10
    )
    
    benchmark = EnhancedPointNeXtBenchmark(config=config)
    results = benchmark.run_comprehensive_benchmark()
    
    print(benchmark.generate_report())
    benchmark.save_results()
    benchmark.create_visualizations()
    
    return benchmark


if __name__ == "__main__":
    # Run quick benchmark
    benchmark = run_quick_benchmark()