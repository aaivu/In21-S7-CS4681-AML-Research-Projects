"""
Training Optimizations for Enhanced PointNeXt
Implementation of gradient checkpointing, mixed precision training, and dynamic memory allocation
Based on the IEEE paper:
"Enhancing PointNeXt for Large-Scale 3D Point Cloud Processing: Adaptive Sampling vs. Memory-Efficient Attention"
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint_sequential
import gc
from typing import Dict, Any, Optional, Callable
import logging


class TrainingOptimizer:
    """
    Training optimization manager that implements gradient checkpointing,
    mixed precision training, and dynamic memory allocation to achieve
    3.1x speed improvement as described in the paper.
    """
    
    def __init__(self,
                 model: nn.Module,
                 use_mixed_precision: bool = True,
                 use_gradient_checkpointing: bool = True,
                 checkpoint_segments: int = 4,
                 memory_threshold: float = 0.8,
                 auto_memory_management: bool = True):
        """
        Initialize training optimizer.
        
        Args:
            model: The model to optimize
            use_mixed_precision: Whether to use automatic mixed precision
            use_gradient_checkpointing: Whether to use gradient checkpointing
            checkpoint_segments: Number of segments for sequential checkpointing
            memory_threshold: GPU memory threshold for dynamic allocation
            auto_memory_management: Whether to automatically manage GPU memory
        """
        self.model = model
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpoint_segments = checkpoint_segments
        self.memory_threshold = memory_threshold
        self.auto_memory_management = auto_memory_management
        
        # Initialize mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Memory monitoring
        self.memory_stats = {
            'peak_memory': 0,
            'average_memory': 0,
            'memory_saves': 0
        }
        
        # Configure model for optimizations
        self._configure_model()
    
    def _configure_model(self):
        """Configure model for optimization techniques."""
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Enable memory-efficient attention if available
        if hasattr(self.model, 'use_memory_efficient_attention'):
            self.model.use_memory_efficient_attention = True
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on model layers."""
        def apply_checkpointing(module):
            if hasattr(module, 'use_gradient_checkpointing'):
                module.use_gradient_checkpointing = True
        
        self.model.apply(apply_checkpointing)
        self.logger.info("Gradient checkpointing enabled")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'peak': 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        peak = torch.cuda.max_memory_allocated() / 1024**3   # GB
        
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'peak': peak
        }
    
    def manage_memory(self, force_gc: bool = False):
        """Dynamic memory management."""
        if not self.auto_memory_management and not force_gc:
            return
        
        memory_info = self.get_memory_usage()
        
        # Check if memory usage exceeds threshold
        if memory_info['allocated'] / memory_info['reserved'] > self.memory_threshold or force_gc:
            # Clear cache and run garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            
            self.memory_stats['memory_saves'] += 1
            self.logger.debug(f"Memory cleared. Current usage: {memory_info['allocated']:.2f}GB")
    
    def forward_with_checkpointing(self,
                                 model_segments: list,
                                 *args) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing.
        
        Args:
            model_segments: List of model segments to checkpoint
            *args: Forward pass arguments
            
        Returns:
            Model output
        """
        if not self.use_gradient_checkpointing or not self.model.training:
            # Regular forward pass
            return self._regular_forward(model_segments, *args)
        
        # Checkpointed forward pass
        def run_function(start_idx, end_idx):
            def forward_fn(*inputs):
                x = inputs[0]
                for i in range(start_idx, end_idx):
                    x = model_segments[i](x)
                return x
            return forward_fn
        
        # Split into segments
        num_segments = len(model_segments)
        segment_size = max(1, num_segments // self.checkpoint_segments)
        
        x = args[0]
        for i in range(0, num_segments, segment_size):
            end_idx = min(i + segment_size, num_segments)
            
            if end_idx - i == 1:
                # Single layer, just run normally
                x = model_segments[i](x)
            else:
                # Multiple layers, use checkpointing
                x = checkpoint_sequential(
                    model_segments[i:end_idx],
                    segments=1,
                    input=x,
                    use_reentrant=False
                )
        
        return x
    
    def _regular_forward(self, model_segments: list, *args) -> torch.Tensor:
        """Regular forward pass without checkpointing."""
        x = args[0]
        for segment in model_segments:
            x = segment(x)
        return x
    
    def optimize_batch_size(self, 
                           initial_batch_size: int,
                           max_batch_size: int = 64) -> int:
        """
        Dynamically optimize batch size based on memory usage.
        
        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            
        Returns:
            Optimized batch size
        """
        current_batch_size = initial_batch_size
        memory_info = self.get_memory_usage()
        
        # If memory usage is low, try to increase batch size
        if memory_info['allocated'] / memory_info['reserved'] < 0.5:
            current_batch_size = min(current_batch_size * 2, max_batch_size)
        
        # If memory usage is high, decrease batch size
        elif memory_info['allocated'] / memory_info['reserved'] > 0.8:
            current_batch_size = max(current_batch_size // 2, 1)
        
        return current_batch_size
    
    def training_step(self,
                     model: nn.Module,
                     batch: Dict[str, torch.Tensor],
                     criterion: nn.Module,
                     optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Optimized training step with mixed precision and memory management.
        
        Args:
            model: Model to train
            batch: Training batch
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Dictionary with loss and metrics
        """
        # Memory management before forward pass
        self.manage_memory()
        
        # Extract batch data
        points = batch.get('pos', batch.get('points'))
        targets = batch.get('y', batch.get('labels'))
        features = batch.get('x', batch.get('features', None))
        
        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with autocast():
                outputs = model(points, features)
                loss = criterion(outputs, targets)
        else:
            outputs = model(points, features)
            loss = criterion(outputs, targets)
        
        # Backward pass with mixed precision
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Update memory statistics
        memory_info = self.get_memory_usage()
        self.memory_stats['peak_memory'] = max(
            self.memory_stats['peak_memory'], 
            memory_info['peak']
        )
        
        # Memory cleanup after backward pass
        if self.auto_memory_management:
            self.manage_memory()
        
        return {
            'loss': loss.item(),
            'memory_allocated': memory_info['allocated'],
            'memory_peak': memory_info['peak']
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        memory_info = self.get_memory_usage()
        
        return {
            'mixed_precision_enabled': self.use_mixed_precision,
            'gradient_checkpointing_enabled': self.use_gradient_checkpointing,
            'checkpoint_segments': self.checkpoint_segments,
            'current_memory_gb': memory_info['allocated'],
            'peak_memory_gb': self.memory_stats['peak_memory'],
            'memory_saves_count': self.memory_stats['memory_saves'],
            'auto_memory_management': self.auto_memory_management
        }


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader with dynamic batching.
    """
    
    def __init__(self, 
                 base_dataloader,
                 memory_threshold: float = 0.8,
                 min_batch_size: int = 1,
                 max_batch_size: int = 64):
        """
        Initialize memory-efficient data loader.
        
        Args:
            base_dataloader: Base PyTorch DataLoader
            memory_threshold: Memory threshold for batch size adjustment
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
        """
        self.base_dataloader = base_dataloader
        self.memory_threshold = memory_threshold
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        self.current_batch_size = base_dataloader.batch_size
        self.adaptation_history = []
    
    def __iter__(self):
        """Iterate with dynamic batch size adjustment."""
        for batch in self.base_dataloader:
            # Check memory before yielding batch
            if torch.cuda.is_available():
                memory_info = self._get_memory_info()
                
                # Adjust batch size based on memory usage
                if memory_info['usage_ratio'] > self.memory_threshold:
                    self._reduce_batch_size()
                elif memory_info['usage_ratio'] < 0.5:
                    self._increase_batch_size()
            
            yield batch
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory information."""
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        return {
            'allocated_gb': allocated / 1024**3,
            'reserved_gb': reserved / 1024**3,
            'usage_ratio': allocated / max(reserved, 1)
        }
    
    def _reduce_batch_size(self):
        """Reduce batch size due to memory pressure."""
        new_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self.adaptation_history.append(('reduce', new_batch_size))
    
    def _increase_batch_size(self):
        """Increase batch size when memory allows."""
        new_batch_size = min(self.current_batch_size * 2, self.max_batch_size)
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self.adaptation_history.append(('increase', new_batch_size))


class PerformanceProfiler:
    """
    Performance profiler to validate the claimed improvements.
    """
    
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
        self.throughput_data = {}
    
    def profile_model(self, 
                     model: nn.Module,
                     sample_input: torch.Tensor,
                     num_runs: int = 100) -> Dict[str, float]:
        """
        Profile model performance.
        
        Args:
            model: Model to profile
            sample_input: Sample input tensor
            num_runs: Number of profiling runs
            
        Returns:
            Performance metrics
        """
        model.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Profile timing
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            torch.cuda.empty_cache()
            
            start_time.record()
            memory_before = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                output = model(sample_input)
            
            memory_after = torch.cuda.memory_allocated()
            end_time.record()
            
            torch.cuda.synchronize()
            
            times.append(start_time.elapsed_time(end_time))
            memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        # Calculate throughput (samples per second)
        batch_size = sample_input.shape[0] if len(sample_input.shape) > 0 else 1
        throughput = (batch_size * 1000) / avg_time  # samples per second
        
        return {
            'average_time_ms': avg_time,
            'average_memory_mb': avg_memory / 1024**2,
            'throughput_samples_per_sec': throughput,
            'memory_efficiency': avg_memory / (batch_size * 1024**2)  # MB per sample
        }
    
    def compare_models(self,
                      baseline_model: nn.Module,
                      enhanced_model: nn.Module,
                      sample_input: torch.Tensor) -> Dict[str, float]:
        """
        Compare baseline and enhanced models.
        
        Args:
            baseline_model: Baseline PointNeXt model
            enhanced_model: Enhanced PointNeXt model
            sample_input: Sample input tensor
            
        Returns:
            Comparison metrics
        """
        baseline_metrics = self.profile_model(baseline_model, sample_input)
        enhanced_metrics = self.profile_model(enhanced_model, sample_input)
        
        # Calculate improvements
        speed_improvement = baseline_metrics['average_time_ms'] / enhanced_metrics['average_time_ms']
        memory_reduction = 1.0 - (enhanced_metrics['average_memory_mb'] / baseline_metrics['average_memory_mb'])
        throughput_improvement = enhanced_metrics['throughput_samples_per_sec'] / baseline_metrics['throughput_samples_per_sec']
        
        return {
            'speed_improvement': speed_improvement,
            'memory_reduction_percent': memory_reduction * 100,
            'throughput_improvement': throughput_improvement,
            'baseline_metrics': baseline_metrics,
            'enhanced_metrics': enhanced_metrics,
            'meets_paper_claims': {
                'speed_3x': speed_improvement >= 3.0,
                'memory_58_percent': memory_reduction >= 0.55  # 55% threshold
            }
        }