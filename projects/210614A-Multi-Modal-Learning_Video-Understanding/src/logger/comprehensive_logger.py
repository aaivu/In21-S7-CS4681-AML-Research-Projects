import torch
import torch.distributed as dist
import numpy as np
from collections import defaultdict
import time
import psutil
import os
import json
from datetime import datetime


class ComprehensiveLogger:
    """
    Comprehensive logging for multi-scale EgoVLP training with TensorBoard support.
    
    Logs:
    1. Losses (per iteration): total_loss, egonce_loss, temporal_loss, lambda_weight
    2. Metrics (per epoch): EgoMCQ accuracy, fusion weights, temperature
    3. System stats (per epoch): GPU memory, training time, throughput
    4. Gradient stats (every 100 iterations): gradient norms, fusion weight gradients
    """
    
    def __init__(self, writer=None, logger=None, log_dir=None, rank=0):
        self.writer = writer
        self.logger = logger
        self.log_dir = log_dir
        self.rank = rank
        
        # Tracking variables
        self.iteration_count = 0
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # Metrics storage
        self.epoch_metrics = defaultdict(list)
        self.iteration_losses = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.system_stats = defaultdict(list)
        
        # GPU memory tracking
        self.gpu_memory_tracker = GPUMemoryTracker()
        
        # Create log directories
        if self.log_dir and self.rank == 0:
            os.makedirs(os.path.join(self.log_dir, 'detailed_logs'), exist_ok=True)
    
    def log_iteration_losses(self, epoch, batch_idx, dl_idx, losses_dict, 
                           model=None, log_gradients=False):
        """
        Log losses per iteration with optional gradient logging.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            dl_idx: Data loader index
            losses_dict: Dictionary containing loss values
            model: Model for gradient analysis
            log_gradients: Whether to log gradient statistics
        """
        self.iteration_count += 1
        
        # Extract loss components
        total_loss = losses_dict.get('total_loss', 0.0)
        egonce_loss = losses_dict.get('egonce_loss', 0.0)
        egonce_v2t = losses_dict.get('egonce_v2t', 0.0)
        egonce_t2v = losses_dict.get('egonce_t2v', 0.0)
        temporal_loss = losses_dict.get('temporal_loss', 0.0)
        lambda_weight = losses_dict.get('lambda_weight', 0.0)
        current_temp = losses_dict.get('temperature', 0.05)
        
        # Store for analysis
        self.iteration_losses['total_loss'].append(total_loss)
        self.iteration_losses['egonce_loss'].append(egonce_loss)
        self.iteration_losses['temporal_loss'].append(temporal_loss)
        
        # TensorBoard logging (per iteration)
        if self.writer and self.rank == 0:
            # Calculate global step
            global_step = self.iteration_count
            
            # Loss logging
            self.writer.add_scalar(f'Loss_iter/total_loss_dl{dl_idx}', total_loss, global_step)
            self.writer.add_scalar(f'Loss_iter/egonce_total_dl{dl_idx}', egonce_loss, global_step)
            self.writer.add_scalar(f'Loss_iter/egonce_v2t_dl{dl_idx}', egonce_v2t, global_step)
            self.writer.add_scalar(f'Loss_iter/egonce_t2v_dl{dl_idx}', egonce_t2v, global_step)
            self.writer.add_scalar(f'Loss_iter/temporal_loss_dl{dl_idx}', temporal_loss, global_step)
            self.writer.add_scalar(f'Loss_iter/lambda_weight_dl{dl_idx}', lambda_weight, global_step)
            self.writer.add_scalar(f'Training_iter/temperature_dl{dl_idx}', current_temp, global_step)
            
            # Fusion weights logging
            if model and hasattr(model, 'video_encoder'):
                if hasattr(model.video_encoder, 'get_fusion_weights'):
                    try:
                        fusion_weights = model.video_encoder.get_fusion_weights()
                        if fusion_weights is not None:
                            for i, weight in enumerate(fusion_weights):
                                self.writer.add_scalar(
                                    f'MultiScale_iter/fusion_weight_{i}_dl{dl_idx}', 
                                    weight, global_step
                                )
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(f"Could not log fusion weights: {e}")
        
        # Gradient logging (every 100 iterations)
        if log_gradients and self.iteration_count % 100 == 0 and model:
            self.log_gradient_stats(model, dl_idx)
        
        # Console logging
        if batch_idx % 10 == 0 and self.rank == 0 and self.logger:
            self.logger.info(
                f'Epoch {epoch} DL{dl_idx} Batch {batch_idx}: '
                f'Total: {total_loss:.4f} | EgoNCE: {egonce_loss:.4f} '
                f'(V2T: {egonce_v2t:.4f}, T2V: {egonce_t2v:.4f}) | '
                f'Temporal: {temporal_loss:.4f} | Temp: {current_temp:.4f}'
            )
    
    def log_gradient_stats(self, model, dl_idx):
        """Log gradient statistics every 100 iterations."""
        if not model or self.rank != 0:
            return
        
        try:
            # Calculate gradient norms
            total_grad_norm = 0.0
            max_grad_norm = 0.0
            fusion_grad_norms = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    max_grad_norm = max(max_grad_norm, param_norm.item())
                    
                    # Check for fusion weight gradients
                    if 'fusion' in name.lower() or 'attention_weights' in name.lower():
                        fusion_grad_norms.append(param_norm.item())
            
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            # Store statistics
            self.gradient_stats['total_norm'].append(total_grad_norm)
            self.gradient_stats['max_norm'].append(max_grad_norm)
            if fusion_grad_norms:
                self.gradient_stats['fusion_grad_mean'].append(np.mean(fusion_grad_norms))
            
            # TensorBoard logging
            if self.writer:
                global_step = self.iteration_count
                self.writer.add_scalar(f'Gradients/total_norm_dl{dl_idx}', total_grad_norm, global_step)
                self.writer.add_scalar(f'Gradients/max_norm_dl{dl_idx}', max_grad_norm, global_step)
                
                if fusion_grad_norms:
                    self.writer.add_scalar(
                        f'Gradients/fusion_weights_mean_dl{dl_idx}', 
                        np.mean(fusion_grad_norms), global_step
                    )
                    self.writer.add_scalar(
                        f'Gradients/fusion_weights_max_dl{dl_idx}', 
                        max(fusion_grad_norms), global_step
                    )
        
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error logging gradient stats: {e}")
    
    def start_epoch_timing(self):
        """Start timing for epoch."""
        self.epoch_start_time = time.time()
        self.gpu_memory_tracker.reset()
    
    def log_epoch_metrics(self, epoch, metrics_dict, model=None):
        """
        Log comprehensive epoch-level metrics.
        
        Args:
            epoch: Current epoch number
            metrics_dict: Dictionary containing epoch metrics
            model: Model for fusion weights extraction
        """
        if self.rank != 0:
            return
        
        # Calculate epoch timing
        epoch_time = 0
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
        
        # Extract metrics
        egomcq_intra = metrics_dict.get('egomcq_intra_accuracy', 0.0)
        egomcq_inter = metrics_dict.get('egomcq_inter_accuracy', 0.0)
        current_temp = metrics_dict.get('temperature', 0.05)
        samples_processed = metrics_dict.get('samples_processed', 0)
        
        # Calculate throughput
        samples_per_second = samples_processed / epoch_time if epoch_time > 0 else 0
        
        # GPU memory statistics
        gpu_stats = self.gpu_memory_tracker.get_stats()
        
        # Store epoch metrics
        self.epoch_metrics['egomcq_intra'].append(egomcq_intra)
        self.epoch_metrics['egomcq_inter'].append(egomcq_inter)
        self.epoch_metrics['epoch_time'].append(epoch_time)
        self.epoch_metrics['samples_per_second'].append(samples_per_second)
        
        # TensorBoard logging (per epoch)
        if self.writer:
            # EgoMCQ Accuracy Metrics
            self.writer.add_scalar('Metrics_epoch/egomcq_intra_accuracy', egomcq_intra, epoch)
            self.writer.add_scalar('Metrics_epoch/egomcq_inter_accuracy', egomcq_inter, epoch)
            self.writer.add_scalar('Metrics_epoch/egomcq_avg_accuracy', 
                                 (egomcq_intra + egomcq_inter) / 2, epoch)
            
            # Temperature
            self.writer.add_scalar('Training_epoch/temperature', current_temp, epoch)
            
            # Fusion weights
            if model and hasattr(model, 'video_encoder'):
                if hasattr(model.video_encoder, 'get_fusion_weights'):
                    try:
                        fusion_weights = model.video_encoder.get_fusion_weights()
                        if fusion_weights is not None:
                            # Log individual fusion weights
                            weight_names = ['w_fine', 'w_medium', 'w_coarse']
                            for i, (weight, name) in enumerate(zip(fusion_weights, weight_names)):
                                self.writer.add_scalar(f'MultiScale_epoch/{name}', weight, epoch)
                            
                            # Log fusion weight entropy (measure of distribution)
                            weights_np = np.array(fusion_weights)
                            weights_norm = weights_np / (weights_np.sum() + 1e-8)
                            entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
                            self.writer.add_scalar('MultiScale_epoch/fusion_entropy', entropy, epoch)
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(f"Could not log fusion weights: {e}")
            
            # System Statistics
            self.writer.add_scalar('System_epoch/epoch_time_seconds', epoch_time, epoch)
            self.writer.add_scalar('System_epoch/samples_per_second', samples_per_second, epoch)
            
            # GPU Memory Statistics
            for gpu_id, stats in gpu_stats.items():
                self.writer.add_scalar(f'System_epoch/gpu_{gpu_id}_memory_max_gb', 
                                     stats['max_memory_gb'], epoch)
                self.writer.add_scalar(f'System_epoch/gpu_{gpu_id}_memory_avg_gb', 
                                     stats['avg_memory_gb'], epoch)
            
            # Overall system stats
            if gpu_stats:
                max_memory_all = max(stats['max_memory_gb'] for stats in gpu_stats.values())
                avg_memory_all = np.mean([stats['avg_memory_gb'] for stats in gpu_stats.values()])
                self.writer.add_scalar('System_epoch/gpu_memory_max_all_gb', max_memory_all, epoch)
                self.writer.add_scalar('System_epoch/gpu_memory_avg_all_gb', avg_memory_all, epoch)
        
        # Console logging
        if self.logger:
            fusion_info = ""
            if model and hasattr(model, 'video_encoder'):
                if hasattr(model.video_encoder, 'get_fusion_weights'):
                    try:
                        fusion_weights = model.video_encoder.get_fusion_weights()
                        if fusion_weights is not None:
                            weights_str = ", ".join([f"{w:.3f}" for w in fusion_weights])
                            fusion_info = f" | Fusion weights: [{weights_str}]"
                    except:
                        pass
            
            gpu_info = ""
            if gpu_stats:
                max_mem = max(stats['max_memory_gb'] for stats in gpu_stats.values())
                gpu_info = f" | GPU mem: {max_mem:.1f}GB"
            
            self.logger.info(
                f'Epoch {epoch} Summary: '
                f'EgoMCQ Intra: {egomcq_intra:.2f}% | Inter: {egomcq_inter:.2f}% | '
                f'Temp: {current_temp:.4f} | Time: {epoch_time:.1f}s | '
                f'Throughput: {samples_per_second:.1f} samples/s'
                f'{fusion_info}{gpu_info}'
            )
    
    def save_detailed_logs(self, epoch):
        """Save detailed logs to JSON files."""
        if self.rank != 0 or not self.log_dir:
            return
        
        try:
            log_data = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'iteration_losses': dict(self.iteration_losses),
                'epoch_metrics': dict(self.epoch_metrics),
                'gradient_stats': dict(self.gradient_stats),
                'system_stats': dict(self.system_stats)
            }
            
            # Save to file
            log_file = os.path.join(self.log_dir, 'detailed_logs', f'epoch_{epoch:03d}.json')
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not save detailed logs: {e}")
    
    def get_summary_stats(self):
        """Get summary statistics for all logged metrics."""
        summary = {}
        
        # Loss summaries
        if self.iteration_losses['total_loss']:
            summary['avg_total_loss'] = np.mean(self.iteration_losses['total_loss'])
            summary['final_total_loss'] = self.iteration_losses['total_loss'][-1]
        
        # Epoch metric summaries
        if self.epoch_metrics['egomcq_intra']:
            summary['best_egomcq_intra'] = max(self.epoch_metrics['egomcq_intra'])
            summary['best_egomcq_inter'] = max(self.epoch_metrics['egomcq_inter'])
            summary['avg_epoch_time'] = np.mean(self.epoch_metrics['epoch_time'])
            summary['avg_throughput'] = np.mean(self.epoch_metrics['samples_per_second'])
        
        return summary


class GPUMemoryTracker:
    """Track GPU memory usage during training."""
    
    def __init__(self):
        self.memory_samples = defaultdict(list)
        self.max_memory = defaultdict(float)
    
    def reset(self):
        """Reset memory tracking for new epoch."""
        self.memory_samples.clear()
        self.max_memory.clear()
    
    def sample(self):
        """Sample current GPU memory usage."""
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                try:
                    memory_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    self.memory_samples[gpu_id].append(memory_gb)
                    self.max_memory[gpu_id] = max(self.max_memory[gpu_id], memory_gb)
                except:
                    pass
    
    def get_stats(self):
        """Get memory statistics for all GPUs."""
        stats = {}
        for gpu_id in self.memory_samples:
            samples = self.memory_samples[gpu_id]
            if samples:
                stats[gpu_id] = {
                    'max_memory_gb': self.max_memory[gpu_id],
                    'avg_memory_gb': np.mean(samples),
                    'current_memory_gb': samples[-1] if samples else 0
                }
        return stats


def create_comprehensive_logger(writer, logger, log_dir, rank=0):
    """Factory function to create comprehensive logger."""
    return ComprehensiveLogger(writer=writer, logger=logger, log_dir=log_dir, rank=rank)