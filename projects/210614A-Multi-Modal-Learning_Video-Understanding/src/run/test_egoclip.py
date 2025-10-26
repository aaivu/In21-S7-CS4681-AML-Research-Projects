import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import sys
sys.path.append('.')

from model import model as module_model
from model.metric import egomcq_accuracy_metrics
from parse_config import ConfigParser
from data_loader.EgoClip_EgoMCQ_dataset import EgoClipEgoMCQ
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime


def test_egomcq_multiscale(config, single_scale=False):
    """
    Test EgoMCQ benchmark with multi-scale video encoder support.
    
    Args:
        config: Configuration object
        single_scale: If True, use only single scale for efficiency comparison
    """
    # Setup data_loader instances
    dataset = EgoClipEgoMCQ(config['data_loader'])
    data_loader = DataLoader(
        dataset,
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        num_workers=config['data_loader']['args']['num_workers']
    )
    
    # Build model architecture
    model = config.init_obj('arch', module_model)
    
    # Load checkpoint
    checkpoint_path = config.resume
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load model weights
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(state_dict, strict=False)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Configure multi-scale vs single-scale mode
    if hasattr(model.video_encoder, 'use_multi_scale'):
        if single_scale:
            print("Running in single-scale mode for efficiency comparison")
            model.video_encoder.use_multi_scale = False
        else:
            print("Running in multi-scale mode")
            model.video_encoder.use_multi_scale = True
    
    # Initialize metric tracking
    all_preds = []
    all_labels = []
    all_types = []
    fusion_weights_log = []
    
    # Timing and memory tracking
    total_inference_time = 0
    memory_usage = []
    batch_times = []
    
    print("Starting EgoMCQ evaluation...")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader, desc="Processing batches")):
            batch_start_time = time.time()
            
            # Move data to device
            video = data['video'].to(device)
            text = data['text'].to(device)
            labels = data['answer'].to(device)
            types = data['type'].to(device)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            # Forward pass
            start_time = time.time()
            
            if hasattr(model, 'compute_loss'):
                # Use the model's compute_loss method for EgoMCQ
                loss_dict = model.compute_loss(data)
                logits = loss_dict.get('logits', None)
            else:
                # Direct forward pass
                outputs = model(video, text)
                logits = outputs.get('logits', None)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            if logits is None:
                print(f"Warning: No logits found in batch {batch_idx}")
                continue
            
            # Log fusion weights if available (for multi-scale analysis)
            if hasattr(model.video_encoder, 'last_fusion_weights') and not single_scale:
                fusion_weights = model.video_encoder.last_fusion_weights
                if fusion_weights is not None:
                    fusion_weights_log.append(fusion_weights.cpu().numpy())
            
            # Store predictions and labels
            predictions = torch.softmax(logits, dim=-1)
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_types.append(types.cpu())
            
            # Memory tracking
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append((peak_memory - initial_memory) / 1024**3)  # GB
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Progress reporting
            if (batch_idx + 1) % 10 == 0:
                avg_time = np.mean(batch_times[-10:])
                if torch.cuda.is_available():
                    avg_memory = np.mean(memory_usage[-10:])
                    print(f"Batch {batch_idx + 1}/{len(data_loader)}: "
                          f"Avg time: {avg_time:.2f}s, Avg memory: {avg_memory:.2f}GB")
                else:
                    print(f"Batch {batch_idx + 1}/{len(data_loader)}: "
                          f"Avg time: {avg_time:.2f}s")
    
    # Concatenate all results
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_types = torch.cat(all_types, dim=0)
    
    # Calculate EgoMCQ accuracy metrics
    metrics = egomcq_accuracy_metrics(all_preds, all_labels, all_types)
    
    # Add performance metrics
    performance_metrics = {
        'total_inference_time': total_inference_time,
        'average_batch_time': np.mean(batch_times),
        'samples_processed': len(all_preds),
        'samples_per_second': len(all_preds) / total_inference_time,
        'mode': 'single_scale' if single_scale else 'multi_scale'
    }
    
    if torch.cuda.is_available():
        performance_metrics.update({
            'average_memory_usage_gb': np.mean(memory_usage),
            'peak_memory_usage_gb': np.max(memory_usage),
            'gpu_name': torch.cuda.get_device_name()
        })
    
    # Fusion weights analysis (multi-scale only)
    if fusion_weights_log and not single_scale:
        fusion_analysis = analyze_fusion_weights(fusion_weights_log)
        performance_metrics['fusion_weights_analysis'] = fusion_analysis
    
    # Combine all metrics
    final_metrics = {
        'egomcq_accuracy': metrics,
        'performance': performance_metrics,
        'timestamp': datetime.datetime.now().isoformat(),
        'checkpoint': checkpoint_path,
        'dataset_size': len(dataset)
    }
    
    return final_metrics


def analyze_fusion_weights(fusion_weights_log):
    """
    Analyze fusion weights distribution across temporal scales.
    
    Args:
        fusion_weights_log: List of fusion weights from each batch
    
    Returns:
        Dictionary with fusion weights statistics
    """
    if not fusion_weights_log:
        return {}
    
    # Concatenate all fusion weights
    all_weights = np.concatenate(fusion_weights_log, axis=0)  # Shape: [N, num_scales]
    
    # Calculate statistics
    mean_weights = np.mean(all_weights, axis=0)
    std_weights = np.std(all_weights, axis=0)
    
    # Scale preferences (which scales are used most)
    scale_preferences = np.argsort(mean_weights)[::-1]  # Descending order
    
    analysis = {
        'num_scales': all_weights.shape[1],
        'mean_weights_per_scale': mean_weights.tolist(),
        'std_weights_per_scale': std_weights.tolist(),
        'scale_preference_order': scale_preferences.tolist(),
        'dominant_scale_weight': float(np.max(mean_weights)),
        'weight_distribution_entropy': float(-np.sum(mean_weights * np.log(mean_weights + 1e-8))),
        'total_samples_analyzed': len(all_weights)
    }
    
    return analysis


def save_results(results, output_path):
    """Save results to JSON file with pretty formatting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    print(f"Results saved to: {output_path}")


def print_results(results):
    """Print formatted results to console."""
    print("\n" + "="*60)
    print("EgoMCQ EVALUATION RESULTS")
    print("="*60)
    
    # EgoMCQ Accuracy
    accuracy_metrics = results['egomcq_accuracy']
    print("\nEgoMCQ Accuracy Metrics:")
    for key, value in accuracy_metrics.items():
        print(f"  {key}: {value:.2f}%")
    
    # Performance Metrics
    performance = results['performance']
    print(f"\nPerformance Metrics ({performance['mode'].upper()}):")
    print(f"  Total samples: {performance['samples_processed']}")
    print(f"  Total inference time: {performance['total_inference_time']:.2f}s")
    print(f"  Average batch time: {performance['average_batch_time']:.3f}s")
    print(f"  Samples per second: {performance['samples_per_second']:.1f}")
    
    if 'average_memory_usage_gb' in performance:
        print(f"  Average GPU memory: {performance['average_memory_usage_gb']:.2f}GB")
        print(f"  Peak GPU memory: {performance['peak_memory_usage_gb']:.2f}GB")
        print(f"  GPU: {performance['gpu_name']}")
    
    # Fusion weights analysis (if available)
    if 'fusion_weights_analysis' in performance:
        fusion_analysis = performance['fusion_weights_analysis']
        print(f"\nFusion Weights Analysis:")
        print(f"  Number of scales: {fusion_analysis['num_scales']}")
        print(f"  Scale preferences: {fusion_analysis['scale_preference_order']}")
        print(f"  Dominant scale weight: {fusion_analysis['dominant_scale_weight']:.3f}")
        print(f"  Weight entropy: {fusion_analysis['weight_distribution_entropy']:.3f}")
        
        print("  Mean weights per scale:")
        for i, weight in enumerate(fusion_analysis['mean_weights_per_scale']):
            print(f"    Scale {i}: {weight:.3f} ± {fusion_analysis['std_weights_per_scale'][i]:.3f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='EgoMCQ Evaluation with Multi-Scale Support')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-o', '--output', default=None, type=str,
                        help='output path for results JSON (default: auto-generated)')
    parser.add_argument('--single_scale', action='store_true',
                        help='use single scale for efficiency comparison')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default config if not provided
    if args.config is None:
        args.config = 'configs/eval/egomcq.json'
    
    # Initialize configuration
    config = ConfigParser.from_args(parser)
    
    # Set GPU device
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Run evaluation
    try:
        results = test_egomcq_multiscale(config, single_scale=args.single_scale)
        
        # Print results
        print_results(results)
        
        # Save results
        if args.output:
            output_path = args.output
        else:
            mode_suffix = "single_scale" if args.single_scale else "multi_scale"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/egomcq_evaluation_{mode_suffix}_{timestamp}.json"
        
        save_results(results, output_path)
        
        print(f"\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())