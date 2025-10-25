#!/usr/bin/env python3
"""
Comprehensive Logging Demo for EgoVLP Multi-Scale Training

This script demonstrates the enhanced logging capabilities including:
1. Per-iteration loss tracking (total, EgoNCE v2t/t2v, temporal, lambda scheduling)
2. Per-epoch metrics (EgoMCQ accuracy, fusion weights, temperature)
3. System statistics (GPU memory, throughput, training time)
4. Gradient analysis (norms, fusion weight gradients every 100 iterations)

Usage:
    python logging_demo.py -c configs/pt/egoclip_rtx3090_optimized.json -r checkpoint.pth
    
    # View TensorBoard logs
    tensorboard --logdir saved/EgoClip_MultiScale_RTX3090/tensorboard
    
    # View detailed JSON logs
    ls saved/EgoClip_MultiScale_RTX3090/detailed_logs/
"""

import argparse
import json
import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np

def parse_tensorboard_logs(log_dir):
    """Parse TensorBoard event files to extract key metrics."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # Get available tags
        scalar_tags = event_acc.Tags()['scalars']
        
        metrics = {}
        
        # Extract key metrics
        for tag in scalar_tags:
            try:
                scalar_events = event_acc.Scalars(tag)
                metrics[tag] = [(e.step, e.value) for e in scalar_events]
            except:
                continue
        
        return metrics
    
    except ImportError:
        print("TensorBoard not available for log parsing")
        return {}
    except Exception as e:
        print(f"Error parsing TensorBoard logs: {e}")
        return {}


def analyze_detailed_logs(detailed_log_dir):
    """Analyze detailed JSON logs for comprehensive insights."""
    if not os.path.exists(detailed_log_dir):
        print(f"Detailed logs directory not found: {detailed_log_dir}")
        return {}
    
    log_files = sorted([f for f in os.listdir(detailed_log_dir) if f.endswith('.json')])
    
    if not log_files:
        print("No detailed log files found")
        return {}
    
    analysis = {
        'epochs_logged': len(log_files),
        'loss_trends': {},
        'fusion_weights_evolution': {},
        'system_performance': {},
        'gradient_health': {}
    }
    
    all_epochs_data = []
    
    for log_file in log_files:
        try:
            with open(os.path.join(detailed_log_dir, log_file)) as f:
                epoch_data = json.load(f)
                all_epochs_data.append(epoch_data)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
    
    if not all_epochs_data:
        return analysis
    
    # Analyze loss trends
    total_losses = []
    egonce_losses = []
    temporal_losses = []
    
    for epoch_data in all_epochs_data:
        losses = epoch_data.get('iteration_losses', {})
        if losses.get('total_loss'):
            total_losses.extend(losses['total_loss'])
        if losses.get('egonce_loss'):
            egonce_losses.extend(losses['egonce_loss'])
        if losses.get('temporal_loss'):
            temporal_losses.extend(losses['temporal_loss'])
    
    if total_losses:
        analysis['loss_trends'] = {
            'total_loss_mean': np.mean(total_losses),
            'total_loss_std': np.std(total_losses),
            'total_loss_final': total_losses[-10:] if len(total_losses) >= 10 else total_losses,
            'egonce_loss_mean': np.mean(egonce_losses) if egonce_losses else 0,
            'temporal_loss_mean': np.mean(temporal_losses) if temporal_losses else 0
        }
    
    # Analyze system performance
    epoch_times = []
    throughputs = []
    
    for epoch_data in all_epochs_data:
        metrics = epoch_data.get('epoch_metrics', {})
        if metrics.get('epoch_time'):
            epoch_times.extend(metrics['epoch_time'])
        if metrics.get('samples_per_second'):
            throughputs.extend(metrics['samples_per_second'])
    
    if epoch_times:
        analysis['system_performance'] = {
            'avg_epoch_time': np.mean(epoch_times),
            'avg_throughput': np.mean(throughputs) if throughputs else 0,
            'total_epochs_analyzed': len(all_epochs_data)
        }
    
    return analysis


def generate_logging_report(log_dir):
    """Generate a comprehensive logging report."""
    print("="*80)
    print("EgoVLP COMPREHENSIVE LOGGING ANALYSIS")
    print("="*80)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log directory: {log_dir}")
    
    # Analyze TensorBoard logs
    tb_log_dir = os.path.join(log_dir, 'tensorboard')
    if os.path.exists(tb_log_dir):
        print(f"\n📊 TensorBoard Logs Found: {tb_log_dir}")
        tb_metrics = parse_tensorboard_logs(tb_log_dir)
        
        if tb_metrics:
            print(f"   Available metric categories: {len(tb_metrics)} scalar tags")
            
            # Categorize metrics
            loss_metrics = [tag for tag in tb_metrics.keys() if 'loss' in tag.lower()]
            multiscale_metrics = [tag for tag in tb_metrics.keys() if 'multiscale' in tag.lower() or 'fusion' in tag.lower()]
            system_metrics = [tag for tag in tb_metrics.keys() if 'system' in tag.lower() or 'gpu' in tag.lower()]
            gradient_metrics = [tag for tag in tb_metrics.keys() if 'gradient' in tag.lower()]
            
            print(f"   Loss metrics: {len(loss_metrics)}")
            print(f"   Multi-scale metrics: {len(multiscale_metrics)}")  
            print(f"   System metrics: {len(system_metrics)}")
            print(f"   Gradient metrics: {len(gradient_metrics)}")
        else:
            print("   No metrics extracted from TensorBoard logs")
    else:
        print(f"\n❌ TensorBoard logs not found at: {tb_log_dir}")
    
    # Analyze detailed JSON logs
    detailed_log_dir = os.path.join(log_dir, 'detailed_logs')
    if os.path.exists(detailed_log_dir):
        print(f"\n📋 Detailed JSON Logs Found: {detailed_log_dir}")
        analysis = analyze_detailed_logs(detailed_log_dir)
        
        if analysis.get('epochs_logged', 0) > 0:
            print(f"   Epochs logged: {analysis['epochs_logged']}")
            
            # Loss analysis
            if analysis.get('loss_trends'):
                trends = analysis['loss_trends']
                print(f"   Average total loss: {trends.get('total_loss_mean', 0):.4f} ± {trends.get('total_loss_std', 0):.4f}")
                print(f"   Average EgoNCE loss: {trends.get('egonce_loss_mean', 0):.4f}")
                print(f"   Average temporal loss: {trends.get('temporal_loss_mean', 0):.4f}")
            
            # Performance analysis
            if analysis.get('system_performance'):
                perf = analysis['system_performance']
                print(f"   Average epoch time: {perf.get('avg_epoch_time', 0):.1f} seconds")
                print(f"   Average throughput: {perf.get('avg_throughput', 0):.1f} samples/second")
        else:
            print("   No epoch data found in detailed logs")
    else:
        print(f"\n❌ Detailed logs not found at: {detailed_log_dir}")
    
    # Check for model checkpoints
    checkpoint_dir = os.path.join(log_dir, 'models')
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        print(f"\n💾 Model Checkpoints: {len(checkpoints)} found")
        if checkpoints:
            print(f"   Latest: {max(checkpoints)}")
    
    print("\n" + "="*80)
    print("LOGGING FEATURES VERIFIED")
    print("="*80)
    
    # Feature checklist
    features = [
        ("Per-iteration losses (total, EgoNCE, temporal)", "✓" if loss_metrics else "❌"),
        ("Temperature scheduling logs", "✓" if any('temp' in tag.lower() for tag in tb_metrics.keys()) else "❌"),
        ("Fusion weights tracking", "✓" if multiscale_metrics else "❌"),
        ("System statistics (GPU, timing)", "✓" if system_metrics else "❌"),
        ("Gradient analysis", "✓" if gradient_metrics else "❌"),
        ("EgoMCQ accuracy metrics", "✓" if any('egomcq' in tag.lower() for tag in tb_metrics.keys()) else "❌"),
        ("Detailed JSON exports", "✓" if analysis.get('epochs_logged', 0) > 0 else "❌")
    ]
    
    for feature, status in features:
        print(f"   {status} {feature}")
    
    print("\n" + "="*80)
    print("VIEWING INSTRUCTIONS")
    print("="*80)
    print(f"1. TensorBoard: tensorboard --logdir {tb_log_dir}")
    print("2. Key metric categories to view:")
    print("   - Loss_iter/: Per-iteration loss tracking")
    print("   - Metrics_epoch/: EgoMCQ accuracy per epoch")
    print("   - MultiScale_epoch/: Fusion weights evolution")
    print("   - System_epoch/: GPU memory and performance")
    print("   - Gradients/: Gradient norms and health")
    print("3. Detailed logs: JSON files in detailed_logs/ directory")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='EgoVLP Comprehensive Logging Analysis')
    parser.add_argument('-l', '--log_dir', default='saved/EgoClip_MultiScale_RTX3090',
                        help='Directory containing training logs')
    parser.add_argument('-c', '--config', 
                        help='Show logging configuration from config file')
    parser.add_argument('--demo', action='store_true',
                        help='Show demo of logging features')
    
    args = parser.parse_args()
    
    if args.demo:
        print("="*80)
        print("EGOCLIP ENHANCED LOGGING FEATURES DEMO")
        print("="*80)
        print()
        print("1. PER-ITERATION LOSS TRACKING:")
        print("   ✓ Total loss (EgoNCE + Temporal)")
        print("   ✓ EgoNCE loss components (V2T and T2V separately)")
        print("   ✓ Temporal consistency loss") 
        print("   ✓ Lambda weight scheduling")
        print("   ✓ Temperature parameter evolution")
        print()
        print("2. PER-EPOCH METRICS:")
        print("   ✓ EgoMCQ Intra-video accuracy")
        print("   ✓ EgoMCQ Inter-video accuracy")
        print("   ✓ Fusion weights: w_fine, w_medium, w_coarse")
        print("   ✓ Current temperature tau")
        print("   ✓ Training throughput (samples/second)")
        print()
        print("3. SYSTEM STATISTICS:")
        print("   ✓ GPU memory usage (max per GPU)")
        print("   ✓ Training time per epoch")
        print("   ✓ Memory efficiency tracking")
        print()
        print("4. GRADIENT ANALYSIS (every 100 iterations):")
        print("   ✓ Total gradient norm")
        print("   ✓ Maximum gradient norm")
        print("   ✓ Fusion weight gradient flow")
        print("   ✓ Gradient health monitoring")
        print()
        print("5. TENSORBOARD INTEGRATION:")
        print("   ✓ Real-time metric visualization")
        print("   ✓ Scalar plots with automatic grouping")
        print("   ✓ Multi-GPU aggregation")
        print()
        print("6. DETAILED JSON EXPORTS:")
        print("   ✓ Per-epoch comprehensive data")
        print("   ✓ Full iteration history")
        print("   ✓ Statistical analysis ready")
        print()
        print("="*80)
        print("USAGE EXAMPLE:")
        print("="*80)
        print("# Start training with enhanced logging")
        print("python run/train_egoclip.py -c configs/pt/egoclip_rtx3090_optimized.json")
        print()
        print("# View real-time metrics")  
        print("tensorboard --logdir saved/EgoClip_MultiScale_RTX3090/tensorboard")
        print()
        print("# Analyze completed training")
        print("python logging_demo.py -l saved/EgoClip_MultiScale_RTX3090")
        print("="*80)
        return
    
    if args.config:
        print("="*60)
        print("LOGGING CONFIGURATION")
        print("="*60)
        try:
            with open(args.config) as f:
                config = json.load(f)
            
            # Extract logging-related config
            logging_config = config.get('logging', {})
            trainer_config = config.get('trainer', {})
            
            print("Trainer logging settings:")
            for key, value in trainer_config.items():
                if 'log' in key.lower() or 'tensorboard' in key.lower():
                    print(f"  {key}: {value}")
            
            print("\nLogging module settings:")
            for key, value in logging_config.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"Error reading config: {e}")
        return
    
    # Analyze existing logs
    if not os.path.exists(args.log_dir):
        print(f"Log directory not found: {args.log_dir}")
        print("Make sure to run training first or check the path.")
        return
    
    generate_logging_report(args.log_dir)


if __name__ == '__main__':
    main()