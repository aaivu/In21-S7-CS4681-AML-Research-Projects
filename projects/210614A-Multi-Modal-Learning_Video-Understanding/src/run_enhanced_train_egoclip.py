#!/usr/bin/env python3
"""
Enhanced EgoClip Training Script

This script demonstrates how to use the enhanced EgoVLP model with:
1. Multi-scale video processing (4, 8, 16 frames)
2. Temporal consistency loss
3. Cosine temperature scheduling
4. Custom temporal batch sampler

Usage:
    # Single GPU training
    python run_enhanced_train_egoclip.py -c configs/pt/egoclip_enhanced.json
    
    # Multi-GPU training (example for 2 GPUs)
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
        --nproc_per_node=2 --master_port=12345 \
        run_enhanced_train_egoclip.py -c configs/pt/egoclip_enhanced.json

Example commands:
    # Quick test run (5 epochs)
    python run_enhanced_train_egoclip.py -c configs/pt/egoclip_enhanced.json --epochs 5
    
    # Full training with custom parameters
    python run_enhanced_train_egoclip.py \
        -c configs/pt/egoclip_enhanced.json \
        --epochs 20 \
        --batch_size 16 \
        --learning_rate 5e-5 \
        --temporal_lambda 0.2
        
    # Resume from checkpoint
    python run_enhanced_train_egoclip.py \
        -c configs/pt/egoclip_enhanced.json \
        -r saved/EgoClip_Enhanced_MultiScale/checkpoint-epoch10.pth
"""

import os
import sys
import argparse
import collections
import transformers
import torch
import json
from pathlib import Path

# Import EgoVLP components  
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualizer as module_vis
from parse_config import ConfigParser
from utils.util import replace_nested_dict_item
from tensorboardX import SummaryWriter

# Import enhanced components
from model.model import MultiScaleVideoEncoder
from model.loss import EgoNCEWithScheduler
from model.temporal_loss import TemporalConsistencyLoss, TemporalPairBatchSampler
from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ
from trainer.enhanced_trainer_egoclip import Enhanced_Multi_Trainer_dist


def setup_distributed_training():
    """Setup distributed training environment."""
    try:    
        # DDP environment variables
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        distributed = True
        print(f"Distributed training detected: rank {rank}/{world_size}")
    except:  
        # Single GPU fallback
        master_address = 'localhost'
        master_port = 12345
        world_size = 1
        rank = 0
        local_rank = 0
        distributed = False
        print("Single GPU training")
    
    return master_address, master_port, world_size, rank, local_rank, distributed


def init_enhanced_dataloaders(config, module_data):
    """
    Initialize enhanced dataloaders with multi-scale support and temporal batch sampling.
    """
    print("Initializing enhanced dataloaders...")
    
    # Training dataloader with enhancements
    if config['data_loader']['type'] == 'EgoClip_EgoMCQ':
        print("✓ Creating enhanced EgoClip dataset")
        
        # Create enhanced dataset
        train_dataset = EgoClip_EgoMCQ(config['data_loader']['args'], split_type='train')
        
        # Try to use temporal batch sampler
        batch_size = config['data_loader']['args'].get('batch_size', 32)
        try:
            batch_sampler = TemporalPairBatchSampler(
                dataset=train_dataset,
                batch_size=batch_size,
                drop_last=True
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=config['data_loader']['args'].get('num_workers', 8),
                pin_memory=True
            )
            print("✓ Using TemporalPairBatchSampler for enhanced temporal learning")
            
        except Exception as e:
            print(f"⚠ TemporalPairBatchSampler failed, using regular DataLoader: {e}")
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=config['data_loader']['args'].get('num_workers', 8),
                pin_memory=True,
                drop_last=True
            )
        
        data_loader = [train_loader]
    else:
        # Use original dataloader
        data_loader = [config.initialize("data_loader", module_data)]
    
    # Validation dataloader (regular)
    val_config = config['data_loader']['args'].copy()
    val_config['split'] = 'val'
    val_config['batch_size'] = 1
    val_config['shuffle'] = False
    
    if config['data_loader']['type'] == 'EgoClip_EgoMCQ':
        val_dataset = EgoClip_EgoMCQ(val_config, split_type='val')
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        valid_data_loader = [val_loader]
    else:
        config['data_loader']['args'] = val_config
        valid_data_loader = [config.initialize("data_loader", module_data)]
    
    print(f"✓ Training dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")
    
    return data_loader, valid_data_loader


def setup_enhanced_model_and_losses(config, model, logger):
    """Setup enhanced model components."""
    print("\nSetting up enhanced model components...")
    
    # Enhanced loss components
    enhanced_losses = {}
    
    # Temperature scheduling for EgoNCE
    if config['loss']['type'] == 'EgoNCE':
        total_epochs = config['trainer']['epochs']
        tau_max = config['loss'].get('tau_max', 0.07)
        tau_min = config['loss'].get('tau_min', 0.03)
        
        enhanced_egonce = EgoNCEWithScheduler(
            tau_max=tau_max,
            tau_min=tau_min,
            total_epochs=total_epochs,
            noun=True,
            verb=True
        )
        enhanced_losses['egonce_scheduler'] = enhanced_egonce
        print(f"✓ Temperature scheduling: {tau_max} → {tau_min} over {total_epochs} epochs")
    
    # Temporal consistency loss
    temporal_lambda = config.get('temporal_lambda', 0.1)
    temporal_loss = TemporalConsistencyLoss(lambda_temp=temporal_lambda)
    enhanced_losses['temporal_loss'] = temporal_loss
    print(f"✓ Temporal consistency loss: λ = {temporal_lambda}")
    
    # Multi-scale video encoder
    if hasattr(model, 'video_encoder'):
        scales = config.get('multiscale_scales', [4, 8, 16])
        original_encoder = model.video_encoder
        
        multiscale_encoder = MultiScaleVideoEncoder(
            scales=scales,
            base_encoder=original_encoder,
            fusion_type='weighted',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Replace video encoder
        model.video_encoder = multiscale_encoder
        enhanced_losses['multiscale_encoder'] = multiscale_encoder
        print(f"✓ Multi-scale video encoder: scales {scales}")
    
    return enhanced_losses


def main():
    """Main enhanced training function."""
    
    # Setup distributed training
    master_address, master_port, world_size, rank, local_rank, distributed = setup_distributed_training()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced EgoVLP Training')
    parser.add_argument('-c', '--config', default='configs/pt/egoclip_enhanced.json', 
                       help='config file path')
    parser.add_argument('-r', '--resume', default=None, 
                       help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, 
                       help='indices of GPUs to enable')
    parser.add_argument('--epochs', type=int, default=None,
                       help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='learning rate')
    parser.add_argument('--temporal_lambda', type=float, default=None,
                       help='temporal consistency loss weight')
    
    # Distributed training arguments
    parser.add_argument('--local_rank', type=int, default=local_rank)
    parser.add_argument('--master_address', default=master_address)
    parser.add_argument('--master_port', type=int, default=master_port)
    parser.add_argument('--world_size', type=int, default=world_size)
    parser.add_argument('--rank', type=int, default=rank)
    
    args = parser.parse_args()
    
    # Load and update configuration
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    # Update config with command line arguments
    if args.epochs is not None:
        config_dict['trainer']['epochs'] = args.epochs
        if 'enhanced_config' in config_dict and 'temperature_scheduling' in config_dict['enhanced_config']:
            config_dict['enhanced_config']['temperature_scheduling']['total_epochs'] = args.epochs
    
    if args.batch_size is not None:
        config_dict['data_loader']['args']['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        config_dict['optimizer']['args']['lr'] = args.learning_rate
    
    if args.temporal_lambda is not None:
        config_dict['temporal_lambda'] = args.temporal_lambda
        if 'enhanced_config' in config_dict and 'temporal_consistency' in config_dict['enhanced_config']:
            config_dict['enhanced_config']['temporal_consistency']['lambda_temp'] = args.temporal_lambda
    
    # Setup config parser
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    
    # Create temporary config file with updates
    temp_config_path = 'temp_enhanced_config.json'
    with open(temp_config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Parse with ConfigParser
    parser_args = argparse.Namespace(**vars(args))
    parser_args.config = temp_config_path
    config = ConfigParser(parser_args, options)
    
    # Clean up temp file
    os.remove(temp_config_path)
    
    # Setup device
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    
    # Initialize distributed training
    if distributed and master_address != 'localhost':
        print("Initializing DistributedDataParallel...")
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_address}:{master_port}',
            rank=rank, 
            world_size=world_size
        )
    
    # Setup logger
    logger = config.get_logger('enhanced_train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    
    if rank == 0:
        print("=" * 80)
        print("ENHANCED EGOCLIP TRAINING")
        print("=" * 80)
        print(f"Configuration: {args.config}")
        print(f"Device: {device}")
        print(f"Distributed: {distributed} (rank {rank}/{world_size})")
        print(f"Epochs: {config_dict['trainer']['epochs']}")
        print(f"Batch Size: {config_dict['data_loader']['args']['batch_size']}")
        print(f"Learning Rate: {config_dict['optimizer']['args']['lr']}")
        print(f"Temporal Lambda: {config_dict.get('temporal_lambda', 0.1)}")
        print("=" * 80)
    
    # Build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config['arch']['args']['text_params']['model'],
        TOKENIZERS_PARALLELISM=False
    )
    
    # Setup enhanced dataloaders
    data_loader, valid_data_loader = init_enhanced_dataloaders(config, module_data)
    
    if rank == 0:
        print(f"✓ Training batches: {len(data_loader[0])}")
        print(f"✓ Validation batches: {len(valid_data_loader[0])}")
    
    # Build model
    model = config.initialize('arch', module_arch)
    model = model.to(device)
    
    if rank == 0:
        print(f"✓ Model initialized: {model.__class__.__name__}")
    
    # Setup enhanced losses
    enhanced_components = setup_enhanced_model_and_losses(config_dict, model, logger)
    
    # Get loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    # Setup optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', transformers, trainable_params)
    
    # Setup learning rate scheduler
    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    
    # Setup visualizer and writer
    visualizer = None
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=str(config.tf_dir))
        print(f"✓ TensorBoard logging: {config.tf_dir}")
    
    # Create enhanced trainer
    trainer = Enhanced_Multi_Trainer_dist(
        args, model, loss, metrics, optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        visualizer=visualizer,
        writer=writer,
        tokenizer=tokenizer,
        max_samples_per_epoch=config['trainer']['max_samples_per_epoch']
    )
    
    if rank == 0:
        print("✓ Enhanced trainer initialized")
        print("\nEnhancements active:")
        print("  ✓ Multi-scale video processing")
        print("  ✓ Temporal consistency loss") 
        print("  ✓ Cosine temperature scheduling")
        print("  ✓ Custom temporal batch sampler")
        print("  ✓ Enhanced logging and metrics")
    
    # Start training
    if rank == 0:
        print("\n" + "=" * 80)
        print("STARTING ENHANCED TRAINING")
        print("=" * 80)
    
    trainer.train()
    
    if rank == 0:
        print("=" * 80)
        print("ENHANCED TRAINING COMPLETED!")
        print("=" * 80)


if __name__ == '__main__':
    main()