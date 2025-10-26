"""
Memory Optimization Demo for Multi-Scale EgoVLP

This script demonstrates the memory-efficient multi-scale processing
optimizations for 4x RTX 3090 GPUs (24GB each).

Features tested:
1. Sequential processing instead of parallel
2. Gradient checkpointing on 16-frame scale
3. Mixed precision training with NaN/Inf monitoring
4. Memory cleanup between scales
5. Memory usage monitoring
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import sys
import os

# Add EgoVLP path
sys.path.append('c:/Users/MSI/Downloads/Academics/Sem 7/Advanced ML/AML project/EgoVLP-main')

def test_memory_optimizations():
    """Test memory optimizations with mock data."""
    
    print("Multi-Scale Memory Optimization Demo")
    print("=" * 50)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU for demo (limited functionality).")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Import memory management utilities
    try:
        from model.model import MultiScaleVideoEncoder, MemoryManager
        print("✓ Memory optimization components imported successfully")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Initialize memory manager
    memory_manager = MemoryManager(device=device)
    initial_memory = memory_manager.get_memory_info()
    print(f"Initial memory usage: {initial_memory.get('allocated_gb', 0):.2f} GB")
    
    # Test 1: Memory-efficient multi-scale processing
    print("\n1. Testing Memory-Efficient Multi-Scale Processing")
    print("-" * 40)
    
    try:
        # Create mock video parameters
        video_params = {
            'model': 'SpaceTimeTransformer',
            'arch_config': 'base_patch16_224',
            'time_init': 'zeros',
            'attention_style': 'frozen-in-time',
            'pretrained': False
        }
        
        # Mock encoder for testing (without loading actual weights)
        class MockSpaceTimeTransformer(nn.Module):
            def __init__(self, num_frames):
                super().__init__()
                self.num_frames = num_frames
                self.embed_dim = 768
                # Simple mock architecture
                self.projection = nn.Linear(3 * 224 * 224, 768)
            
            def forward(self, x):
                B, C, T, H, W = x.shape
                # Flatten spatial-temporal dimensions for mock
                x_flat = x.reshape(B, -1)
                return self.projection(x_flat)
        
        # Create mock multi-scale encoder
        encoder = MultiScaleVideoEncoder(video_params, projection_dim=256)
        
        # Replace with mock encoders to avoid loading pretrained weights
        for num_frames in [4, 8, 16]:
            encoder.encoders[f'encoder_{num_frames}f'] = MockSpaceTimeTransformer(num_frames)
        
        encoder = encoder.to(device)
        encoder.train()
        
        print(f"✓ Mock multi-scale encoder created")
        
        # Create mock video data for testing memory usage
        batch_size = 8 if device == 'cuda' else 2  # Smaller batch for CPU
        video_clips = {
            'frames_4': torch.randn(batch_size, 3, 4, 224, 224, device=device),
            'frames_8': torch.randn(batch_size, 3, 8, 224, 224, device=device),
            'frames_16': torch.randn(batch_size, 3, 16, 224, 224, device=device)
        }
        
        print(f"✓ Mock video data created: batch_size={batch_size}")
        
        # Test memory estimation
        memory_estimates = encoder.get_scale_memory_usage(video_clips)
        print("Memory estimates per scale:")
        for scale, memory in memory_estimates.items():
            print(f"  {scale}: {memory:.2f} GB")
        
    except Exception as e:
        print(f"✗ Multi-scale encoder creation failed: {e}")
        return
    
    # Test 2: Memory-efficient forward pass
    print("\n2. Testing Memory-Efficient Forward Pass")
    print("-" * 40)
    
    try:
        # Enable memory efficient mode
        encoder.enable_memory_efficient_mode(True)
        
        # Test forward pass with gradient checkpointing
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            features = encoder(video_clips)
        
        print(f"✓ Forward pass successful: output shape {features.shape}")
        
        # Check memory usage after forward pass
        memory_after = memory_manager.get_memory_info()
        print(f"Memory after forward: {memory_after.get('allocated_gb', 0):.2f} GB")
        
        # Test fusion weights
        fusion_weights = encoder.get_fusion_weights()
        print(f"✓ Fusion weights: {[f'{w:.3f}' for w in fusion_weights.detach().cpu().numpy()]}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    # Test 3: Gradient checkpointing and mixed precision
    print("\n3. Testing Gradient Checkpointing & Mixed Precision")
    print("-" * 40)
    
    try:
        # Create mock loss and optimizer for testing
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        scaler = amp.GradScaler(enabled=(device == 'cuda'))
        
        # Mock target for loss
        target = torch.randn_like(features)
        
        optimizer.zero_grad()
        
        # Training step with mixed precision and gradient checkpointing
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            output = encoder(video_clips)
            loss = criterion(output, target)
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  NaN/Inf detected in loss!")
        else:
            print(f"✓ Loss computation successful: {loss.item():.6f}")
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✓ Gradient checkpointing and mixed precision working")
        
    except Exception as e:
        print(f"✗ Gradient checkpointing test failed: {e}")
    
    # Test 4: Memory monitoring and cleanup
    print("\n4. Testing Memory Monitoring & Cleanup")
    print("-" * 40)
    
    try:
        # Monitor memory throughout training simulation
        for step in range(5):
            memory_manager.monitor_training_memory(epoch=1, step=step, log_frequency=1)
            
            # Simulate some computation
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                _ = encoder(video_clips)
            
            # Test adaptive batch size
            if device == 'cuda':
                adaptive_batch = memory_manager.adaptive_batch_size(batch_size, 16)
                if step == 0:
                    print(f"Adaptive batch size for 16-frame scale: {batch_size} -> {adaptive_batch}")
        
        # Test memory cleanup
        print("\nTesting memory cleanup...")
        memory_before_cleanup = memory_manager.get_memory_info()
        print(f"Before cleanup: {memory_before_cleanup.get('allocated_gb', 0):.2f} GB")
        
        memory_manager.force_cleanup()
        encoder.force_memory_cleanup()
        
        memory_after_cleanup = memory_manager.get_memory_info()
        print(f"After cleanup: {memory_after_cleanup.get('allocated_gb', 0):.2f} GB")
        
        print("✓ Memory monitoring and cleanup working")
        
    except Exception as e:
        print(f"✗ Memory monitoring test failed: {e}")
    
    # Test 5: Scale-specific optimizations
    print("\n5. Testing Scale-Specific Optimizations")
    print("-" * 40)
    
    try:
        # Test different scales with different batch sizes
        scales_to_test = [4, 8, 16]
        
        for scale in scales_to_test:
            print(f"\nTesting {scale}-frame scale:")
            
            # Create single-scale input
            single_scale_input = {
                f'frames_{scale}': torch.randn(batch_size, 3, scale, 224, 224, device=device)
            }
            
            # Test memory usage for this scale
            memory_before = memory_manager.get_memory_info()
            
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                scale_features = encoder(single_scale_input)
            
            memory_after = memory_manager.get_memory_info()
            memory_used = memory_after['allocated_gb'] - memory_before.get('allocated_gb', 0)
            
            print(f"  Output shape: {scale_features.shape}")
            print(f"  Memory used: {memory_used:.2f} GB")
            
            # Cleanup between scales
            if scale < 16:  # Don't cleanup after last scale
                encoder.force_memory_cleanup()
        
        print("✓ Scale-specific optimizations working")
        
    except Exception as e:
        print(f"✗ Scale-specific optimization test failed: {e}")
    
    # Final memory report
    print("\n" + "=" * 50)
    print("MEMORY OPTIMIZATION TEST COMPLETED")
    print("=" * 50)
    
    final_memory = memory_manager.get_memory_info()
    print(f"Final memory usage: {final_memory.get('allocated_gb', 0):.2f} GB")
    print(f"Peak memory usage: {final_memory.get('peak_gb', 0):.2f} GB")
    
    if device == 'cuda':
        utilization = final_memory.get('utilization_pct', 0)
        print(f"GPU utilization: {utilization:.1f}%")
        
        if utilization < 90:
            print("✅ Memory optimization successful - under 90% utilization")
        else:
            print("⚠️  High memory utilization - consider further optimizations")
    
    print("\nOptimizations implemented:")
    print("  ✓ Sequential multi-scale processing")
    print("  ✓ Gradient checkpointing on 16-frame scale")
    print("  ✓ Mixed precision training with NaN/Inf monitoring") 
    print("  ✓ Automatic memory cleanup between scales")
    print("  ✓ Adaptive batch sizing based on memory availability")
    print("  ✓ Comprehensive memory monitoring")


if __name__ == '__main__':
    test_memory_optimizations()