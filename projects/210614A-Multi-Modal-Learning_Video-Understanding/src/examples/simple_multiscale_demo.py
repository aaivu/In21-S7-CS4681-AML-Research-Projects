"""
Simple example demonstrating MultiScaleVideoEncoder usage.

This script shows how to use the new MultiScaleVideoEncoder class
with minimal dependencies and clear examples.
"""

import torch
import torch.nn as nn


def demo_multiscale_encoder():
    """
    Demonstrate MultiScaleVideoEncoder with synthetic data.
    """
    
    print("Multi-Scale Video Encoder Demo")
    print("==============================")
    
    # Video parameters (matching EgoVLP configuration)
    video_params = {
        'model': 'SpaceTimeTransformer',
        'arch_config': 'base_patch16_224',
        'pretrained': True,
        'time_init': 'zeros'
    }
    
    # Create MultiScaleVideoEncoder
    try:
        from model.model import MultiScaleVideoEncoder
        encoder = MultiScaleVideoEncoder(video_params, projection_dim=768)
        print("✓ MultiScaleVideoEncoder created successfully")
    except Exception as e:
        print(f"✗ Error creating encoder: {e}")
        print("  Make sure pretrained/jx_vit_base_p16_224-80ecf9dd.pth exists")
        return
    
    # Create synthetic multi-scale video data
    batch_size = 4
    video_clips = {
        'frames_4': torch.randn(batch_size, 4, 3, 224, 224),   # Fine scale
        'frames_8': torch.randn(batch_size, 8, 3, 224, 224),   # Medium scale
        'frames_16': torch.randn(batch_size, 16, 3, 224, 224)  # Coarse scale
    }
    
    print(f"\nInput shapes:")
    for key, tensor in video_clips.items():
        print(f"  {key}: {tensor.shape}")
    
    # Forward pass
    print("\nProcessing video clips sequentially...")
    try:
        with torch.no_grad():  # Inference mode
            fused_features = encoder(video_clips)
        
        print(f"✓ Output shape: {fused_features.shape}")
        print(f"✓ Expected: [batch_size={batch_size}, projection_dim=768]")
        
        # Display fusion weights
        fusion_weights = encoder.get_fusion_weights()
        print(f"\nFusion weights after softmax: {fusion_weights.detach().numpy()}")
        print(f"Sum of weights: {fusion_weights.sum().item():.6f} (should be 1.0)")
        
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        return
    
    print("\n" + "="*50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*50)


def demo_temporal_consistency():
    """
    Demonstrate temporal consistency loss with synthetic data.
    """
    
    print("\nTemporal Consistency Loss Demo")
    print("==============================")
    
    try:
        from model.temporal_loss import TemporalConsistencyLoss
        
        # Create temporal loss
        temporal_loss = TemporalConsistencyLoss(lambda_start=0.1, lambda_end=0.3)
        print("✓ TemporalConsistencyLoss created successfully")
        
        # Synthetic video features
        batch_size = 6
        feature_dim = 768
        video_features = torch.randn(batch_size, feature_dim)
        
        # Define temporal pairs: (idx1, idx2, time_gap)
        # Pairs within 2 seconds should have consistency loss applied
        temporal_pairs = [
            (0, 1, 0.5),  # 0.5 seconds apart - should apply loss
            (2, 3, 1.2),  # 1.2 seconds apart - should apply loss
            (4, 5, 3.0),  # 3.0 seconds apart - should NOT apply loss
        ]
        
        print(f"\nVideo features shape: {video_features.shape}")
        print(f"Temporal pairs: {temporal_pairs}")
        
        # Compute loss at different epochs
        total_epochs = 100
        for epoch in [0, 25, 50, 75, 99]:
            loss = temporal_loss(video_features, temporal_pairs, epoch, total_epochs)
            progress = epoch / total_epochs
            expected_lambda = 0.1 + (0.3 - 0.1) * progress
            
            print(f"Epoch {epoch:2d}: Loss = {loss.item():.6f}, Lambda = {expected_lambda:.3f}")
        
        print("✓ Temporal consistency loss computed successfully")
        
    except Exception as e:
        print(f"✗ Error in temporal consistency demo: {e}")


def demo_multiscale_data_loading():
    """
    Demonstrate multi-scale frame sampling logic.
    """
    
    print("\nMulti-Scale Data Loading Demo")
    print("=============================")
    
    # Simulate video metadata
    video_metadata = {
        'video_uid': 'sample_video_001',
        'clip_start': 10.0,  # seconds
        'clip_end': 12.5,    # seconds (2.5 second clip)
        'clip_text': 'Person cooking in kitchen'
    }
    
    print(f"Video clip: {video_metadata['clip_start']}s to {video_metadata['clip_end']}s")
    print(f"Duration: {video_metadata['clip_end'] - video_metadata['clip_start']}s")
    
    # Demonstrate frame sampling logic
    clip_duration = video_metadata['clip_end'] - video_metadata['clip_start']
    scales = [4, 8, 16]
    
    print(f"\nFrame sampling for {clip_duration}s clip:")
    
    for num_frames in scales:
        if clip_duration < 1.0 and num_frames == 16:
            print(f"  {num_frames} frames: Skip coarse scale (clip too short)")
        else:
            frame_interval = clip_duration / num_frames
            print(f"  {num_frames} frames: Sample every {frame_interval:.3f}s")
    
    # Simulate the multi-scale output structure
    batch_size = 2
    simulated_output = {}
    
    for num_frames in scales:
        if not (clip_duration < 1.0 and num_frames == 16):
            simulated_output[f'frames_{num_frames}'] = torch.zeros(batch_size, num_frames, 3, 224, 224)
    
    print(f"\nSimulated output structure:")
    for key, tensor in simulated_output.items():
        print(f"  {key}: {tensor.shape}")


if __name__ == '__main__':
    print("EgoVLP Multi-Scale Enhancement Demo")
    print("===================================")
    
    # Run demos
    demo_multiscale_encoder()
    demo_temporal_consistency()
    demo_multiscale_data_loading()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED!")
    print("="*60)
    
    print("\nKey Features Demonstrated:")
    print("• MultiScaleVideoEncoder with sequential processing")
    print("• Learnable fusion weights with softmax normalization")
    print("• Temporal consistency loss with scheduled lambda")
    print("• Multi-scale frame sampling from video clips")
    print("• Memory-efficient sequential processing")
    
    print("\nNext Steps:")
    print("1. Integrate with your EgoVLP training pipeline")
    print("2. Adjust fusion weights and temporal loss parameters")
    print("3. Experiment with different scales and projection dimensions")
    print("4. Monitor fusion weight evolution during training")