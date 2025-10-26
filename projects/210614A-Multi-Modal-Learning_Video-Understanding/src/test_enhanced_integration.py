"""
Quick test script to verify enhanced EgoVLP integration.

This script tests the key components without requiring full dataset or training.
"""

import sys
import os
sys.path.append('c:/Users/MSI/Downloads/Academics/Sem 7/Advanced ML/AML project/EgoVLP-main')

import torch
import torch.nn as nn
import json

def test_imports():
    """Test that all enhanced components can be imported."""
    print("Testing enhanced component imports...")
    
    try:
        from model.model import MultiScaleVideoEncoder
        print("✓ MultiScaleVideoEncoder imported successfully")
    except Exception as e:
        print(f"✗ MultiScaleVideoEncoder import failed: {e}")
    
    try:
        from model.loss import EgoNCEWithScheduler, TemperatureScheduler
        print("✓ Temperature scheduling components imported successfully")
    except Exception as e:
        print(f"✗ Temperature scheduling import failed: {e}")
    
    try:
        from model.temporal_loss import TemporalConsistencyLoss, TemporalPairBatchSampler
        print("✓ Temporal components imported successfully")
    except Exception as e:
        print(f"✗ Temporal components import failed: {e}")
    
    try:
        from trainer.enhanced_trainer_egoclip import Enhanced_Multi_Trainer_dist
        print("✓ Enhanced trainer imported successfully")
    except Exception as e:
        print(f"✗ Enhanced trainer import failed: {e}")


def test_temperature_scheduling():
    """Test temperature scheduling functionality."""
    print("\nTesting temperature scheduling...")
    
    try:
        from model.loss import TemperatureScheduler, EgoNCEWithScheduler
        
        # Test scheduler
        scheduler = TemperatureScheduler(tau_max=0.07, tau_min=0.03, total_epochs=10)
        
        print("Temperature schedule:")
        for epoch in [0, 2, 5, 9, 10]:
            temp = scheduler.get_temperature(epoch)
            print(f"  Epoch {epoch}: {temp:.6f}")
        
        # Test enhanced EgoNCE
        loss_fn = EgoNCEWithScheduler(tau_max=0.07, tau_min=0.03, total_epochs=10)
        
        # Mock data
        batch_size = 4
        similarity = torch.randn(batch_size, batch_size) * 0.5
        mask_v = torch.eye(batch_size)
        mask_n = torch.eye(batch_size)
        
        # Test loss computation
        loss, info = loss_fn(similarity, mask_v, mask_n, current_epoch=5)
        print(f"✓ Enhanced EgoNCE test: loss={loss.item():.6f}, temp={info['current_temperature']:.6f}")
        
    except Exception as e:
        print(f"✗ Temperature scheduling test failed: {e}")


def test_multiscale_encoder():
    """Test multi-scale video encoder."""
    print("\nTesting multi-scale video encoder...")
    
    try:
        from model.model import MultiScaleVideoEncoder
        
        # Create a mock base encoder
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3 * 224 * 224, 768)  # Simplified
            
            def forward(self, x):
                B, C, T, H, W = x.shape
                x = x.reshape(B, -1)  # Flatten for simplicity
                return self.linear(x)
        
        base_encoder = MockEncoder()
        
        # Create multi-scale encoder
        encoder = MultiScaleVideoEncoder(
            scales=[4, 8, 16],
            base_encoder=base_encoder,
            fusion_type='weighted'
        )
        
        # Test forward pass
        video = torch.randn(2, 3, 16, 224, 224)  # [B, C, T, H, W]
        output = encoder(video)
        
        print(f"✓ Multi-scale encoder test: input {video.shape} -> output {output.shape}")
        
        # Test fusion weights
        weights = encoder.get_fusion_weights()
        print(f"✓ Fusion weights: {[f'{w:.3f}' for w in weights]}")
        
    except Exception as e:
        print(f"✗ Multi-scale encoder test failed: {e}")


def test_temporal_consistency():
    """Test temporal consistency loss."""
    print("\nTesting temporal consistency loss...")
    
    try:
        from model.temporal_loss import TemporalConsistencyLoss
        
        # Create temporal loss
        temporal_loss = TemporalConsistencyLoss(lambda_temp=0.1)
        
        # Mock video embeddings and temporal pairs
        batch_size = 6
        embed_dim = 256
        video_embeds = torch.randn(batch_size, embed_dim)
        
        # Mock temporal pairs (indices of adjacent clips)
        temporal_pairs = [(0, 1), (2, 3), (4, 5)]
        
        # Compute loss
        loss = temporal_loss(video_embeds, temporal_pairs)
        print(f"✓ Temporal consistency test: loss={loss.item():.6f}")
        
    except Exception as e:
        print(f"✗ Temporal consistency test failed: {e}")


def test_configuration_loading():
    """Test enhanced configuration loading.""" 
    print("\nTesting enhanced configuration...")
    
    try:
        config_path = 'c:/Users/MSI/Downloads/Academics/Sem 7/Advanced ML/AML project/EgoVLP-main/configs/pt/egoclip_enhanced.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check key enhanced parameters
        assert config.get('use_enhancements') == True, "use_enhancements not set"
        assert 'multiscale_scales' in config, "multiscale_scales missing"
        assert 'temporal_lambda' in config, "temporal_lambda missing"
        assert 'enhanced_config' in config, "enhanced_config missing"
        
        print("✓ Enhanced configuration loaded successfully")
        print(f"  - Multi-scale scales: {config['multiscale_scales']}")
        print(f"  - Temporal lambda: {config['temporal_lambda']}")
        print(f"  - Temperature range: {config['loss']['tau_min']} - {config['loss']['tau_max']}")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")


def test_integration():
    """Test that components work together."""
    print("\nTesting component integration...")
    
    try:
        # Import all components
        from model.model import MultiScaleVideoEncoder
        from model.loss import EgoNCEWithScheduler
        from model.temporal_loss import TemporalConsistencyLoss
        
        # Create components
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3 * 224 * 224, 256)
            
            def forward(self, x):
                B, C, T, H, W = x.shape
                return self.linear(x.reshape(B, -1))
        
        # Multi-scale encoder
        multiscale_encoder = MultiScaleVideoEncoder(
            scales=[4, 8, 16],
            base_encoder=MockEncoder(),
            fusion_type='weighted'
        )
        
        # Enhanced losses
        egonce_loss = EgoNCEWithScheduler(tau_max=0.07, tau_min=0.03, total_epochs=10)
        temporal_loss = TemporalConsistencyLoss(lambda_temp=0.1)
        
        # Mock training step
        video = torch.randn(4, 3, 16, 224, 224)
        video_embeds = multiscale_encoder(video)  # Multi-scale processing
        
        # Compute losses
        similarity = torch.mm(video_embeds, video_embeds.t())
        mask = torch.eye(4)
        
        contrastive_loss, info = egonce_loss(similarity, mask, mask, current_epoch=5)
        temporal_pairs = [(0, 1), (2, 3)]
        consistency_loss = temporal_loss(video_embeds, temporal_pairs)
        
        total_loss = contrastive_loss + consistency_loss
        
        print(f"✓ Integration test successful:")
        print(f"  - Video processing: {video.shape} -> {video_embeds.shape}")
        print(f"  - Contrastive loss: {contrastive_loss.item():.6f}")
        print(f"  - Temporal loss: {consistency_loss.item():.6f}")
        print(f"  - Total loss: {total_loss.item():.6f}")
        print(f"  - Current temperature: {info['current_temperature']:.6f}")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")


def main():
    """Run all tests."""
    print("EgoVLP Enhanced Components Test Suite")
    print("=" * 50)
    
    test_imports()
    test_temperature_scheduling()
    test_multiscale_encoder()
    test_temporal_consistency()
    test_configuration_loading()
    test_integration()
    
    print("\n" + "=" * 50)
    print("TEST SUITE COMPLETED!")
    print("=" * 50)
    
    print("\nIf all tests passed (✓), your enhanced EgoVLP integration is ready!")
    print("Next steps:")
    print("1. Prepare your EgoClip dataset")
    print("2. Run: python run_enhanced_train_egoclip.py -c configs/pt/egoclip_enhanced.json")
    print("3. Monitor training with: tensorboard --logdir saved/")


if __name__ == '__main__':
    main()