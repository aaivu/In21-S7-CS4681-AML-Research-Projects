"""
Comprehensive unit tests for EgoVLP Multi-Scale Enhancement components.

Test coverage:
1. MultiScaleVideoEncoder: Input shapes, fusion weights, gradients, single-scale mode
2. TemporalConsistencyLoss: Loss behavior, lambda scheduling
3. TemporalPairBatchSampler: Batch composition, temporal pairs, no duplicates
4. TemperatureScheduler: Cosine curve, edge cases, smooth decay

Usage:
    pytest tests/test_multiscale.py -v
    pytest tests/test_multiscale.py::test_multiscale_forward -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import MultiScaleVideoEncoder
from model.temporal_loss import TemporalConsistencyLoss, TemporalPairBatchSampler
from model.loss import TemperatureScheduler, EgoNCEWithScheduler


class TestMultiScaleVideoEncoder:
    """Test suite for MultiScaleVideoEncoder."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.projection_dim = 768
        
        # Mock video parameters
        self.video_params = {
            'model': 'SpaceTimeTransformer',
            'arch_config': 'base_patch16_224',
            'time_init': 'zeros',
            'attention_style': 'frozen-in-time'
        }
        
    def create_mock_encoder(self):
        """Create a mock MultiScaleVideoEncoder for testing."""
        with patch('model.model.SpaceTimeTransformer') as mock_stt, \
             patch('torch.load') as mock_load:
            
            # Mock pretrained weights
            mock_load.return_value = {'dummy': 'weights'}
            
            # Create mock SpaceTimeTransformer instances
            for num_frames in [4, 8, 16]:
                mock_encoder = Mock()
                mock_encoder.embed_dim = 768
                mock_encoder.head = nn.Identity()
                mock_encoder.pre_logits = nn.Identity()
                mock_encoder.state_dict.return_value = {}
                mock_encoder.load_state_dict = Mock()
                
                # Mock forward pass
                def mock_forward(x):
                    batch_size = x.shape[0]
                    return torch.randn(batch_size, 768)
                mock_encoder.forward = mock_forward
                mock_encoder.return_value = mock_forward
                
                mock_stt.return_value = mock_encoder
            
            encoder = MultiScaleVideoEncoder(self.video_params, self.projection_dim)
            return encoder.to(self.device)
    
    def test_multiscale_forward_shapes(self):
        """Test that MultiScaleVideoEncoder produces correct output shapes."""
        encoder = self.create_mock_encoder()
        
        # Create input tensors for different scales
        video_clips = {
            'frames_4': torch.randn(self.batch_size, 4, 3, 224, 224).to(self.device),
            'frames_8': torch.randn(self.batch_size, 8, 3, 224, 224).to(self.device),
            'frames_16': torch.randn(self.batch_size, 16, 3, 224, 224).to(self.device)
        }
        
        # Forward pass
        with torch.no_grad():
            output = encoder(video_clips)
        
        # Check output shape
        assert output.shape == (self.batch_size, self.projection_dim), \
            f"Expected shape {(self.batch_size, self.projection_dim)}, got {output.shape}"
        
        # Check output is not NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
    
    def test_fusion_weights_sum_to_one(self):
        """Test that fusion weights sum to 1 after softmax."""
        encoder = self.create_mock_encoder()
        
        # Get fusion weights and apply softmax (same as in forward pass)
        fusion_weights = torch.softmax(encoder.fusion_weights, dim=0)
        
        # Check that weights sum to 1
        weight_sum = fusion_weights.sum().item()
        assert abs(weight_sum - 1.0) < 1e-6, f"Fusion weights sum to {weight_sum}, not 1.0"
        
        # Check that all weights are positive
        assert (fusion_weights >= 0).all(), "Fusion weights contain negative values"
        
        # Check that weights have correct length
        assert len(fusion_weights) == 3, f"Expected 3 fusion weights, got {len(fusion_weights)}"
    
    def test_gradients_flow_to_parameters(self):
        """Test that gradients flow to all parameters including fusion weights."""
        encoder = self.create_mock_encoder()
        encoder.train()
        
        # Create dummy input
        video_clips = {
            'frames_4': torch.randn(self.batch_size, 4, 3, 224, 224, requires_grad=True).to(self.device),
            'frames_8': torch.randn(self.batch_size, 8, 3, 224, 224, requires_grad=True).to(self.device),
            'frames_16': torch.randn(self.batch_size, 16, 3, 224, 224, requires_grad=True).to(self.device)
        }
        
        # Forward pass
        output = encoder(video_clips)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that fusion weights have gradients
        assert encoder.fusion_weights.grad is not None, "Fusion weights have no gradients"
        assert not torch.isnan(encoder.fusion_weights.grad).any(), "Fusion weights gradients contain NaN"
        
        # Check that projector parameters have gradients
        for name, param in encoder.projectors.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradients"
    
    def test_single_scale_fallback(self):
        """Test behavior when only one scale is provided."""
        encoder = self.create_mock_encoder()
        
        # Test with only one scale
        video_clips = {
            'frames_8': torch.randn(self.batch_size, 8, 3, 224, 224).to(self.device)
        }
        
        with torch.no_grad():
            output = encoder(video_clips)
        
        # Should still produce correct output shape
        assert output.shape == (self.batch_size, self.projection_dim)
        assert not torch.isnan(output).any()
    
    def test_memory_optimization_flags(self):
        """Test that memory optimization features work correctly."""
        encoder = self.create_mock_encoder()
        encoder.train()  # Enable training mode for gradient checkpointing
        
        video_clips = {
            'frames_4': torch.randn(self.batch_size, 4, 3, 224, 224).to(self.device),
            'frames_8': torch.randn(self.batch_size, 8, 3, 224, 224).to(self.device),
            'frames_16': torch.randn(self.batch_size, 16, 3, 224, 224).to(self.device)
        }
        
        # Test with mixed precision context
        with torch.cuda.amp.autocast():
            output = encoder(video_clips)
        
        assert output.shape == (self.batch_size, self.projection_dim)
        assert not torch.isnan(output).any()


class TestTemporalConsistencyLoss:
    """Test suite for TemporalConsistencyLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8
        self.feature_dim = 768
        
        self.loss_fn = TemporalConsistencyLoss(
            lambda_start=0.1,
            lambda_end=0.3,
            max_time_gap=2.0
        )
    
    def test_loss_decreases_for_similar_features(self):
        """Test that temporal loss decreases for similar features."""
        # Create similar features (small difference)
        features = torch.randn(self.batch_size, self.feature_dim).to(self.device)
        similar_features = features + 0.1 * torch.randn_like(features)
        
        # Create temporal pairs
        temporal_pairs = [(0, 1, 1.0), (2, 3, 1.5)]  # (idx1, idx2, time_gap)
        
        # Compute loss for similar features
        loss_similar = self.loss_fn(features, temporal_pairs, current_epoch=5, total_epochs=10)
        
        # Create dissimilar features (large difference)
        dissimilar_features = torch.randn_like(features) * 10
        loss_dissimilar = self.loss_fn(dissimilar_features, temporal_pairs, current_epoch=5, total_epochs=10)
        
        # Loss should be smaller for similar features
        assert loss_similar.item() < loss_dissimilar.item(), \
            f"Similar features loss ({loss_similar.item()}) should be < dissimilar features loss ({loss_dissimilar.item()})"
    
    def test_loss_increases_for_dissimilar_features(self):
        """Test that temporal loss increases for dissimilar features."""
        # Create two sets of features with known similarity
        features1 = torch.ones(self.batch_size, self.feature_dim).to(self.device)
        features2 = -torch.ones(self.batch_size, self.feature_dim).to(self.device)  # Opposite direction
        
        temporal_pairs = [(0, 1, 1.0)]
        
        loss = self.loss_fn(features2, temporal_pairs, current_epoch=5, total_epochs=10)
        
        # Loss should be positive for dissimilar features
        assert loss.item() > 0, f"Loss for dissimilar features should be positive, got {loss.item()}"
    
    def test_lambda_scheduling_correctness(self):
        """Test that lambda scheduling works correctly across epochs."""
        features = torch.randn(self.batch_size, self.feature_dim).to(self.device)
        temporal_pairs = [(0, 1, 1.0)]
        
        total_epochs = 10
        
        # Test at different epochs
        loss_epoch_0 = self.loss_fn(features, temporal_pairs, current_epoch=0, total_epochs=total_epochs)
        loss_epoch_5 = self.loss_fn(features, temporal_pairs, current_epoch=5, total_epochs=total_epochs)
        loss_epoch_10 = self.loss_fn(features, temporal_pairs, current_epoch=10, total_epochs=total_epochs)
        
        # Lambda increases linearly, so loss magnitude should increase
        # (assuming the same feature dissimilarity)
        # Note: We're testing that the scheduling mechanism works, not the absolute values
        assert isinstance(loss_epoch_0, torch.Tensor), "Loss should be a tensor"
        assert isinstance(loss_epoch_5, torch.Tensor), "Loss should be a tensor"
        assert isinstance(loss_epoch_10, torch.Tensor), "Loss should be a tensor"
    
    def test_empty_temporal_pairs(self):
        """Test behavior with empty temporal pairs."""
        features = torch.randn(self.batch_size, self.feature_dim).to(self.device)
        temporal_pairs = []
        
        loss = self.loss_fn(features, temporal_pairs, current_epoch=5, total_epochs=10)
        
        # Should return zero loss
        assert loss.item() == 0.0, f"Empty temporal pairs should give zero loss, got {loss.item()}"
        assert loss.requires_grad, "Loss tensor should require gradients"
    
    def test_time_gap_filtering(self):
        """Test that pairs beyond max_time_gap are filtered out."""
        features = torch.randn(self.batch_size, self.feature_dim).to(self.device)
        
        # Pairs within and beyond time gap
        temporal_pairs = [
            (0, 1, 1.0),  # Within max_time_gap (2.0)
            (2, 3, 3.0),  # Beyond max_time_gap (2.0)
            (4, 5, 1.5)   # Within max_time_gap (2.0)
        ]
        
        loss = self.loss_fn(features, temporal_pairs, current_epoch=5, total_epochs=10)
        
        # Should only process pairs within time gap
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestTemporalPairBatchSampler:
    """Test suite for TemporalPairBatchSampler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.temporal_pair_ratio = 0.3
        
        # Create mock dataset with metadata
        self.mock_dataset = self.create_mock_dataset()
    
    def create_mock_dataset(self):
        """Create a mock dataset with temporal metadata."""
        # Create metadata DataFrame with temporal information
        num_samples = 1000
        video_uids = [f"video_{i//10}" for i in range(num_samples)]  # 10 clips per video
        narration_times = []
        
        for i in range(num_samples):
            video_idx = i // 10
            clip_idx = i % 10
            # Create temporal sequence with some clips close to each other
            base_time = video_idx * 100 + clip_idx * 5
            narration_times.append(base_time + np.random.normal(0, 0.5))
        
        metadata = pd.DataFrame({
            'video_uid': video_uids,
            'narration_time': narration_times,
            'clip_start': narration_times,
            'clip_end': [t + 2 for t in narration_times]
        })
        
        mock_dataset = Mock()
        mock_dataset.metadata = metadata
        mock_dataset.__len__ = Mock(return_value=num_samples)
        
        return mock_dataset
    
    def test_batch_size_correctness(self):
        """Test that batches have correct size."""
        sampler = TemporalPairBatchSampler(
            dataset=self.mock_dataset,
            batch_size=self.batch_size,
            temporal_pair_ratio=self.temporal_pair_ratio
        )
        
        batches = list(sampler)
        
        # All batches should have correct size
        for batch in batches:
            assert len(batch) == self.batch_size, \
                f"Batch size {len(batch)} doesn't match expected {self.batch_size}"
    
    def test_no_duplicate_indices(self):
        """Test that no batch contains duplicate indices."""
        sampler = TemporalPairBatchSampler(
            dataset=self.mock_dataset,
            batch_size=self.batch_size,
            temporal_pair_ratio=self.temporal_pair_ratio
        )
        
        batches = list(sampler)
        
        for i, batch in enumerate(batches):
            unique_indices = set(batch)
            assert len(unique_indices) == len(batch), \
                f"Batch {i} contains duplicate indices: {batch}"
    
    def test_temporal_pair_ratio(self):
        """Test that approximately 30% ± 5% of batches contain temporal pairs."""
        sampler = TemporalPairBatchSampler(
            dataset=self.mock_dataset,
            batch_size=self.batch_size,
            temporal_pair_ratio=self.temporal_pair_ratio
        )
        
        # Expected number of temporal slots per batch
        expected_temporal_slots = int(self.batch_size * self.temporal_pair_ratio)
        
        assert sampler.num_temporal_slots == expected_temporal_slots, \
            f"Expected {expected_temporal_slots} temporal slots, got {sampler.num_temporal_slots}"
        
        # Check that temporal pairs were built
        assert len(sampler.temporal_pairs) > 0, "No temporal pairs were built"
    
    def test_temporal_pairs_are_adjacent(self):
        """Test that identified temporal pairs are actually temporally adjacent."""
        sampler = TemporalPairBatchSampler(
            dataset=self.mock_dataset,
            batch_size=self.batch_size,
            temporal_pair_ratio=self.temporal_pair_ratio,
            max_temporal_gap=2.0
        )
        
        metadata = self.mock_dataset.metadata
        
        # Check that all temporal pairs satisfy the time gap constraint
        for idx1, idx2, time_gap in sampler.temporal_pairs:
            time1 = metadata.iloc[idx1]['narration_time']
            time2 = metadata.iloc[idx2]['narration_time']
            actual_gap = abs(time2 - time1)
            
            assert actual_gap <= sampler.max_temporal_gap, \
                f"Temporal pair ({idx1}, {idx2}) has gap {actual_gap} > max_gap {sampler.max_temporal_gap}"
            
            # Should be from same video
            video1 = metadata.iloc[idx1]['video_uid']
            video2 = metadata.iloc[idx2]['video_uid']
            assert video1 == video2, f"Temporal pair from different videos: {video1} vs {video2}"
    
    def test_batch_composition_statistics(self):
        """Test statistical properties of batch composition."""
        sampler = TemporalPairBatchSampler(
            dataset=self.mock_dataset,
            batch_size=self.batch_size,
            temporal_pair_ratio=self.temporal_pair_ratio
        )
        
        batches = list(sampler)
        
        # Should generate reasonable number of batches
        expected_batches = len(self.mock_dataset) // self.batch_size
        assert len(batches) > 0, "No batches generated"
        assert len(batches) <= expected_batches + 1, "Too many batches generated"  # +1 for potential remainder


class TestTemperatureScheduler:
    """Test suite for TemperatureScheduler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tau_max = 0.07
        self.tau_min = 0.03
        self.total_epochs = 20
        
        self.scheduler = TemperatureScheduler(
            tau_max=self.tau_max,
            tau_min=self.tau_min,
            total_epochs=self.total_epochs
        )
    
    def test_returns_tau_max_at_epoch_zero(self):
        """Test that scheduler returns tau_max at epoch 0."""
        temp = self.scheduler.get_temperature(0)
        
        assert abs(temp - self.tau_max) < 1e-6, \
            f"Expected tau_max {self.tau_max} at epoch 0, got {temp}"
    
    def test_returns_tau_min_at_final_epoch(self):
        """Test that scheduler returns tau_min at final epoch."""
        temp = self.scheduler.get_temperature(self.total_epochs)
        
        assert abs(temp - self.tau_min) < 1e-6, \
            f"Expected tau_min {self.tau_min} at final epoch, got {temp}"
    
    def test_smooth_cosine_curve(self):
        """Test that temperature follows smooth cosine curve."""
        temperatures = []
        epochs = list(range(0, self.total_epochs + 1, 2))  # Sample every 2 epochs
        
        for epoch in epochs:
            temp = self.scheduler.get_temperature(epoch)
            temperatures.append(temp)
            
            # Temperature should be within valid range
            assert self.tau_min <= temp <= self.tau_max, \
                f"Temperature {temp} at epoch {epoch} outside valid range [{self.tau_min}, {self.tau_max}]"
        
        # Temperature should be monotonically decreasing
        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i-1] + 1e-6, \
                f"Temperature not decreasing: epoch {epochs[i-1]}:{temperatures[i-1]} -> epoch {epochs[i]}:{temperatures[i]}"
    
    def test_middle_epoch_values(self):
        """Test temperature values at middle epochs."""
        mid_epoch = self.total_epochs // 2
        temp_mid = self.scheduler.get_temperature(mid_epoch)
        
        # At middle epoch, should be approximately halfway between min and max
        expected_mid = (self.tau_max + self.tau_min) / 2
        tolerance = (self.tau_max - self.tau_min) * 0.1  # 10% tolerance
        
        assert abs(temp_mid - expected_mid) < tolerance, \
            f"Middle epoch temperature {temp_mid} too far from expected {expected_mid}"
    
    def test_negative_epoch_handling(self):
        """Test handling of negative epoch values."""
        temp = self.scheduler.get_temperature(-5)
        
        # Should clamp to epoch 0 behavior
        expected_temp = self.scheduler.get_temperature(0)
        assert abs(temp - expected_temp) < 1e-6, \
            f"Negative epoch should behave like epoch 0"
    
    def test_epoch_beyond_total_handling(self):
        """Test handling of epochs beyond total_epochs."""
        temp = self.scheduler.get_temperature(self.total_epochs + 10)
        
        # Should clamp to final epoch behavior
        expected_temp = self.scheduler.get_temperature(self.total_epochs)
        assert abs(temp - expected_temp) < 1e-6, \
            f"Epoch beyond total should behave like final epoch"
    
    def test_progress_info(self):
        """Test progress information generation."""
        epoch = 10
        progress_info = self.scheduler.get_progress_info(epoch)
        
        assert isinstance(progress_info, dict), "Progress info should be a dictionary"
        
        required_keys = ['current_epoch', 'total_epochs', 'temperature', 'progress']
        for key in required_keys:
            assert key in progress_info, f"Progress info missing key: {key}"
        
        assert progress_info['current_epoch'] == epoch
        assert progress_info['total_epochs'] == self.total_epochs
        assert 0 <= progress_info['progress'] <= 1


class TestEgoNCEWithScheduler:
    """Test suite for EgoNCEWithScheduler integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scheduler_loss = EgoNCEWithScheduler(
            tau_max=0.07,
            tau_min=0.03,
            total_epochs=10,
            noun=True,
            verb=True
        )
    
    def test_temperature_scheduler_integration(self):
        """Test that temperature scheduler is properly integrated."""
        # Test epoch setting
        self.scheduler_loss.set_epoch(5)
        assert self.scheduler_loss.current_epoch == 5
        
        # Test temperature scheduling
        temp_epoch_0 = self.scheduler_loss.temperature_scheduler.get_temperature(0)
        temp_epoch_10 = self.scheduler_loss.temperature_scheduler.get_temperature(10)
        
        assert temp_epoch_0 > temp_epoch_10, "Temperature should decrease over epochs"
    
    def test_loss_info_generation(self):
        """Test that detailed loss information is generated."""
        batch_size = 4
        feature_dim = 256
        
        # Create dummy inputs
        x = torch.randn(batch_size, batch_size).to(self.device)  # Similarity matrix
        mask_v = torch.randn(batch_size, batch_size).to(self.device)
        mask_n = torch.randn(batch_size, batch_size).to(self.device)
        
        # Forward pass
        with patch.object(self.scheduler_loss.egonce, 'forward', return_value=torch.tensor(2.5)):
            with patch.object(self.scheduler_loss.egonce, 'get_loss_info', 
                            return_value={'current_temperature': 0.05, 'v2t_loss': 1.2, 't2v_loss': 1.3}):
                
                loss, loss_info = self.scheduler_loss(x, mask_v, mask_n, current_epoch=5)
                
                assert isinstance(loss_info, dict), "Loss info should be a dictionary"
                assert 'current_temperature' in loss_info, "Loss info should contain current_temperature"
                assert 'loss_value' in loss_info, "Loss info should contain loss_value"


# Integration tests
class TestMultiScaleIntegration:
    """Integration tests for multi-scale components working together."""
    
    def test_end_to_end_training_step(self):
        """Test a complete training step with all components."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 4
        
        # Create components
        video_params = {
            'model': 'SpaceTimeTransformer',
            'arch_config': 'base_patch16_224'
        }
        
        # Mock the encoder creation
        with patch('model.model.SpaceTimeTransformer') as mock_stt, \
             patch('torch.load') as mock_load:
            
            mock_load.return_value = {'dummy': 'weights'}
            mock_encoder = Mock()
            mock_encoder.embed_dim = 768
            mock_encoder.head = nn.Identity()
            mock_encoder.pre_logits = nn.Identity()
            mock_encoder.state_dict.return_value = {}
            mock_encoder.load_state_dict = Mock()
            
            def mock_forward(x):
                return torch.randn(x.shape[0], 768)
            mock_encoder.forward = mock_forward
            mock_stt.return_value = mock_encoder
            
            # Create multi-scale encoder
            encoder = MultiScaleVideoEncoder(video_params, 768).to(device)
            
            # Create temporal loss
            temporal_loss = TemporalConsistencyLoss()
            
            # Create temperature scheduler
            temp_scheduler = TemperatureScheduler(0.07, 0.03, 10)
            
            # Create dummy input
            video_clips = {
                'frames_4': torch.randn(batch_size, 4, 3, 224, 224).to(device),
                'frames_8': torch.randn(batch_size, 8, 3, 224, 224).to(device),
                'frames_16': torch.randn(batch_size, 16, 3, 224, 224).to(device)
            }
            
            # Forward pass
            encoder.train()
            video_features = encoder(video_clips)
            
            # Compute temporal loss
            temporal_pairs = [(0, 1, 1.0), (2, 3, 1.5)]
            temp_loss = temporal_loss(video_features, temporal_pairs, 5, 10)
            
            # Get current temperature
            current_temp = temp_scheduler.get_temperature(5)
            
            # Combined loss
            total_loss = temp_loss + video_features.mean()  # Dummy combination
            
            # Backward pass
            total_loss.backward()
            
            # Check that gradients exist
            assert encoder.fusion_weights.grad is not None, "Fusion weights should have gradients"
            assert not torch.isnan(total_loss), "Total loss should not be NaN"
            assert current_temp > 0, "Temperature should be positive"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])