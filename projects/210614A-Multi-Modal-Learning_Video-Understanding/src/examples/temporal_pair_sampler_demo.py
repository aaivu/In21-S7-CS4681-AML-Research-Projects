"""
Demo script for TemporalPairBatchSampler usage with EgoClip dataset.

This script demonstrates how to use the custom batch sampler that ensures
temporal pairs are included in training batches for temporal consistency loss.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

# Import the new sampler
from model.temporal_loss import TemporalPairBatchSampler, TemporalConsistencyLoss


class MockEgoClipDataset:
    """
    Mock EgoClip dataset for demonstration purposes.
    
    Simulates the actual EgoClip metadata structure with video_uid, 
    narration_time, clip_start, and clip_end columns.
    """
    
    def __init__(self, num_samples=1000, num_videos=50):
        self.num_samples = num_samples
        self.num_videos = num_videos
        
        # Generate mock metadata similar to EgoClip format
        self.metadata = self._generate_mock_metadata()
    
    def _generate_mock_metadata(self):
        """Generate mock metadata with realistic temporal structure."""
        data = []
        
        for video_idx in range(self.num_videos):
            video_uid = f"video_{video_idx:04d}"
            
            # Generate 10-30 clips per video with realistic timestamps
            num_clips = np.random.randint(10, 30)
            base_time = np.random.uniform(0, 3600)  # Random start within an hour
            
            for clip_idx in range(num_clips):
                # Sequential narration times with some variation
                narration_time = base_time + clip_idx * 5 + np.random.uniform(-2, 2)
                
                # Clip boundaries around narration time
                clip_start = narration_time + np.random.uniform(-1, 1)
                clip_end = clip_start + np.random.uniform(2, 8)  # 2-8 second clips
                
                data.append({
                    'video_uid': video_uid,
                    'narration_time': narration_time,
                    'clip_start': max(0, clip_start),
                    'clip_end': clip_end,
                    'clip_text': f'Action {clip_idx} in {video_uid}',
                    'tag_noun': '[1, 5, 12]',  # Mock noun tags
                    'tag_verb': '[2, 8]'       # Mock verb tags
                })
        
        # Convert to DataFrame (similar to EgoClip structure)
        df = pd.DataFrame(data)
        
        # Sort by video_uid and narration_time for realistic structure
        df = df.sort_values(['video_uid', 'narration_time']).reset_index(drop=True)
        
        return df
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Return mock data item."""
        sample = self.metadata.iloc[idx]
        
        # Mock video tensors (multi-scale)
        video_data = {
            'frames_4': torch.randn(4, 3, 224, 224),
            'frames_8': torch.randn(8, 3, 224, 224),
            'frames_16': torch.randn(16, 3, 224, 224)
        }
        
        return {
            'video': video_data,
            'text': sample['clip_text'],
            'video_uid': sample['video_uid'],
            'narration_time': sample['narration_time'],
            'meta': {
                'video_uid': sample['video_uid'],
                'narration_time': sample['narration_time']
            }
        }


def analyze_temporal_pairs(dataset, sampler):
    """
    Analyze the temporal pairs found by the sampler.
    
    Args:
        dataset: EgoClip dataset
        sampler: TemporalPairBatchSampler instance
    """
    print("Temporal Pair Analysis")
    print("=" * 40)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of videos: {len(sampler.video_clips)}")
    print(f"Total temporal pairs found: {len(sampler.temporal_pairs)}")
    
    # Analyze distribution of temporal pairs by video
    pairs_per_video = defaultdict(int)
    temporal_distances = []
    
    for idx1, idx2, distance in sampler.temporal_pairs:
        video_uid1 = dataset.metadata.iloc[idx1]['video_uid']
        video_uid2 = dataset.metadata.iloc[idx2]['video_uid']
        
        assert video_uid1 == video_uid2, "Temporal pairs should be from same video"
        
        pairs_per_video[video_uid1] += 1
        temporal_distances.append(distance)
    
    print(f"Average pairs per video: {np.mean(list(pairs_per_video.values())):.1f}")
    print(f"Average temporal distance: {np.mean(temporal_distances):.2f}s")
    print(f"Max temporal distance: {np.max(temporal_distances):.2f}s")
    
    # Show example temporal pairs
    print("\nExample temporal pairs:")
    for i, (idx1, idx2, distance) in enumerate(sampler.temporal_pairs[:5]):
        sample1 = dataset.metadata.iloc[idx1]
        sample2 = dataset.metadata.iloc[idx2]
        
        print(f"  Pair {i+1}: {sample1['video_uid']}")
        print(f"    Clip 1: t={sample1['narration_time']:.1f}s")
        print(f"    Clip 2: t={sample2['narration_time']:.1f}s")
        print(f"    Distance: {distance:.2f}s")


def demo_batch_sampling():
    """
    Demonstrate batch sampling with temporal pairs.
    """
    print("Batch Sampling Demo")
    print("=" * 40)
    
    # Create mock dataset
    dataset = MockEgoClipDataset(num_samples=500, num_videos=20)
    
    # Create temporal pair batch sampler
    batch_size = 16
    temporal_pair_ratio = 0.3
    
    sampler = TemporalPairBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        temporal_pair_ratio=temporal_pair_ratio,
        max_temporal_gap=2.0
    )
    
    # Analyze the temporal pairs
    analyze_temporal_pairs(dataset, sampler)
    
    # Create DataLoader with custom sampler
    def collate_fn(batch):
        """Custom collate function to handle batch data."""
        video_data = {}
        texts = []
        video_uids = []
        narration_times = []
        
        for item in batch:
            texts.append(item['text'])
            video_uids.append(item['video_uid'])
            narration_times.append(item['narration_time'])
            
            # Stack video data
            for key, tensor in item['video'].items():
                if key not in video_data:
                    video_data[key] = []
                video_data[key].append(tensor)
        
        # Stack tensors
        for key in video_data:
            video_data[key] = torch.stack(video_data[key])
        
        return {
            'video': video_data,
            'text': texts,
            'video_uid': video_uids,
            'narration_time': narration_times
        }
    
    # Note: We can't use DataLoader directly with our custom sampler that returns metadata
    # Instead, we'll demonstrate the sampler directly
    
    print(f"\nBatch Sampling Results:")
    print(f"Batch size: {batch_size}")
    print(f"Temporal pair ratio: {temporal_pair_ratio}")
    print(f"Expected temporal pairs per batch: {int(batch_size * temporal_pair_ratio) // 2}")
    
    # Sample a few batches
    batch_count = 0
    total_pairs = 0
    
    for batch_indices, temporal_pairs_metadata in sampler:
        batch_count += 1
        num_pairs = len(temporal_pairs_metadata)
        total_pairs += num_pairs
        
        if batch_count <= 3:  # Show first 3 batches
            print(f"\nBatch {batch_count}:")
            print(f"  Indices: {batch_indices}")
            print(f"  Temporal pairs: {num_pairs}")
            
            for pair_idx, (batch_pos1, batch_pos2, distance) in enumerate(temporal_pairs_metadata):
                dataset_idx1 = batch_indices[batch_pos1]
                dataset_idx2 = batch_indices[batch_pos2]
                
                sample1 = dataset.metadata.iloc[dataset_idx1]
                sample2 = dataset.metadata.iloc[dataset_idx2]
                
                print(f"    Pair {pair_idx + 1}: batch_pos({batch_pos1},{batch_pos2}) -> dataset_idx({dataset_idx1},{dataset_idx2})")
                print(f"      Video: {sample1['video_uid']}, Times: {sample1['narration_time']:.1f}s, {sample2['narration_time']:.1f}s")
                print(f"      Distance: {distance:.2f}s")
        
        if batch_count >= 5:  # Limit to 5 batches for demo
            break
    
    avg_pairs_per_batch = total_pairs / batch_count
    print(f"\nSummary after {batch_count} batches:")
    print(f"  Average temporal pairs per batch: {avg_pairs_per_batch:.1f}")
    print(f"  Target pairs per batch: {int(batch_size * temporal_pair_ratio) // 2}")
    print(f"  Achievement ratio: {avg_pairs_per_batch / (int(batch_size * temporal_pair_ratio) // 2) * 100:.1f}%")


def demo_temporal_consistency_training():
    """
    Demonstrate how to use temporal pairs in training with consistency loss.
    """
    print("\n" + "=" * 50)
    print("Temporal Consistency Training Demo")
    print("=" * 50)
    
    # Create dataset and sampler
    dataset = MockEgoClipDataset(num_samples=200, num_videos=10)
    sampler = TemporalPairBatchSampler(dataset, batch_size=8, temporal_pair_ratio=0.5)
    
    # Create temporal consistency loss
    temporal_loss_fn = TemporalConsistencyLoss(lambda_start=0.1, lambda_end=0.3)
    
    print("Simulating training with temporal consistency loss...")
    
    # Simulate a few training steps
    for step, (batch_indices, temporal_pairs_metadata) in enumerate(sampler):
        if step >= 3:  # Just show first 3 steps
            break
            
        # Simulate video feature extraction
        batch_size = len(batch_indices)
        video_features = torch.randn(batch_size, 768)  # Mock video embeddings
        
        # Convert temporal pairs metadata for loss function
        # Format: [(dataset_idx1, dataset_idx2, temporal_distance), ...]
        temporal_pairs_for_loss = []
        for batch_pos1, batch_pos2, distance in temporal_pairs_metadata:
            temporal_pairs_for_loss.append((batch_pos1, batch_pos2, distance))
        
        # Compute temporal consistency loss
        current_epoch = 5
        total_epochs = 100
        
        consistency_loss = temporal_loss_fn(
            video_features, 
            temporal_pairs_for_loss, 
            current_epoch, 
            total_epochs
        )
        
        print(f"\nTraining Step {step + 1}:")
        print(f"  Batch size: {batch_size}")
        print(f"  Temporal pairs: {len(temporal_pairs_metadata)}")
        print(f"  Consistency loss: {consistency_loss.item():.6f}")
        
        # Show pair details
        for i, (batch_pos1, batch_pos2, distance) in enumerate(temporal_pairs_metadata):
            cosine_sim = torch.cosine_similarity(
                video_features[batch_pos1].unsqueeze(0),
                video_features[batch_pos2].unsqueeze(0)
            ).item()
            
            print(f"    Pair {i+1}: positions ({batch_pos1},{batch_pos2}), "
                  f"distance={distance:.2f}s, cosine_sim={cosine_sim:.3f}")


if __name__ == '__main__':
    print("TemporalPairBatchSampler Demo")
    print("=" * 60)
    
    # Run demonstrations
    demo_batch_sampling()
    demo_temporal_consistency_training()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    
    print("\nKey Features Demonstrated:")
    print("✓ Custom batch sampler inheriting from torch.utils.data.Sampler")
    print("✓ Grouping clips by video_uid from EgoClip metadata")
    print("✓ Identifying adjacent clips using narration_time timestamps")
    print("✓ Ensuring 30% of batch contains temporal pairs")
    print("✓ Returning indices and metadata for temporal pairs")
    print("✓ Integration with temporal consistency loss")
    
    print("\nUsage in Training:")
    print("1. Create TemporalPairBatchSampler with your EgoClip dataset")
    print("2. Iterate through batches to get (indices, temporal_pairs_metadata)")
    print("3. Use temporal_pairs_metadata with TemporalConsistencyLoss")
    print("4. Apply consistency loss during training")
    
    print("\nNext Steps:")
    print("• Replace MockEgoClipDataset with actual EgoClip dataset")
    print("• Integrate with your training loop")
    print("• Tune temporal_pair_ratio and max_temporal_gap parameters")
    print("• Monitor temporal consistency loss during training")