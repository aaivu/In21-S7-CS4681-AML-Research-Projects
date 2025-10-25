import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Sampler
import random
from collections import defaultdict


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for adjacent video clips.
    
    Computes consistency between clips (V_i, T_i) and (V_{i+1}, T_{i+1}) from the same video
    when they are within 2 seconds of each other.
    """
    
    def __init__(self, lambda_start=0.1, lambda_end=0.3, max_time_gap=2.0):
        super().__init__()
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.max_time_gap = max_time_gap
        
    def forward(self, video_features, temporal_pairs, current_epoch, total_epochs):
        """
        Compute temporal consistency loss.
        
        Args:
            video_features: Video embeddings [batch_size, feature_dim]
            temporal_pairs: List of tuples (idx1, idx2, time_gap) indicating temporal pairs
            current_epoch: Current training epoch
            total_epochs: Total number of training epochs
            
        Returns:
            loss: Scalar temporal consistency loss
        """
        if not temporal_pairs or len(temporal_pairs) == 0:
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)
        
        # Calculate current lambda with linear schedule
        progress = current_epoch / max(total_epochs, 1)
        current_lambda = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        
        total_loss = 0.0
        num_pairs = 0
        
        for idx1, idx2, time_gap in temporal_pairs:
            # Only apply loss if clips are within max_time_gap seconds
            if time_gap <= self.max_time_gap:
                v_i = video_features[idx1]  # [feature_dim]
                v_j = video_features[idx2]  # [feature_dim]
                
                # Compute cosine similarity
                cosine_sim = F.cosine_similarity(v_i.unsqueeze(0), v_j.unsqueeze(0), dim=1)
                
                # Loss = 1 - cosine_similarity
                pair_loss = 1.0 - cosine_sim
                total_loss += pair_loss
                num_pairs += 1
        
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
            return current_lambda * total_loss
        else:
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)


class TemporalPairBatchSampler(Sampler):
    """
    Custom batch sampler that ensures temporal pairs are included in training batches.
    
    Inherits from torch.utils.data.Sampler and groups clips by video_uid from EgoClip metadata.
    Identifies adjacent clips based on consecutive narration timestamps and ensures 30% of 
    each batch contains at least one temporal pair.
    """
    
    def __init__(self, dataset, batch_size, temporal_pair_ratio=0.3, max_temporal_gap=2.0):
        """
        Initialize the temporal pair batch sampler.
        
        Args:
            dataset: EgoClip dataset with metadata containing video_uid, narration_time
            batch_size: Size of each batch
            temporal_pair_ratio: Fraction of batch that should contain temporal pairs (default: 0.3)
            max_temporal_gap: Maximum time gap (seconds) to consider clips as temporal pairs (default: 2.0)
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.temporal_pair_ratio = temporal_pair_ratio
        self.max_temporal_gap = max_temporal_gap
        self.num_temporal_slots = int(batch_size * temporal_pair_ratio)
        
        # Build temporal adjacency information
        self._build_temporal_adjacency()
        
    def _build_temporal_adjacency(self):
        """
        Build mapping of temporal adjacency between clips.
        
        Groups clips by video_uid and identifies adjacent clips based on narration timestamps.
        Creates pairs of clips that are temporally close (within max_temporal_gap seconds).
        """
        self.temporal_pairs = []
        self.video_clips = defaultdict(list)  # video_uid -> list of (index, narration_time, clip_start, clip_end)
        self.all_indices = list(range(len(self.dataset.metadata)))
        
        # Group clips by video_uid and collect temporal information
        for idx in self.all_indices:
            try:
                sample = self.dataset.metadata.iloc[idx]
                video_uid = sample['video_uid']
                
                # Get temporal information
                if 'narration_time' in sample:
                    narration_time = float(sample['narration_time'])
                else:
                    # Fallback to clip_start if narration_time not available
                    narration_time = float(sample.get('clip_start', 0))
                
                clip_start = float(sample.get('clip_start', narration_time))
                clip_end = float(sample.get('clip_end', narration_time + 1.0))
                
                self.video_clips[video_uid].append((idx, narration_time, clip_start, clip_end))
                
            except (KeyError, ValueError) as e:
                # Skip samples with missing or invalid temporal information
                continue
        
        # Find temporal pairs within each video
        for video_uid, clips in self.video_clips.items():
            # Sort clips by narration_time for finding consecutive clips
            clips.sort(key=lambda x: x[1])  # Sort by narration_time
            
            for i in range(len(clips)):
                for j in range(i + 1, len(clips)):
                    idx1, narr_time1, start1, end1 = clips[i]
                    idx2, narr_time2, start2, end2 = clips[j]
                    
                    # Calculate temporal gap between clips
                    # Use both narration time gap and actual clip time gap
                    narration_gap = abs(narr_time2 - narr_time1)
                    clip_gap = max(0, start2 - end1)  # Gap between end of first clip and start of second
                    
                    # Consider as temporal pair if either narration times are close or clips are adjacent
                    if narration_gap <= self.max_temporal_gap or clip_gap <= self.max_temporal_gap:
                        temporal_distance = min(narration_gap, clip_gap)
                        self.temporal_pairs.append((idx1, idx2, temporal_distance))
                    
                    # If clips are getting too far apart in time, stop checking
                    if narration_gap > self.max_temporal_gap * 3:
                        break
        
        # Shuffle temporal pairs for variety
        random.shuffle(self.temporal_pairs)
        
        print(f"Built temporal adjacency: {len(self.temporal_pairs)} temporal pairs from {len(self.video_clips)} videos")
    
    def __iter__(self):
        """
        Generate batches with temporal pairs.
        
        Yields:
            Tuple[List[int], List[Tuple]]: (batch_indices, temporal_pairs_metadata)
                - batch_indices: List of dataset indices for the batch
                - temporal_pairs_metadata: List of (batch_idx1, batch_idx2, temporal_distance) 
                  indicating which samples in the batch are temporal pairs
        """
        # Shuffle all indices for random sampling
        shuffled_indices = self.all_indices.copy()
        random.shuffle(shuffled_indices)
        
        # Keep track of used temporal pairs to avoid duplicates
        used_pairs = set()
        available_pairs = [pair for pair in self.temporal_pairs if pair not in used_pairs]
        
        batch_count = 0
        max_batches = len(self.all_indices) // self.batch_size
        
        while batch_count < max_batches and len(shuffled_indices) >= self.batch_size:
            batch_indices = []
            temporal_pairs_metadata = []
            
            # Step 1: Add temporal pairs (30% of batch)
            pairs_added = 0
            target_pairs = min(self.num_temporal_slots // 2, len(available_pairs))  # Each pair adds 2 samples
            
            for idx1, idx2, temporal_distance in available_pairs:
                if pairs_added >= target_pairs:
                    break
                
                # Check if both indices are still available
                if idx1 in shuffled_indices and idx2 in shuffled_indices:
                    # Add the temporal pair to batch
                    batch_indices.extend([idx1, idx2])
                    
                    # Record metadata (positions in batch, temporal distance)
                    pair_metadata = (len(batch_indices)-2, len(batch_indices)-1, temporal_distance)
                    temporal_pairs_metadata.append(pair_metadata)
                    
                    # Remove from available indices
                    shuffled_indices.remove(idx1)
                    shuffled_indices.remove(idx2)
                    
                    pairs_added += 1
            
            # Step 2: Fill remaining slots with random samples
            remaining_slots = self.batch_size - len(batch_indices)
            
            for _ in range(remaining_slots):
                if shuffled_indices:
                    idx = shuffled_indices.pop(0)
                    batch_indices.append(idx)
                else:
                    # If we run out of unique indices, repeat the last one
                    if batch_indices:
                        batch_indices.append(batch_indices[-1])
            
            # Ensure batch is exactly the right size
            batch_indices = batch_indices[:self.batch_size]
            
            # Update used pairs to avoid reusing
            for idx1, idx2, temporal_distance in available_pairs[:pairs_added]:
                used_pairs.add((idx1, idx2, temporal_distance))
            
            # Remove used pairs from available pairs
            available_pairs = [pair for pair in available_pairs[pairs_added:] if pair not in used_pairs]
            
            yield batch_indices, temporal_pairs_metadata
            batch_count += 1
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return len(self.all_indices) // self.batch_size


class TemporalBatchSampler:
    """
    Legacy batch sampler - kept for backward compatibility.
    Use TemporalPairBatchSampler for new implementations.
    """
    
    def __init__(self, dataset, batch_size, temporal_pair_ratio=0.3):
        print("Warning: TemporalBatchSampler is deprecated. Use TemporalPairBatchSampler instead.")
        self.sampler = TemporalPairBatchSampler(dataset, batch_size, temporal_pair_ratio)
        
    def __iter__(self):
        return self.sampler.__iter__()
    
    def __len__(self):
        return self.sampler.__len__()


class EnhancedEgoNCEWithTemporal(nn.Module):
    """
    Enhanced EgoNCE loss with temporal consistency.
    
    Combines the original EgoNCE loss with temporal consistency regularization.
    """
    
    def __init__(self, temperature=0.05, noun=True, verb=True, 
                 temporal_lambda_start=0.1, temporal_lambda_end=0.3):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature
        
        # Temporal consistency loss
        self.temporal_loss = TemporalConsistencyLoss(
            lambda_start=temporal_lambda_start,
            lambda_end=temporal_lambda_end
        )
    
    def forward(self, similarity_matrix, mask_v, mask_n, video_features=None, 
                temporal_pairs=None, current_epoch=0, total_epochs=100):
        """
        Forward pass with EgoNCE and temporal consistency losses.
        
        Args:
            similarity_matrix: Similarity matrix between text and video embeddings
            mask_v: Verb mask for EgoNCE
            mask_n: Noun mask for EgoNCE  
            video_features: Video embeddings for temporal consistency
            temporal_pairs: List of temporal pairs in the batch
            current_epoch: Current training epoch
            total_epochs: Total training epochs
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with loss components
        """
        # Original EgoNCE loss
        mask_diag = torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun:
            mask = mask_n + mask_diag
        else:
            mask = mask_v + mask_diag

        i_sm = F.softmax(similarity_matrix / self.temperature, dim=1)
        j_sm = F.softmax(similarity_matrix.t() / self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1))
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1))
        loss_j = jdiag.sum() / len(jdiag)
        
        egonce_loss = -loss_i - loss_j
        
        # Temporal consistency loss
        temporal_loss = torch.tensor(0.0, device=similarity_matrix.device)
        if video_features is not None and temporal_pairs is not None:
            temporal_loss = self.temporal_loss(video_features, temporal_pairs, 
                                             current_epoch, total_epochs)
        
        total_loss = egonce_loss + temporal_loss
        
        loss_dict = {
            'egonce_loss': egonce_loss,
            'temporal_loss': temporal_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict