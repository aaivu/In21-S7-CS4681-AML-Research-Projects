"""
Enhanced nnFormer Modules
Multi-Scale Cross-Attention Framework for 3D Medical Image Segmentation

This module implements the enhanced components for nnFormer including:
- Multi-scale cross-attention mechanism
- Spatial resolution alignment
- Adaptive feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np


class SpatialAlignmentModule(nn.Module):
    """
    Spatial Resolution Alignment Module
    
    Handles spatial resolution mismatches between different encoder stages
    using differentiable trilinear interpolation and adaptive pooling.
    """
    
    def __init__(self, mode='trilinear'):
        super().__init__()
        self.mode = mode
    
    def forward(self, x, target_size):
        """
        Args:
            x: Input feature tensor (B, C, D, H, W)
            target_size: Target spatial dimensions (D, H, W)
        Returns:
            Aligned feature tensor
        """
        if isinstance(target_size, torch.Tensor):
            target_size = target_size.shape[-3:]
        
        # Interpolate to target size
        x_aligned = F.interpolate(
            x, 
            size=target_size, 
            mode=self.mode, 
            align_corners=True if self.mode != 'nearest' else None
        )
        return x_aligned


class MultiScaleCrossAttention(nn.Module):
    """
    Multi-Scale Cross-Attention Mechanism
    
    Enables bidirectional information flow between encoder stages,
    allowing high-resolution features to incorporate semantic guidance
    from lower-resolution stages.
    
    Args:
        dim_q: Dimension of query features (current stage)
        dim_kv: Dimension of key-value features (previous stage)
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projections
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
    """
    
    def __init__(self, dim_q, dim_kv, num_heads=8, qkv_bias=True, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5
        
        # Query projection from current stage
        self.q_proj = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        
        # Key and Value projections from previous stage
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Spatial alignment
        self.spatial_align = SpatialAlignmentModule(mode='trilinear')
        
    def forward(self, x_q, x_kv, return_attention=False):
        """
        Args:
            x_q: Query features from current stage (B, N_q, C_q)
            x_kv: Key-Value features from previous stage (B, N_kv, C_kv)
            return_attention: Whether to return attention weights
        Returns:
            Cross-attended features (B, N_q, C_q)
        """
        B_q, N_q, C_q = x_q.shape
        B_kv, N_kv, C_kv = x_kv.shape
        
        # Project queries
        Q = self.q_proj(x_q).reshape(B_q, N_q, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        
        # Project keys and values
        K = self.k_proj(x_kv).reshape(B_kv, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        V = self.v_proj(x_kv).reshape(B_kv, N_kv, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        
        # Compute attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ V).transpose(1, 2).reshape(B_q, N_q, C_q)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion Strategy
    
    Learns optimal weighting strategies for combining multi-scale features
    using channel-wise and spatial attention mechanisms.
    
    Args:
        dims: List of feature dimensions from different scales
        reduced_dim: Reduced dimension for attention computation
    """
    
    def __init__(self, dims, reduced_dim=64):
        super().__init__()
        self.dims = dims
        self.num_scales = len(dims)
        
        # Channel attention for each scale
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(dim, reduced_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(reduced_dim, dim, 1),
                nn.Sigmoid()
            ) for dim in dims
        ])
        
        # Spatial attention for each scale
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, reduced_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(reduced_dim, 1, 1),
                nn.Sigmoid()
            ) for dim in dims
        ])
        
        # Spatial alignment
        self.spatial_align = SpatialAlignmentModule(mode='trilinear')
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
    def forward(self, features, target_size=None):
        """
        Args:
            features: List of feature tensors from different scales [(B, C_i, D_i, H_i, W_i), ...]
            target_size: Target spatial size for alignment (D, H, W)
        Returns:
            Fused feature tensor (B, C_max, D, H, W)
        """
        if target_size is None:
            # Use the size of the first feature
            target_size = features[0].shape[-3:]
        
        aligned_features = []
        attention_weights = []
        
        for i, feat in enumerate(features):
            # Align spatial dimensions
            feat_aligned = self.spatial_align(feat, target_size)
            
            # Compute channel attention
            channel_att = self.channel_attention[i](feat_aligned)
            feat_channel = feat_aligned * channel_att
            
            # Compute spatial attention
            spatial_att = self.spatial_attention[i](feat_channel)
            feat_spatial = feat_channel * spatial_att
            
            aligned_features.append(feat_spatial)
            
            # Compute combined attention weight
            global_att = channel_att.mean(dim=[2, 3, 4], keepdim=True)
            spatial_att_global = spatial_att.mean(dim=1, keepdim=True)
            combined_att = global_att * spatial_att_global
            attention_weights.append(combined_att)
        
        # Normalize fusion weights
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Fused features with learned weights
        fused = sum([w * feat for w, feat in zip(fusion_weights, aligned_features)])
        
        return fused, attention_weights


class CrossScaleInteractionBlock(nn.Module):
    """
    Cross-Scale Interaction Block
    
    Combines cross-attention and adaptive fusion with residual connections
    for robust multi-scale feature interaction.
    
    Args:
        dim_current: Dimension of current stage features
        dim_prev: Dimension of previous stage features
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    
    def __init__(self, dim_current, dim_prev, num_heads=8, mlp_ratio=4.,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim_current)
        self.norm_kv = nn.LayerNorm(dim_prev)
        
        # Cross-attention from previous stage
        self.cross_attn = MultiScaleCrossAttention(
            dim_q=dim_current,
            dim_kv=dim_prev,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP for feature transformation
        self.norm2 = nn.LayerNorm(dim_current)
        mlp_hidden_dim = int(dim_current * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim_current, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim_current),
            nn.Dropout(drop)
        )
        
    def forward(self, x_current, x_prev):
        """
        Args:
            x_current: Features from current stage (B, N, C_current)
            x_prev: Features from previous stage (B, N_prev, C_prev)
        Returns:
            Enhanced features (B, N, C_current)
        """
        # Cross-attention with residual connection
        shortcut = x_current
        x_current = self.norm1(x_current)
        x_prev = self.norm_kv(x_prev)
        
        x_cross = self.cross_attn(x_current, x_prev)
        x_current = shortcut + self.drop_path(x_cross)
        
        # MLP with residual connection
        x_current = x_current + self.drop_path(self.mlp(self.norm2(x_current)))
        
        return x_current


class ProgressiveTrainingController(nn.Module):
    """
    Progressive Training Controller
    
    Gradually introduces cross-attention connections during training,
    starting with intra-scale attention and progressively enabling
    cross-scale interactions.
    """
    
    def __init__(self, num_stages=4, warmup_epochs=50, total_epochs=1000):
        super().__init__()
        self.num_stages = num_stages
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Stage activation schedule
        self.stage_schedule = self._create_schedule()
        
    def _create_schedule(self):
        """Create progressive activation schedule for cross-attention stages"""
        schedule = []
        epochs_per_stage = (self.total_epochs - self.warmup_epochs) // (self.num_stages - 1)
        
        for i in range(self.num_stages):
            if i == 0:
                # First stage always active (intra-scale)
                schedule.append(0)
            else:
                # Progressive activation of cross-scale connections
                schedule.append(self.warmup_epochs + (i - 1) * epochs_per_stage)
        
        return schedule
    
    def get_active_stages(self, epoch):
        """
        Get list of active cross-attention stages for current epoch
        
        Args:
            epoch: Current training epoch
        Returns:
            List of boolean indicating which stages are active
        """
        self.current_epoch = epoch
        active = []
        
        for i, start_epoch in enumerate(self.stage_schedule):
            active.append(epoch >= start_epoch)
        
        return active
    
    def get_cross_attention_weight(self, epoch, stage_idx):
        """
        Get weighting factor for cross-attention at specific stage
        
        Args:
            epoch: Current training epoch
            stage_idx: Index of the stage
        Returns:
            Weight factor (0 to 1)
        """
        start_epoch = self.stage_schedule[stage_idx]
        
        if epoch < start_epoch:
            return 0.0
        
        # Gradual ramp-up over 20 epochs
        ramp_epochs = 20
        if epoch < start_epoch + ramp_epochs:
            return (epoch - start_epoch) / ramp_epochs
        
        return 1.0


def create_enhanced_encoder_modules(embed_dim, depths, num_heads, num_stages=4):
    """
    Factory function to create cross-scale interaction modules for encoder
    
    Args:
        embed_dim: Base embedding dimension
        depths: List of depths for each stage
        num_heads: List of number of heads for each stage
        num_stages: Number of encoder stages
    Returns:
        ModuleList of cross-scale interaction blocks
    """
    cross_scale_modules = nn.ModuleList()
    
    for i in range(1, num_stages):
        dim_current = int(embed_dim * 2 ** i)
        dim_prev = int(embed_dim * 2 ** (i - 1))
        
        module = CrossScaleInteractionBlock(
            dim_current=dim_current,
            dim_prev=dim_prev,
            num_heads=num_heads[i],
            mlp_ratio=4.,
            drop=0.,
            attn_drop=0.1,
            drop_path=0.1
        )
        cross_scale_modules.append(module)
    
    return cross_scale_modules
