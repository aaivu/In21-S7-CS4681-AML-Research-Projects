"""
Enhanced nnFormer Architecture for BraTS 2021 Brain Tumor Segmentation

This module implements the enhanced nnFormer with multi-scale cross-attention
mechanisms for improved 3D medical image segmentation.

Key Features:
- Multi-scale cross-attention between encoder stages
- Adaptive feature fusion with learned weights
- Progressive training strategy
- Enhanced for BraTS 2021 dataset (4 modalities, 3 tumor regions)
"""

from einops import rearrange
from copy import deepcopy
from nnformer.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

# Import enhanced modules
from nnformer.network_architecture.enhanced_modules import (
    MultiScaleCrossAttention,
    AdaptiveFeatureFusion,
    CrossScaleInteractionBlock,
    SpatialAlignmentModule,
    ProgressiveTrainingController,
    create_enhanced_encoder_modules
)

# Import base modules from nnFormer_acdc
from nnformer.network_architecture.nnFormer_acdc import (
    ContiguousGrad,
    Mlp,
    window_partition,
    window_reverse,
    WindowAttention,
    SwinTransformerBlock,
    PatchMerging,
    Patch_Expanding,
    PatchEmbed
)


class EnhancedBasicLayer(nn.Module):
    """
    Enhanced Basic Layer with Cross-Scale Interaction
    
    Extends the standard BasicLayer to include cross-attention mechanisms
    for multi-scale feature interaction.
    """
    
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True,
                 i_layer=None,
                 use_cross_attention=False,
                 cross_attn_module=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.depth = depth
        self.i_layer = i_layer
        self.use_cross_attention = use_cross_attention
        self.cross_attn_module = cross_attn_module
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        # Patch merging layer
        if downsample is not None:
            if i_layer == 1:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=1)
            elif i_layer == 2:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=2)
            elif i_layer == 0:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=0)
            else:
                self.downsample = None
        else:
            self.downsample = None
    
    def forward(self, x, S, H, W, x_prev=None, cross_attn_weight=1.0):
        """
        Args:
            x: Current stage features (B, L, C)
            S, H, W: Spatial dimensions
            x_prev: Previous stage features for cross-attention (B, L_prev, C_prev)
            cross_attn_weight: Weight for cross-attention (0 to 1)
        """
        attn_mask = None
        
        # Apply self-attention blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)
        
        # Apply cross-attention if enabled and previous features provided
        if self.use_cross_attention and self.cross_attn_module is not None and x_prev is not None:
            if cross_attn_weight > 0:
                x_cross = self.cross_attn_module(x, x_prev)
                x = x + cross_attn_weight * x_cross
        
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            if self.i_layer != 1 and self.i_layer != 2:
                Ws, Wh, Ww = S, (H + 1) // 2, (W + 1) // 2
            else:
                Ws, Wh, Ww = S // 2, (H + 1) // 2, (W + 1) // 2
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            return x, S, H, W, x, S, H, W


class EnhancedEncoder(nn.Module):
    """
    Enhanced Encoder with Multi-Scale Cross-Attention
    
    Extends the baseline encoder with cross-attention mechanisms between
    consecutive stages for improved multi-scale feature interaction.
    """
    
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=4,  # 4 modalities for BraTS
                 embed_dim=192,  # Increased from 96 to 192
                 depths=[2, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],  # Doubled for richer representations
                 window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
                 down_stride=[[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 enable_cross_attention=True):
        super().__init__()
        
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.enable_cross_attention = enable_cross_attention
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Create cross-attention modules
        self.cross_scale_modules = nn.ModuleList()
        if enable_cross_attention:
            for i in range(1, self.num_layers):
                dim_current = int(embed_dim * 2 ** i)
                dim_prev = int(embed_dim * 2 ** (i - 1))
                
                cross_attn = CrossScaleInteractionBlock(
                    dim_current=dim_current,
                    dim_prev=dim_prev,
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.1
                )
                self.cross_scale_modules.append(cross_attn)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            use_cross_attn = enable_cross_attention and i_layer > 0
            cross_attn_module = self.cross_scale_modules[i_layer - 1] if use_cross_attn else None
            
            layer = EnhancedBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    pretrain_img_size[0] // down_stride[i_layer][0],
                    pretrain_img_size[1] // down_stride[i_layer][1],
                    pretrain_img_size[2] // down_stride[i_layer][2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                i_layer=i_layer,
                use_cross_attention=use_cross_attn,
                cross_attn_module=cross_attn_module
            )
            self.layers.append(layer)
        
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        
        # Add norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
    
    def forward(self, x, cross_attn_weights=None):
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            cross_attn_weights: List of weights for cross-attention at each stage
        Returns:
            List of multi-scale features
        """
        if cross_attn_weights is None:
            cross_attn_weights = [1.0] * (self.num_layers - 1)
        
        x = self.patch_embed(x)
        down = []
        features_for_cross_attn = []
        
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.pos_drop(x)
        
        for i in range(self.num_layers):
            layer = self.layers[i]
            
            # Get previous stage features for cross-attention
            x_prev = features_for_cross_attn[-1] if len(features_for_cross_attn) > 0 else None
            cross_weight = cross_attn_weights[i - 1] if i > 0 else 1.0
            
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww, x_prev, cross_weight)
            
            # Store features for cross-attention
            features_for_cross_attn.append(x_out)
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out_norm = norm_layer(x_out)
                out = x_out_norm.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                down.append(out)
        
        return down


# Import Decoder from base implementation (can be enhanced similarly if needed)
from nnformer.network_architecture.nnFormer_acdc import Decoder, final_patch_expanding


class EnhancednnFormer(SegmentationNetwork):
    """
    Enhanced nnFormer for BraTS 2021 Brain Tumor Segmentation
    
    Features:
    - Multi-scale cross-attention in encoder
    - 4 input modalities (T1, T1ce, T2, FLAIR)
    - 3 output classes (ET, TC, WT) + background
    - Enhanced feature dimensions (192 base)
    """
    
    def __init__(self,
                 crop_size=[64, 128, 128],
                 embedding_dim=192,  # Increased for better representation
                 input_channels=4,  # 4 MRI modalities
                 num_classes=4,  # Background + 3 tumor regions
                 conv_op=nn.Conv3d,
                 depths=[2, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],
                 patch_size=[1, 4, 4],
                 window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
                 down_stride=[[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]],
                 deep_supervision=True,
                 enable_cross_attention=True,
                 enable_adaptive_fusion=False,
                 enable_enhanced_training=False):
        super(EnhancednnFormer, self).__init__()
        
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.enable_cross_attention = enable_cross_attention
        self.enable_adaptive_fusion = enable_adaptive_fusion
        self.enable_enhanced_training = enable_enhanced_training
        
        self.upscale_logits_ops = []
        self.upscale_logits_ops.append(lambda x: x)
        
        embed_dim = embedding_dim
        
        # Enhanced encoder with cross-attention
        self.model_down = EnhancedEncoder(
            pretrain_img_size=crop_size,
            window_size=window_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            in_chans=input_channels,
            down_stride=down_stride,
            enable_cross_attention=enable_cross_attention
        )
        
        # Decoder (using base implementation)
        self.decoder = Decoder(
            pretrain_img_size=crop_size,
            embed_dim=embed_dim,
            window_size=window_size[::-1][1:],
            patch_size=patch_size,
            num_heads=num_heads[::-1][1:],
            depths=depths[::-1][1:],
            up_stride=down_stride[::-1][1:]
        )
        
        # Adaptive feature fusion (optional)
        if enable_adaptive_fusion:
            feature_dims = [int(embed_dim * 2 ** i) for i in range(len(depths))]
            self.adaptive_fusion = AdaptiveFeatureFusion(feature_dims)
        else:
            self.adaptive_fusion = None
        
        # Progressive training controller
        if enable_enhanced_training:
            self.training_controller = ProgressiveTrainingController(
                num_stages=len(depths),
                warmup_epochs=50,
                total_epochs=1000
            )
        else:
            self.training_controller = None
        
        # Final output heads
        self.final = []
        if self.do_ds:
            for i in range(len(depths) - 1):
                self.final.append(final_patch_expanding(embed_dim * 2 ** i, num_classes, patch_size=patch_size))
        else:
            self.final.append(final_patch_expanding(embed_dim, num_classes, patch_size=patch_size))
        
        self.final = nn.ModuleList(self.final)
        
        # Auxiliary supervision heads for intermediate stages (if enhanced training)
        if enable_enhanced_training:
            self.auxiliary_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(int(embed_dim * 2 ** i), num_classes, 1),
                    nn.Upsample(scale_factor=2 ** i, mode='trilinear', align_corners=True)
                ) for i in range(1, len(depths))
            ])
        else:
            self.auxiliary_heads = None
    
    def forward(self, x, epoch=None):
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            epoch: Current training epoch (for progressive training)
        Returns:
            Segmentation output(s)
        """
        # Get cross-attention weights based on training progress
        cross_attn_weights = None
        if self.training_controller is not None and epoch is not None:
            cross_attn_weights = [
                self.training_controller.get_cross_attention_weight(epoch, i)
                for i in range(1, len(self.model_down.layers))
            ]
        
        # Encoder forward pass
        skips = self.model_down(x, cross_attn_weights)
        neck = skips[-1]
        
        # Optional adaptive fusion
        if self.adaptive_fusion is not None:
            neck_fused, _ = self.adaptive_fusion([neck], target_size=neck.shape[-3:])
            neck = neck_fused
        
        # Decoder forward pass
        out = self.decoder(neck, skips)
        
        # Generate outputs
        seg_outputs = []
        if self.do_ds:
            for i in range(len(out)):
                seg_outputs.append(self.final[-(i + 1)](out[i]))
            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out[-1]))
            return seg_outputs[-1]
    
    def get_auxiliary_outputs(self, encoder_features):
        """
        Get auxiliary outputs from intermediate encoder stages
        (for enhanced training with multi-scale supervision)
        """
        if self.auxiliary_heads is None:
            return None
        
        aux_outputs = []
        for i, feat in enumerate(encoder_features[1:]):
            aux_out = self.auxiliary_heads[i](feat)
            aux_outputs.append(aux_out)
        
        return aux_outputs


def create_enhanced_nnformer_brats(enable_cross_attention=True,
                                    enable_adaptive_fusion=False,
                                    enable_enhanced_training=False):
    """
    Factory function to create Enhanced nnFormer for BraTS 2021
    
    Args:
        enable_cross_attention: Enable multi-scale cross-attention
        enable_adaptive_fusion: Enable adaptive feature fusion
        enable_enhanced_training: Enable enhanced training strategy
    Returns:
        Enhanced nnFormer model
    """
    model = EnhancednnFormer(
        crop_size=[64, 128, 128],
        embedding_dim=192,
        input_channels=4,
        num_classes=4,
        conv_op=nn.Conv3d,
        depths=[2, 2, 2, 2],
        num_heads=[6, 12, 24, 48],
        patch_size=[1, 4, 4],
        window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
        down_stride=[[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]],
        deep_supervision=True,
        enable_cross_attention=enable_cross_attention,
        enable_adaptive_fusion=enable_adaptive_fusion,
        enable_enhanced_training=enable_enhanced_training
    )
    
    return model
