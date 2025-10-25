import os
import pdb

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel
from model.video_transformer import SpaceTimeTransformer
from utils.util import state_dict_data_parallel_fix

class FrozenInTime(BaseModel):
    def __init__(self,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros'):
        super().__init__()

        self.video_params = video_params
        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        # pdb.set_trace()
        if self.text_params['model'].startswith('distilbert'):
            self.text_model = AutoModel.from_pretrained('distilbert-base-uncased',
                   cache_dir='pretrained/distilbert-base-uncased')
        else:
            self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()

        pretrained = video_params['pretrained']
        if video_params['model'] == "SpaceTimeTransformer":
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            vit_init = video_params.get('vit_init', 'imagenet-21k')
            if arch_config == 'base_patch16_224':
                # you can download the checkpoint via wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
                # vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=pretrained)
                vit_model = torch.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                model = SpaceTimeTransformer(num_frames=num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style)
            else:
                raise NotImplementedError

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                # vit_checkpoint = vit_model.state_dict()
                # model.load_state_dict(vit_checkpoint, strict=False)
                vit_checkpoint = vit_model
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
                model.load_state_dict(new_vit_dict, strict=False)
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # for backwards compatibility (old models)
        self.video_model.fc = nn.Identity()

        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(nn.ReLU(),
                                     nn.Linear(self.text_model.config.hidden_size, projection_dim),
                                     )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            # checkpoint = torch.load(load_checkpoint)
            local_rank = int(os.environ['LOCAL_RANK'])  # fixed by qinghong.
            checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(local_rank))
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

    def set_device(self, device):
        self.device = device

    def forward(self, data, video_only=False, return_embeds=True):
        if video_only:
            video_data = data['video']
            video_embeddings = self.compute_video(video_data)
            return video_embeddings

        text_data = data['text']
        video_data = data['video']

        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']    # not implement for bert
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class MultiScaleVideoEncoder(nn.Module):
    """
    Multi-scale video encoder that processes video clips at 3 different frame counts.
    
    Inherits from the base video encoder architecture and processes clips sequentially
    at 4, 8, and 16 frame scales, then combines features using learnable fusion weights.
    """
    
    def __init__(self, video_params, projection_dim=768):
        super().__init__()
        
        self.video_params = video_params
        self.projection_dim = projection_dim
        self.scales = [4, 8, 16]
        
        # Create video encoders for each scale
        self.encoders = nn.ModuleDict()
        
        for num_frames in self.scales:
            # Create SpaceTimeTransformer for this scale
            if video_params['model'] == "SpaceTimeTransformer":
                time_init = video_params.get('time_init', 'zeros')
                attention_style = video_params.get('attention_style', 'frozen-in-time')
                arch_config = video_params.get('arch_config', 'base_patch16_224')
                
                if arch_config == 'base_patch16_224':
                    # Load pretrained weights
                    vit_model = torch.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                               time_init=time_init,
                                               attention_style=attention_style)
                    model.head = nn.Identity()
                    model.pre_logits = nn.Identity()
                    
                    # Load pretrained weights
                    new_vit_dict = state_dict_data_parallel_fix(vit_model, model.state_dict())
                    model.load_state_dict(new_vit_dict, strict=False)
                    
                    self.encoders[f'encoder_{num_frames}f'] = model
                else:
                    raise NotImplementedError(f"Architecture {arch_config} not implemented")
            else:
                raise NotImplementedError(f"Video model {video_params['model']} not implemented")
        
        # Get feature dimension from the first encoder
        self.feature_dim = self.encoders[f'encoder_{self.scales[0]}f'].embed_dim
        
        # Projection layers for each scale to ensure consistent output dimension
        self.projectors = nn.ModuleDict()
        for num_frames in self.scales:
            self.projectors[f'proj_{num_frames}f'] = nn.Linear(self.feature_dim, projection_dim)
        
        # Learnable fusion weights initialized as [0.33, 0.33, 0.33]
        self.fusion_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))
        
    def forward(self, video_clips):
        """
        Memory-optimized forward pass processing clips at multiple scales sequentially.
        
        Optimizations for 4x RTX 3090 (24GB each):
        1. Sequential processing (fine -> medium -> coarse) to avoid OOM
        2. Gradient checkpointing on 16-frame scale only (longest sequence)
        3. Mixed precision with careful NaN/Inf monitoring
        4. Explicit memory cleanup between scales
        
        Args:
            video_clips: Dict containing video tensors for different scales
                        Keys: 'frames_4', 'frames_8', 'frames_16'
                        Values: Tensors of shape [batch_size, num_frames, C, H, W]
        
        Returns:
            v_fused: Fused video features of shape [batch_size, projection_dim]
        """
        scale_features = []
        
        # Process each scale sequentially: fine -> medium -> coarse for memory efficiency
        for i, num_frames in enumerate(self.scales):
            clip_key = f'frames_{num_frames}'
            encoder_key = f'encoder_{num_frames}f'
            projector_key = f'proj_{num_frames}f'
            
            if clip_key in video_clips:
                video_clip = video_clips[clip_key]  # [batch_size, num_frames, C, H, W]
                
                # Use mixed precision with autocast context
                with torch.cuda.amp.autocast(enabled=True):
                    # Apply gradient checkpointing only to the coarse scale (16 frames)
                    # This is the longest sequence and most memory intensive
                    if num_frames == 16 and self.training:
                        # Gradient checkpointing wrapper for memory efficiency
                        def checkpoint_forward(video_input):
                            features = self.encoders[encoder_key](video_input)
                            return self.projectors[projector_key](features)
                        
                        # Use gradient checkpointing for the 16-frame scale
                        projected_features = torch.utils.checkpoint.checkpoint(
                            checkpoint_forward, 
                            video_clip,
                            use_reentrant=False  # More memory efficient
                        )
                    else:
                        # Regular forward pass for 4-frame and 8-frame scales
                        features = self.encoders[encoder_key](video_clip)  # [batch_size, feature_dim]
                        projected_features = self.projectors[projector_key](features)  # [batch_size, projection_dim]
                
                # Check for NaN/Inf values in mixed precision training
                if torch.isnan(projected_features).any() or torch.isinf(projected_features).any():
                    print(f"Warning: NaN/Inf detected in {num_frames}-frame scale features!")
                    # Replace NaN/Inf with zeros to prevent training collapse
                    projected_features = torch.nan_to_num(projected_features, nan=0.0, posinf=0.0, neginf=0.0)
                
                scale_features.append(projected_features)
                
                # Explicit memory cleanup after processing each scale
                # Clear intermediate activations to free GPU memory
                if i < len(self.scales) - 1:  # Don't clear after last scale
                    if 'features' in locals():
                        del features
                    torch.cuda.empty_cache()  # Clear GPU cache between scales if needed
                    
            else:
                # If scale not provided, create zero tensor
                batch_size = list(video_clips.values())[0].shape[0]
                zero_features = torch.zeros(
                    batch_size, self.projection_dim, 
                    device=self.fusion_weights.device,
                    dtype=torch.float16 if torch.cuda.amp.is_autocast_enabled() else torch.float32
                )
                scale_features.append(zero_features)
        
        # Apply softmax to fusion weights before combining
        # Use mixed precision for fusion computation as well
        with torch.cuda.amp.autocast(enabled=True):
            normalized_weights = F.softmax(self.fusion_weights, dim=0)
            
            # Combine scale features: v_fused = w[0]*v_fine + w[1]*v_medium + w[2]*v_coarse
            v_fused = sum(w * features for w, features in zip(normalized_weights, scale_features))
            
            # Final NaN/Inf check on fused features
            if torch.isnan(v_fused).any() or torch.isinf(v_fused).any():
                print("Warning: NaN/Inf detected in fused features!")
                v_fused = torch.nan_to_num(v_fused, nan=0.0, posinf=0.0, neginf=0.0)
        
        return v_fused
    
    def get_fusion_weights(self):
        """Return the current fusion weights after softmax normalization."""
        return F.softmax(self.fusion_weights, dim=0)
    
    def get_memory_stats(self):
        """
        Get current GPU memory statistics for monitoring.
        
        Returns:
            dict: Memory statistics including allocated, reserved, and free memory
        """
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            }
        return {}
    
    def enable_memory_efficient_mode(self, enable=True):
        """
        Enable/disable memory efficient mode.
        
        Args:
            enable (bool): Whether to enable memory efficient processing
        """
        self.memory_efficient = enable
        
        # Set gradient checkpointing for all encoders if enabled
        for encoder in self.encoders.values():
            if hasattr(encoder, 'gradient_checkpointing'):
                encoder.gradient_checkpointing = enable
    
    def force_memory_cleanup(self):
        """
        Force GPU memory cleanup. Use with caution during training.
        """
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    def get_scale_memory_usage(self, video_clips):
        """
        Estimate memory usage for each scale without forward pass.
        
        Args:
            video_clips: Input video clips dict
            
        Returns:
            dict: Estimated memory usage per scale in GB
        """
        memory_usage = {}
        
        for num_frames in self.scales:
            clip_key = f'frames_{num_frames}'
            if clip_key in video_clips:
                video_clip = video_clips[clip_key]
                batch_size, frames, channels, height, width = video_clip.shape
                
                # Estimate memory usage (rough calculation)
                input_memory = video_clip.numel() * video_clip.element_size() / 1024**3
                
                # Rough estimate of activation memory (depends on architecture)
                # SpaceTimeTransformer has attention matrices that scale with sequence length
                sequence_length = frames * (height // 16) * (width // 16)  # Patch-based
                attention_memory = batch_size * sequence_length * sequence_length * 4 / 1024**3  # FP32
                
                total_memory = input_memory + attention_memory * 2  # Forward + backward
                memory_usage[f'{num_frames}_frames'] = total_memory
        
        return memory_usage


def memory_efficient_forward_wrapper(model_func):
    """
    Decorator for memory-efficient forward passes with automatic cleanup.
    
    Args:
        model_func: The model's forward function to wrap
        
    Returns:
        Wrapped function with memory management
    """
    def wrapper(*args, **kwargs):
        # Record initial memory
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            # Execute forward pass
            result = model_func(*args, **kwargs)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"GPU OOM detected in {model_func.__name__}!")
                print(f"Initial memory: {initial_memory / 1024**3:.2f} GB")
                print(f"Current memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
                # Force cleanup and retry with gradient checkpointing
                torch.cuda.empty_cache()
                raise e
            else:
                raise e
        finally:
            # Cleanup after forward pass
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                memory_increase = (current_memory - initial_memory) / 1024**3
                if memory_increase > 1.0:  # If memory increased by more than 1GB
                    torch.cuda.empty_cache()
    
    return wrapper


class MemoryManager:
    """
    Utility class for managing GPU memory during multi-scale training.
    """
    
    def __init__(self, device='cuda', memory_fraction=0.9):
        self.device = device
        self.memory_fraction = memory_fraction
        self.peak_memory = 0
        
    def get_memory_info(self):
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        self.peak_memory = max(self.peak_memory, allocated)
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'peak_gb': self.peak_memory,
            'utilization_pct': (allocated / total) * 100
        }
    
    def check_memory_available(self, required_gb):
        """Check if sufficient memory is available."""
        info = self.get_memory_info()
        available = info.get('free_gb', 0)
        return available >= required_gb
    
    def force_cleanup(self):
        """Force memory cleanup and garbage collection."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def adaptive_batch_size(self, base_batch_size, current_scale_frames):
        """
        Adaptively adjust batch size based on available memory and scale.
        
        Args:
            base_batch_size: Base batch size for 4-frame scale
            current_scale_frames: Number of frames in current scale
            
        Returns:
            Adjusted batch size
        """
        info = self.get_memory_info()
        available_memory = info.get('free_gb', 0)
        
        # Scale factor based on frame count (linear approximation)
        memory_scale_factor = current_scale_frames / 4.0
        
        # Adjust batch size based on available memory
        if available_memory < 5.0:  # Less than 5GB available
            scale_factor = 0.5
        elif available_memory < 10.0:  # Less than 10GB available
            scale_factor = 0.75
        else:
            scale_factor = 1.0
        
        adjusted_batch_size = max(1, int(base_batch_size * scale_factor / memory_scale_factor))
        return adjusted_batch_size
    
    def monitor_training_memory(self, epoch, step, log_frequency=100):
        """
        Monitor memory usage during training.
        
        Args:
            epoch: Current epoch
            step: Current step
            log_frequency: Log every N steps
        """
        if step % log_frequency == 0:
            info = self.get_memory_info()
            print(f"Epoch {epoch}, Step {step} - Memory: "
                  f"{info['allocated_gb']:.1f}GB allocated, "
                  f"{info['utilization_pct']:.1f}% utilization, "
                  f"Peak: {info['peak_gb']:.1f}GB")
            
            # Warning if memory usage is high
            if info['utilization_pct'] > 85:
                print(f"⚠️  High memory usage detected: {info['utilization_pct']:.1f}%")


class MultiScaleFrozenInTime(FrozenInTime):
    """
    Enhanced EgoVLP model with multi-scale temporal modeling.
    
    Processes videos at 3 scales: 4 frames (fine), 8 frames (medium), 16 frames (coarse)
    and learns fusion weights to combine multi-scale features.
    """
    
    def __init__(self, 
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal',
                 load_temporal_fix='zeros',
                 multi_scale_frames=[4, 8, 16],
                 fusion_type='learned_weighted'):
        
        # Initialize base model with largest frame count for maximum capacity
        max_frames = max(multi_scale_frames)
        video_params_max = video_params.copy()
        video_params_max['num_frames'] = max_frames
        
        super().__init__(video_params_max, text_params, projection_dim, 
                        load_checkpoint, projection, load_temporal_fix)
        
        self.multi_scale_frames = multi_scale_frames
        self.fusion_type = fusion_type
        self.num_scales = len(multi_scale_frames)
        
        # Create video models for each scale
        self.video_models = nn.ModuleDict()
        self.vid_projs = nn.ModuleDict()
        
        for i, num_frames in enumerate(multi_scale_frames):
            scale_name = f'scale_{i}_{num_frames}f'
            
            # Create video model for this scale
            if video_params['model'] == "SpaceTimeTransformer":
                time_init = video_params.get('time_init', 'zeros')
                attention_style = video_params.get('attention_style', 'frozen-in-time')
                arch_config = video_params.get('arch_config', 'base_patch16_224')
                
                if arch_config == 'base_patch16_224':
                    vit_model = torch.load("pretrained/jx_vit_base_p16_224-80ecf9dd.pth", map_location="cpu")
                    model = SpaceTimeTransformer(num_frames=num_frames,
                                               time_init=time_init,
                                               attention_style=attention_style)
                    model.head = nn.Identity()
                    model.pre_logits = nn.Identity()
                    ftr_dim = model.embed_dim
                    
                    # Load pretrained weights
                    if load_checkpoint in ["", None]:
                        new_vit_dict = state_dict_data_parallel_fix(vit_model, model.state_dict())
                        model.load_state_dict(new_vit_dict, strict=False)
                    
                    self.video_models[scale_name] = model
                    
                    # Create projection layer for this scale
                    if projection == 'minimal':
                        vid_proj = nn.Sequential(nn.Linear(ftr_dim, projection_dim))
                    else:
                        vid_proj = nn.Identity()
                    
                    self.vid_projs[scale_name] = vid_proj
        
        # Fusion mechanism
        if fusion_type == 'learned_weighted':
            # Learnable fusion weights
            self.fusion_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.fusion_attention = nn.MultiheadAttention(projection_dim, num_heads=8, batch_first=True)
            self.scale_embeddings = nn.Parameter(torch.randn(self.num_scales, projection_dim))
        
        # Remove original video model and projection as we use multi-scale versions
        del self.video_model
        del self.vid_proj
    
    def compute_video(self, video_data):
        """
        Compute video embeddings at multiple temporal scales and fuse them.
        
        Args:
            video_data: Dict containing video tensors at different scales
                       Keys: 'video_4f', 'video_8f', 'video_16f'
        
        Returns:
            fused_embeddings: Tensor of shape [batch_size, projection_dim]
        """
        scale_embeddings = []
        
        for i, num_frames in enumerate(self.multi_scale_frames):
            scale_name = f'scale_{i}_{num_frames}f'
            video_key = f'video_{num_frames}f'
            
            if video_key in video_data:
                # Process video at this scale
                video_emb = self.video_models[scale_name](video_data[video_key])
                video_emb = self.vid_projs[scale_name](video_emb)
                scale_embeddings.append(video_emb)
            else:
                # If specific scale not provided, use temporal subsampling from largest scale
                full_video = video_data['video']  # Assume this has max frames
                # Temporal subsampling to get desired number of frames
                if full_video.shape[1] >= num_frames:  # [B, T, C, H, W]
                    indices = torch.linspace(0, full_video.shape[1] - 1, num_frames).long()
                    subsampled_video = full_video[:, indices]
                    video_emb = self.video_models[scale_name](subsampled_video)
                    video_emb = self.vid_projs[scale_name](video_emb)
                    scale_embeddings.append(video_emb)
        
        # Fuse multi-scale embeddings
        if self.fusion_type == 'learned_weighted':
            # Weighted sum with learnable weights
            weights = F.softmax(self.fusion_weights, dim=0)
            fused_embeddings = sum(w * emb for w, emb in zip(weights, scale_embeddings))
            
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            # Stack embeddings: [batch_size, num_scales, projection_dim]
            stacked_embeddings = torch.stack(scale_embeddings, dim=1)
            
            # Add scale embeddings
            batch_size = stacked_embeddings.shape[0]
            scale_embs = self.scale_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            stacked_embeddings = stacked_embeddings + scale_embs
            
            # Apply attention to fuse
            fused_embeddings, _ = self.fusion_attention(
                stacked_embeddings, stacked_embeddings, stacked_embeddings
            )
            # Global average pooling across scales
            fused_embeddings = fused_embeddings.mean(dim=1)
        
        else:
            # Simple average fusion as fallback
            fused_embeddings = torch.stack(scale_embeddings).mean(dim=0)
        
        return fused_embeddings
    
    def forward(self, data, video_only=False, return_embeds=True, return_multi_scale=False):
        """
        Forward pass with multi-scale processing.
        
        Args:
            data: Input data dict
            video_only: If True, only compute video embeddings
            return_embeds: If True, return embeddings instead of similarity matrix
            return_multi_scale: If True, return individual scale embeddings for consistency loss
        
        Returns:
            If return_multi_scale=True: text_embeddings, fused_video_embeddings, scale_video_embeddings
            Else: text_embeddings, fused_video_embeddings (or similarity matrix)
        """
        if video_only:
            video_data = data['video'] if isinstance(data['video'], dict) else {'video': data['video']}
            if return_multi_scale:
                # Return individual scale embeddings for temporal consistency loss
                scale_embeddings = []
                for i, num_frames in enumerate(self.multi_scale_frames):
                    scale_name = f'scale_{i}_{num_frames}f'
                    video_key = f'video_{num_frames}f'
                    
                    if video_key in video_data:
                        video_emb = self.video_models[scale_name](video_data[video_key])
                        video_emb = self.vid_projs[scale_name](video_emb)
                        scale_embeddings.append(video_emb)
                
                fused_embeddings = self.compute_video(video_data)
                return fused_embeddings, scale_embeddings
            else:
                return self.compute_video(video_data)

        text_data = data['text']
        video_data = data['video'] if isinstance(data['video'], dict) else {'video': data['video']}

        text_embeddings = self.compute_text(text_data)
        
        if return_multi_scale:
            # Return individual scale embeddings for temporal consistency loss
            scale_embeddings = []
            for i, num_frames in enumerate(self.multi_scale_frames):
                scale_name = f'scale_{i}_{num_frames}f'
                video_key = f'video_{num_frames}f'
                
                if video_key in video_data:
                    video_emb = self.video_models[scale_name](video_data[video_key])
                    video_emb = self.vid_projs[scale_name](video_emb)
                    scale_embeddings.append(video_emb)
            
            fused_video_embeddings = self.compute_video(video_data)
            
            if return_embeds:
                return text_embeddings, fused_video_embeddings, scale_embeddings
            else:
                return sim_matrix(text_embeddings, fused_video_embeddings), scale_embeddings
        else:
            video_embeddings = self.compute_video(video_data)
            
            if return_embeds:
                return text_embeddings, video_embeddings
            else:
                return sim_matrix(text_embeddings, video_embeddings)

if __name__ == "__main__":
    pass
