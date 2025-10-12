__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', 
                 verbose:bool=False, use_multiscale:bool=False, **kwargs):
        
        super().__init__()
        
        # Enhanced multi-scale configuration
        multi_scale = getattr(configs, 'multi_scale', None)
        self.use_multiscale = multi_scale is not None
        
        if self.use_multiscale:
            # Parse multi-scale options
            if len(multi_scale) == 0:  # --multi_scale (no args) = use all
                scales_to_use = ['small', 'medium', 'large']
            else:  # --multi_scale small large = use only specified
                scales_to_use = multi_scale
            
            print(f'PatchTST Multi-Scale Configuration: {scales_to_use}')
        else:
            print('PatchTST Single-Scale Mode')
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        # Multi-scale patching setup
        if self.use_multiscale:
            # Define all possible scales
            all_scales = {
                'small': (max(patch_len // 2, 4), max(stride // 2, 2)),
                'medium': (patch_len, stride), 
                'large': (min(patch_len * 2, context_window // 4), min(stride * 2, context_window // 8))
            }
            
            # Select only requested scales
            selected_scales = {k: v for k, v in all_scales.items() if k in scales_to_use}
            
            self.patch_sizes = [v[0] for v in selected_scales.values()]
            self.strides = [v[1] for v in selected_scales.values()]
            self.scale_names = list(selected_scales.keys())
            
            print(f'Using scales: {self.scale_names}')
            print(f'Patch sizes: {self.patch_sizes}')
            print(f'Strides: {self.strides}')
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            
            if self.use_multiscale:
                # Multi-scale models for decomposition
                self.models_trend = nn.ModuleList([
                    PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, 
                                    patch_len=p_len, stride=s_len, max_seq_len=max_seq_len, n_layers=n_layers, 
                                    d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, 
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                    key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, 
                                    store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, 
                                    head_dropout=head_dropout, padding_patch=padding_patch, 
                                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                                    revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs)
                    for p_len, s_len in zip(self.patch_sizes, self.strides)
                ])
                
                self.models_res = nn.ModuleList([
                    PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, 
                                    patch_len=p_len, stride=s_len, max_seq_len=max_seq_len, n_layers=n_layers, 
                                    d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, 
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                    key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, 
                                    store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, 
                                    head_dropout=head_dropout, padding_patch=padding_patch, 
                                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                                    revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs)
                    for p_len, s_len in zip(self.patch_sizes, self.strides)
                ])
                
                # Fusion layers for multi-scale features
                self.fusion_trend = nn.Linear(len(self.patch_sizes) * target_window, target_window)
                self.fusion_res = nn.Linear(len(self.patch_sizes) * target_window, target_window)
            else:
                # Original single-scale models
                self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, verbose=verbose, **kwargs)
                self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            if self.use_multiscale:
                # Multi-scale single model
                self.models = nn.ModuleList([
                    PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, 
                                    patch_len=p_len, stride=s_len, max_seq_len=max_seq_len, n_layers=n_layers, 
                                    d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, 
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                    key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, 
                                    store_attn=store_attn, pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, 
                                    head_dropout=head_dropout, padding_patch=padding_patch, 
                                    pretrain_head=pretrain_head, head_type=head_type, individual=individual, 
                                    revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs)
                    for p_len, s_len in zip(self.patch_sizes, self.strides)
                ])
                
                # Fusion layer
                self.fusion = nn.Linear(len(self.patch_sizes) * target_window, target_window)
            else:
                # Original single model
                self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                      max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                      n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                      dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                      attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                      pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                                      pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                      subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            # Use transpose + contiguous for better performance than permute
            res_init = res_init.transpose(1, 2).contiguous()  # [Batch, Channel, Input length]
            trend_init = trend_init.transpose(1, 2).contiguous()  # [Batch, Channel, Input length]
            
            if self.use_multiscale:
                # Multi-scale processing for decomposition
                trend_outputs = []
                res_outputs = []
                
                for model_trend, model_res in zip(self.models_trend, self.models_res):
                    trend_out = model_trend(trend_init)  # [Batch, Channel, target_window]
                    res_out = model_res(res_init)        # [Batch, Channel, target_window]
                    
                    trend_outputs.append(trend_out)
                    res_outputs.append(res_out)
                
                # Concatenate multi-scale outputs
                trend_concat = torch.cat(trend_outputs, dim=2)  # [Batch, Channel, target_window * n_scales]
                res_concat = torch.cat(res_outputs, dim=2)      # [Batch, Channel, target_window * n_scales]
                
                # Fuse multi-scale features
                trend = self.fusion_trend(trend_concat)  # [Batch, Channel, target_window]
                res = self.fusion_res(res_concat)        # [Batch, Channel, target_window]
            else:
                # Original single-scale processing
                trend = self.model_trend(trend_init)
                res = self.model_res(res_init)
            
            x = res + trend
            x = x.transpose(1, 2).contiguous()    # [Batch, target_window, Channel]
        else:
            x = x.transpose(1, 2).contiguous()    # [Batch, Channel, Input length]
            
            if self.use_multiscale:
                # Multi-scale processing
                outputs = []
                
                for model in self.models:
                    out = model(x)  # [Batch, Channel, target_window]
                    outputs.append(out)
                
                # Concatenate and fuse multi-scale outputs
                x_concat = torch.cat(outputs, dim=2)  # [Batch, Channel, target_window * n_scales]
                x = self.fusion(x_concat)             # [Batch, Channel, target_window]
            else:
                # Original single processing
                x = self.model(x)
                
            x = x.transpose(1, 2).contiguous()    # [Batch, target_window, Channel]
        return x