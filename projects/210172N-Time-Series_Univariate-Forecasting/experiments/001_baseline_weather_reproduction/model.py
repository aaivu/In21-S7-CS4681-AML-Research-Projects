
# Copied from C:\Users\Anushanga\Desktop\Projects\In21-S7-CS4681-AML-Research-Projects\projects\210172N-Time-Series_Univariate-Forecasting\data\PatchTST\PatchTST_supervised\layers\RevIN.py

import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

# Copied from C:\Users\Anushanga\Desktop\Projects\In21-S7-CS4681-AML-Research-Projects\projects\210172N-Time-Series_Univariate-Forecasting\data\PatchTST\PatchTST_supervised\layers\PatchTST_layers.py

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class TSTiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.W_P = nn.Linear(patch_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.pe = pe
        self.learn_pe = learn_pe
        if pe == 'zeros':
            self.W_pos = nn.Parameter(torch.zeros(1, patch_num, d_model))
        elif pe == 'sincos':
            self.W_pos = nn.Parameter(torch.zeros(1, patch_num, d_model), requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(c_in, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> torch.Tensor:
        n_vars = x.shape[1]
        x = self.ln(self.W_P(x))
        x = self.dropout(x + self.W_pos)
        bs, n_vars, patch_num, d_model = x.shape
        x = x.reshape(bs * n_vars, patch_num, d_model)
        x, attns = self.encoder(x)
        x = x.reshape(bs, n_vars, patch_num, d_model)
        return x, attns

class TSTEncoder(nn.Module):
    def __init__(self, c_in, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0.1, dropout=0.1, activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer(c_in, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                                      norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: torch.Tensor, key_padding_mask: bool = None, attn_mask: bool = None):
        output = src
        attns = []
        if self.res_attention:
            for mod in self.layers:
                output, attn = mod(output, prev_attn=attns[-1] if len(attns) > 0 else None)
                attns.append(attn)
            return output, attns
        else:
            for mod in self.layers:
                output, attn = mod(output)
                if attn is not None:
                    attns.append(attn)
            return output, attns

class TSTEncoderLayer(nn.Module):
    def __init__(self, c_in, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0.1, bias=True, activation="gelu",
                 res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.dropout_attn = nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                self.get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        self.dropout_ffn = nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: torch.Tensor, prev_attn: torch.Tensor = None, key_padding_mask: bool = None, attn_mask: bool = None):
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev_attn=prev_attn, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn, scores = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
            self.scores = scores
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src, attn

    def get_activation_fn(self, activation):
        if activation == "relu": return nn.ReLU()
        elif activation == "gelu": return nn.GELU()
        raise ValueError(f'{activation} is not available. Please use "relu" or "gelu"')

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_k, attn_dropout, res_attention=res_attention, lsa=lsa)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, prev_attn: torch.Tensor = None,
                key_padding_mask: bool = None, attn_mask: bool = None):
        bs = Q.size(0)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev_attn, key_padding_mask, attn_mask)
        else:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, key_padding_mask, attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        return output, attn_weights, attn_scores

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.d_k = d_k
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        self.lsa = lsa

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, prev_attn: torch.Tensor = None,
                key_padding_mask: bool = None, attn_mask: bool = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if self.res_attention and prev_attn is not None:
            scores += prev_attn
        if key_padding_mask is not None:
            scores.masked_fill_(key_padding_mask, -np.inf)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -np.inf)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights, scores

# Copied from C:\Users\Anushanga\Desktop\Projects\In21-S7-CS4681-AML-Research-Projects\projects\210172N-Time-Series_Univariate-Forecasting\data\PatchTST\PatchTST_supervised\layers\PatchTST_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:int=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:int=None, d_v:int=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:str='auto',
                 padding_var:int=None, attn_mask:torch.Tensor=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout=0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True,
                 subtract_last = False, verbose:bool=False, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head for pretraining
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)


    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]

        # model
        if self.individual:
            z_out = []
            for i in range(self.n_vars):
                z_in = z[:,i,:,:].unsqueeze(1)                                              # z_in: [bs x 1 x patch_len x patch_num]
                z_out_i, _ = self.backbone(z_in)                                            # z_out_i: [bs x 1 x d_model x patch_num]
                z_out.append(z_out_i)
            z = torch.cat(z_out, dim=1)                                                     # z: [bs x nvars x d_model x patch_num]
        else:
            z, _ = self.backbone(z)                                                         # z: [bs x nvars x d_model x patch_num]

        # head
        z = self.head(z)                                                                    # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z

    def create_pretrain_head(self, head_nf, c_in, fc_dropout):
        return nn.Sequential(
                    nn.Dropout(fc_dropout),
                    nn.Conv1d(head_nf, c_in, 1)
                )

