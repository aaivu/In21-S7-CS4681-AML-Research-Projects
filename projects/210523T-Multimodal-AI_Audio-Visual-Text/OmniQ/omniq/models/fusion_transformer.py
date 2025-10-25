import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    """
    Thin Transformer encoder stack for sequence fusion.
    x: (B, L, D) -> (B, L, D)
    """
    def __init__(self, d_model: int, depth: int = 2, n_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()

        # Create custom transformer encoder layer to avoid version compatibility issues
        self.layers = nn.ModuleList([
            TransformerEncoderLayerCustom(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=mlp_ratio * d_model,
                dropout=dropout
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerEncoderLayerCustom(nn.Module):
    """Custom Transformer Encoder Layer to avoid PyTorch version issues."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Use manual attention implementation to avoid version issues
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None):
        # Pre-norm architecture
        src2 = self.norm1(src)

        # Manual multi-head attention
        B, L, D = src2.shape

        # Project to Q, K, V
        q = self.q_proj(src2).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, L, head_dim)
        k = self.k_proj(src2).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, L, head_dim)
        v = self.v_proj(src2).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, L, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, nhead, L, L)

        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, nhead, L, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        # Output projection
        src2 = self.out_proj(attn_output)
        src = src + self.dropout1(src2)

        # Feed forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src