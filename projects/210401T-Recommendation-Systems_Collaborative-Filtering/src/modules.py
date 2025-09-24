# -*- coding: utf-8 -*-
"""
Converted from TensorFlow to PyTorch
Original Author: Kyubyong Park (2017)
Conversion: Nethum, 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################
# Positional Encoding
#########################################
def positional_encoding(sentence_length, dim, device='cpu'):
    """
    Returns positional encoding as in Vaswani et al. (2017).
    Args:
        sentence_length: int, sequence length
        dim: int, model dimension
        device: torch device
    Returns:
        Tensor of shape [sentence_length, dim]
    """
    pos = torch.arange(sentence_length, dtype=torch.float, device=device).unsqueeze(1)
    i = torch.arange(dim, dtype=torch.float, device=device).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / dim)
    angle_rads = pos * angle_rates

    pe = torch.zeros(sentence_length, dim, device=device)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pe


#########################################
# Layer Normalization
#########################################
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


#########################################
# Embedding
#########################################
class Embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zero_pad=True, scale=True):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, num_units)
        self.scale = scale
        self.zero_pad = zero_pad
        if zero_pad:
            with torch.no_grad():
                self.embed.weight[0].fill_(0)

    def forward(self, x):
        out = self.embed(x)
        if self.scale:
            out = out * (self.embed.embedding_dim ** 0.5)
        return out


#########################################
# Multi-Head Attention
#########################################
class MultiHeadAttention(nn.Module):
    def __init__(self, num_units, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert num_units % num_heads == 0, "num_units must be divisible by num_heads"

        self.num_heads = num_heads
        self.num_units = num_units
        self.head_dim = num_units // num_heads

        self.W_Q = nn.Linear(num_units, num_units)
        self.W_K = nn.Linear(num_units, num_units)
        self.W_V = nn.Linear(num_units, num_units)
        self.fc = nn.Linear(num_units, num_units)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, mask=None):
        B, T_q, _ = queries.size()
        _, T_k, _ = keys.size()

        Q = self.W_Q(queries).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(keys).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(keys).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(B, T_q, self.num_units)

        # Residual connection
        out = self.fc(context) + queries
        return out


#########################################
# Position-wise Feed Forward Network
#########################################
class PositionwiseFeedForward(nn.Module):
    def __init__(self, num_units, hidden_dim=2048, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(num_units, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim, num_units, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x.transpose(1, 2)  # (B, num_units, T)
        out = self.relu(self.conv1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)
        return out + x  # Residual connection

#########################################
# SASRec Model
#########################################
class SASRec(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(SASRec, self).__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.dropout_rate = args.dropout_rate

        # Item embedding
        self.item_emb = Embedding(itemnum + 1, self.hidden_units)  # +1 for padding idx
        # Positional encoding
        self.positional_encoding = positional_encoding(self.maxlen, self.hidden_units)

        # Dropout and LayerNorm
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layernorm_emb = LayerNorm(self.hidden_units)

        # Transformer blocks
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(self.hidden_units, self.num_heads, self.dropout_rate)
            for _ in range(self.num_blocks)
        ])
        self.layernorms_attn = nn.ModuleList([LayerNorm(self.hidden_units) for _ in range(self.num_blocks)])
        self.ffn_layers = nn.ModuleList([
            PositionwiseFeedForward(self.hidden_units, self.hidden_units * 4, self.dropout_rate)
            for _ in range(self.num_blocks)
        ])
        self.layernorms_ffn = nn.ModuleList([LayerNorm(self.hidden_units) for _ in range(self.num_blocks)])

        # Prediction layers
        self.final_layer = nn.Linear(self.hidden_units, self.hidden_units)

    def forward(self, seq):
        """
        seq: (batch_size, maxlen)
        """
        device = seq.device
        seq_emb = self.item_emb(seq)
        seq_emb = seq_emb + self.positional_encoding.to(device)
        seq_emb = self.dropout(seq_emb)
        seq_emb = self.layernorm_emb(seq_emb)

        # Attention + FFN blocks
        mask = torch.tril(torch.ones((self.maxlen, self.maxlen), device=device)).unsqueeze(0).unsqueeze(0)
        for i in range(self.num_blocks):
            seq_emb = self.attention_layers[i](seq_emb, seq_emb, mask=mask)
            seq_emb = self.layernorms_attn[i](seq_emb)
            seq_emb = self.ffn_layers[i](seq_emb)
            seq_emb = self.layernorms_ffn[i](seq_emb)

        return seq_emb  # Shape: (batch, maxlen, hidden_units)


    def predict(self, user_ids, seq, item_idx):
        """
        Predict scores for a given user, sequence, and item candidates.
        Args:
            user_ids: Tensor [batch_size]
            seq: Tensor [batch_size, maxlen]
            item_idx: Tensor [num_items]
        """
        self.eval()
        with torch.no_grad():
            # Get user sequence representation
            seq_output = self.forward(seq)  # [B, maxlen, hidden_units]
            seq_output = seq_output[:, -1, :]  # last position representation

            # Get item embeddings
            item_emb = self.item_emb(item_idx)  # [num_items, hidden_units]

            # Compute dot product (B=1 in evaluation)
            scores = torch.matmul(seq_output, item_emb.t())  # [B, num_items]
            return scores.detach().cpu().numpy().flatten()

    # def predict(self, seq):
    #     """
    #     For inference: Get representation of last position
    #     """
    #     seq_emb = self.forward(seq)
    #     return seq_emb[:, -1, :]  # (batch, hidden_units)

    def calculate_loss(self, u, seq, pos, neg):
        """
        seq: (batch_size, maxlen)
        pos: positive samples
        neg: negative samples
        """
        seq_emb = self.forward(seq)
        pos_emb = self.item_emb(pos)
        neg_emb = self.item_emb(neg)

        # Only last position prediction (or all positions?)
        pos_logits = torch.sum(seq_emb * pos_emb, dim=-1)
        neg_logits = torch.sum(seq_emb * neg_emb, dim=-1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_logits) + 1e-24) +
                           torch.log(1 - torch.sigmoid(neg_logits) + 1e-24))

        return loss
