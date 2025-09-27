import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(SASRec, self).__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units
        self.num_heads = args.num_heads
        self.num_blocks = args.num_blocks
        self.dropout_rate = args.dropout_rate

        # Item embedding + positional embedding
        self.item_emb = nn.Embedding(itemnum + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, self.hidden_units)
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_units,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_units,
            dropout=self.dropout_rate,
            activation="relu",
            batch_first=True,  # [B, L, H]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_blocks)

        # Prediction layer (dot product scoring is done directly later)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_seq, pos, neg, test_item=None):
        """
        input_seq: [B, L]
        pos:       [B, L]
        neg:       [B, L]
        test_item: [N] (optional, for evaluation)
        """
        mask = (input_seq > 0)  # [B, L]

        # Embedding lookup
        seq_emb = self.item_emb(input_seq)  # [B, L, H]

        # Positional encoding
        positions = torch.arange(self.maxlen, device=input_seq.device).unsqueeze(0).expand_as(input_seq)
        seq_emb = seq_emb + self.pos_emb(positions)

        # Dropout + masking
        seq_emb = self.emb_dropout(seq_emb)
        seq_emb = seq_emb * mask.unsqueeze(-1)

        # Transformer encoding with causal mask
        L = input_seq.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=input_seq.device), diagonal=1).bool()
        seq_emb = self.encoder(seq_emb, mask=causal_mask)
        seq_emb = seq_emb * mask.unsqueeze(-1)  # keep padding zeroed

        # Flatten sequence
        B, L, H = seq_emb.shape
        seq_emb = seq_emb.view(B * L, H)

        # Positive & negative embeddings
        pos_emb = self.item_emb(pos.view(-1))  # [B*L, H]
        neg_emb = self.item_emb(neg.view(-1))  # [B*L, H]

        # Dot product scores
        pos_logits = torch.sum(seq_emb * pos_emb, dim=-1)  # [B*L]
        neg_logits = torch.sum(seq_emb * neg_emb, dim=-1)  # [B*L]

        # Mask out padding positions
        istarget = (pos.view(-1) != 0).float()

        # Loss
        loss = (
            F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits), reduction="none")
            + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none")
        )
        loss = torch.sum(loss * istarget) / torch.sum(istarget)

        # AUC (approximate version, since sign() is not differentiable)
        auc = torch.sum(((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget) / torch.sum(istarget)

        # Test-time prediction
        test_logits = None
        if test_item is not None:
            test_emb = self.item_emb(test_item)  # [N, H]
            last_emb = seq_emb.view(B, L, H)[:, -1, :]  # [B, H]
            test_logits = torch.matmul(last_emb, test_emb.t())  # [B, N]

        return loss, auc, test_logits
