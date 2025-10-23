################################################################################
# LSTM + (optional) MoE Language Model with Weight Tying (Configurable)
################################################################################
import torch
import torch.nn as nn
from .moe import MoE


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_layers=2,
        use_moe=False,
        moe_output_size=None,
        moe_num_experts=4,
        moe_hidden_size=128,
        moe_k=2,
        moe_mode="baseline",      
        lora_rank=8
    ):
        super(LSTMLanguageModel, self).__init__()
        self.use_moe = use_moe
        self.vocab_size = vocab_size

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Stacked LSTM with dropout between layers
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # === MoE Integration ===
        if self.use_moe:
            assert moe_output_size is not None, "Need to specify moe_output_size (= vocab_size for LM)"
            self.moe = MoE(
                input_size=hidden_size,
                output_size=moe_output_size,
                num_experts=moe_num_experts,
                hidden_size=moe_hidden_size,
                k=moe_k,
                mode=moe_mode,     
                lora_rank=lora_rank
            )
        else:
            # Projection layer (LM head)
            self.fc = nn.Linear(hidden_size, vocab_size, bias=False)
            self.fc.weight = self.embed.weight  # weight tying

        self.dropout = nn.Dropout(0.3)
        nn.init.normal_(self.embed.weight, mean=0.0, std=hidden_size ** -0.5)

    def forward(self, x, hidden=None):
        """
        Forward pass
        Args:
            x: [batch_size, seq_len]
            hidden: (h0, c0)
        Returns:
            out: [batch, seq_len, vocab_size]
            hidden: new hidden states
            aux_loss: scalar tensor (0 if no MoE)
        """
        emb = self.embed(x)
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)

        if self.use_moe:
            batch, seq, hidden_dim = output.size()
            output = output.reshape(batch * seq, hidden_dim)
            out, aux_loss = self.moe(output)
            out = out.reshape(batch, seq, -1)
            return out, hidden, aux_loss
        else:
            out = self.fc(output)
            return out, hidden, torch.tensor(0.0, device=out.device)
