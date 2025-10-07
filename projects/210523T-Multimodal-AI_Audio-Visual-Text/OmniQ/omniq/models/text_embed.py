# omniq/models/text_embed.py
from typing import List, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextTokenEmbedder(nn.Module):
    """
    Tokenizer + (optionally pretrained) word embeddings only.
    Defaults to bert-base-uncased dims (768) to match your fusion.
    """
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 d_model: int = 768,
                 max_len: int = 128,
                 use_pretrained: bool = False,
                 trainable: bool = False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tok.vocab_size
        self.max_len = max_len
        self.emb = nn.Embedding(self.vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        if use_pretrained:
            try:
                mdl = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True, trust_remote_code=True)
                w = mdl.get_input_embeddings().weight.data
                if w.shape == self.emb.weight.data.shape:
                    self.emb.weight.data.copy_(w)
                del mdl
            except Exception:
                # fallback: random init is fine
                pass

        self.emb.weight.requires_grad = trainable
        self.pos.weight.requires_grad = trainable

    def tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt")
        return out["input_ids"], out["attention_mask"]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, L) -> embeddings: (B, L, D)
        """
        B, L = input_ids.shape
        x = self.emb(input_ids)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = x + self.pos(pos)
        return x
