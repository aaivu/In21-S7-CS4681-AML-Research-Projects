import torch, torch.nn as nn
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

FALLBACK_QWEN = "Qwen/Qwen2.5-7B"

class QwenTextEmbedder(nn.Module):
    """
    Qwen 2.5 / Qwen 2.5-Omni text embeddings only.
    - Loads tokenizer + input embedding matrix from the LM.
    - Projects to d_out (e.g., 768) to match fusion.
    - Ensures a [MASK] token is present for MLM-style training.
    - If Omni class isn't supported in local transformers, falls back to Qwen2.5 text-only.
    """
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-Omni",
                 d_out: int = 768,
                 max_len: int = 128,
                 trainable: bool = False,
                 add_mask_token: bool = True,
                 project_if_needed: bool = True):
        super().__init__()

        # try to load the requested model first; if it fails, fall back
        load_name = model_name
        try:
            self.tok = AutoTokenizer.from_pretrained(load_name, trust_remote_code=True, use_fast=False)
        except Exception:
            load_name = FALLBACK_QWEN
            self.tok = AutoTokenizer.from_pretrained(load_name, trust_remote_code=True, use_fast=False)

        # ensure [MASK] exists
        added_mask = False
        if add_mask_token and self.tok.mask_token is None:
            self.tok.add_special_tokens({"mask_token": "[MASK]"})
            added_mask = True

        # load LM to get embeddings (cpu, half precision to save RAM)
        try:
            base = AutoModelForCausalLM.from_pretrained(
                load_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
        except Exception:
            # hard fallback if Omni isn't supported yet
            load_name = FALLBACK_QWEN
            self.tok = AutoTokenizer.from_pretrained(load_name, trust_remote_code=True, use_fast=False)
            if add_mask_token and self.tok.mask_token is None:
                self.tok.add_special_tokens({"mask_token": "[MASK]"})
                added_mask = True
            base = AutoModelForCausalLM.from_pretrained(
                load_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="cpu"
            )

        if added_mask:
            # make LM aware of the new token before we read weights
            base.resize_token_embeddings(len(self.tok))

        w = base.get_input_embeddings().weight.data.float()  # (vocab, d_in)
        vocab_size, d_in = w.shape

        self.emb = nn.Embedding(vocab_size, d_in)
        self.emb.weight.data.copy_(w)
        self.emb.weight.requires_grad = trainable

        # optional projection to fusion dim
        if project_if_needed and d_in != d_out:
            self.proj = nn.Linear(d_in, d_out, bias=False)
            d_proj = d_out
        else:
            self.proj = nn.Identity()
            d_proj = d_in

        self.pos = nn.Embedding(max_len, d_proj)
        nn.init.normal_(self.pos.weight, std=0.02)

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.d_model_out = d_proj

        # free large model
        del base

    @property
    def mask_token_id(self):
        return self.tok.mask_token_id

    def tokenize(self, texts: List[str]):
        out = self.tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return out["input_ids"], out["attention_mask"]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.emb(input_ids)            # (B,L,d_in)
        x = self.proj(x)                   # (B,L,d_out)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = x + self.pos(pos)
        return x
