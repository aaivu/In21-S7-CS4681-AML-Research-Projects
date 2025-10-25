import random, torch
from typing import Tuple
def video_mask_indices(T: int, mask_ratio: float = 0.4) -> torch.BoolTensor:
    k = max(1, int(round(T * mask_ratio)))
    idx = torch.zeros(T, dtype=torch.bool)
    idx[torch.randperm(T)[:k]] = True
    return idx

def mlm_mask(input_ids: torch.Tensor, mask_token_id: int, vocab_size: int,
             prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BERT-style: 80% [MASK], 10% random, 10% keep. Labels = id or -100 if not masked.
    """
    B, L = input_ids.shape
    labels = input_ids.clone()
    mask = torch.rand(B, L, device=input_ids.device) < prob
    labels[~mask] = -100

    # 80% -> [MASK]
    choice = torch.rand(B, L, device=input_ids.device)
    mask80 = mask & (choice < 0.8)
    input_ids = input_ids.clone()
    input_ids[mask80] = mask_token_id

    # 10% -> random
    mask10 = mask & (choice >= 0.8) & (choice < 0.9)
    rand_ids = torch.randint(low=0, high=vocab_size, size=input_ids.shape, device=input_ids.device)
    input_ids[mask10] = rand_ids[mask10]
    # 10% -> unchanged (mask but keep token)

    return input_ids, labels
