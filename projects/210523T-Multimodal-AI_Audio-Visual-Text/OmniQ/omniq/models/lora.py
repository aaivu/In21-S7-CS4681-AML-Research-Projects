# omniq/models/lora.py
from typing import Iterable, List, Tuple
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear with LoRA: W*x + (alpha/r) * BA*x
    Keeps the original weight frozen by default; trains A,B.
    """
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias is not None

        # frozen base
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias_param = None
        if self.bias:
            self.bias_param = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

        # LoRA params
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math_sqrt2())
        nn.init.zeros_(self.lora_B)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.t()
        if self.bias_param is not None:
            base = base + self.bias_param
        # Ensure LoRA params are on the same device as x
        lora_A = self.lora_A.to(x.device)
        lora_B = self.lora_B.to(x.device)
        lora = self.drop(x) @ lora_A @ lora_B
        return base + self.scaling * lora

def math_sqrt2():  # small helper for init
    return 2 ** 0.5

def loraize_linear_modules(module: nn.Module, r: int, alpha: int, dropout: float = 0.0, name_filter: Tuple[str,...] = ()):
    """
    Replace nn.Linear modules inside `module` with LoRALinear.
    If name_filter is provided, only layers whose qualified name contains any token are LoRA-ized.
    """
    for fullname, sub in list(module.named_modules()):
        # skip the top-level 'module' itself
        if isinstance(sub, nn.Linear):
            if name_filter and not any(tok in fullname for tok in name_filter):
                continue
            parent = get_parent_by_qualified_name(module, fullname)
            attr = fullname.split('.')[-1]
            setattr(parent, attr, LoRALinear(sub, r=r, alpha=alpha, dropout=dropout))

def get_parent_by_qualified_name(root: nn.Module, qname: str) -> nn.Module:
    parts = qname.split('.')
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    return cur

def mark_only_lora_and_head_trainable(model: nn.Module, head_attr: str = "head"):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze LoRA params
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
    # unfreeze classification head if present
    if hasattr(model, head_attr):
        for p in getattr(model, head_attr).parameters():
            p.requires_grad = True

def collect_lora_params(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params
