from typing import Optional, List
import torch
import torch.nn as nn
import timm
from .qwen_embed import QwenTextEmbedder
from .fusion_mamba import FusionMamba
from .text_embed import TextTokenEmbedder

class OmniQMamba(nn.Module):
    """
    Visual encoder (Swin-2D per-frame features) + Mamba fusion + classifier.
    UCF101 has no text; we treat this as vision-only for now (still goes through fusion).
    """
    def __init__(self,
                 backbone_name: str = "swin_tiny_patch4_window7_224",
                 num_classes: int = 101,
                 max_frames: int = 64,
                 fusion_depth: int = 2,
                 d_state: int = 128,
                 expand: int = 2,
                 dropout: float = 0.1,
                 use_text: bool = False,
                 text_model_name: str = "bert-base-uncased",
                 text_max_len: int = 64,
                 text_use_pretrained: bool = False,
                 text_trainable: bool = False):
        super().__init__()

        # 1) Visual encoder -> per-frame features (no classifier)
        self.vision = timm.create_model(backbone_name, pretrained=True,
                                        num_classes=0, global_pool='avg')
        self.d_model = self.vision.num_features
        self.num_classes = num_classes
        self.max_frames = max_frames

        # 2) Tokens & embeddings
        self.cls_vis = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.cls_txt = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.type_embed = nn.Embedding(2, self.d_model)   # 0=video/vis_cls, 1=text/txt_cls
        self.time_embed = nn.Embedding(max_frames, self.d_model)

        nn.init.normal_(self.cls_vis, std=0.02)
        nn.init.normal_(self.cls_txt, std=0.02)
        nn.init.normal_(self.time_embed.weight, std=0.02)

        # 3) Optional text
        self.use_text = use_text
        self.text_provider = "qwen"  # default provider label; overridden by args below
        if use_text:
            provider = text_model_name.lower()
            self.text_provider = "qwen" if "qwen" in provider else "bert"
            if self.text_provider == "qwen":
                self.text = QwenTextEmbedder(
                    model_name=text_model_name,
                    d_out=self.d_model,
                    max_len=text_max_len,
                    trainable=text_trainable,
                    add_mask_token=True,
                    project_if_needed=True
                )
            else:
                from .text_embed import TextTokenEmbedder
                self.text = TextTokenEmbedder(
                    model_name=text_model_name,
                    d_model=self.d_model,
                    max_len=text_max_len,
                    use_pretrained=text_use_pretrained,
                    trainable=text_trainable
                )
            self.text_max_len = text_max_len

        # 4) Fusion via Mamba
        self.fusion = FusionMamba(d_model=self.d_model,
                                  depth=fusion_depth,
                                  d_state=d_state,
                                  expand=expand,
                                  dropout=dropout,
                                  bidirectional=True)

        # 5) Head
        self.head = nn.Linear(self.d_model, num_classes)


    @torch.no_grad()
    def _frames_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W) -> tokens_v: (B, T, D) by applying 2D Swin per-frame.
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)      # (B*T,C,H,W)
        feats = self.vision(x)                                   # (B*T, D)
        tokens = feats.view(B, T, -1)                            # (B, T, D)
        return tokens

    def forward(self, video: torch.Tensor, text_input: dict = None) -> torch.Tensor:
        """
        video: (B, C, T, H, W)
        text_input: Optional dict with key 'input_ids': (B, L)
        """
        B, _, T, _, _ = video.shape
        T = min(T, self.max_frames)

        # video -> tokens
        tokens_v = self._frames_to_tokens(video[:, :, :T])       # (B,T,D)

        # add temporal + type embeddings to video tokens
        time_ids = torch.arange(T, device=video.device).clamp(max=self.max_frames-1)
        tokens_v = tokens_v + self.time_embed(time_ids).unsqueeze(0)        # (B,T,D)
        tokens_v = tokens_v + self.type_embed.weight[0].view(1,1,-1)        # type 0

        # prepend VIS_CLS
        seq = torch.cat([self.cls_vis.expand(B, 1, -1), tokens_v], dim=1)   # (B,1+T,D)

        # optional text branch
        if self.use_text and text_input is not None and "input_ids" in text_input:
            ids = text_input["input_ids"].to(video.device)                  # (B,L)
            txt = self.text(ids)                                            # (B,L,D)
            txt = txt + self.type_embed.weight[1].view(1,1,-1)              # type 1
            # prepend TXT_CLS then append to seq
            txt = torch.cat([self.cls_txt.expand(B, 1, -1), txt], dim=1)    # (B,1+L,D)
            seq = torch.cat([seq, txt], dim=1)                              # (B,1+T+1+L,D)

        # fuse
        fused = self.fusion(seq)                                            # (B,Ltot,D)
        pooled = fused[:, 0]                                                # VIS_CLS
        logits = self.head(pooled)
        return logits
