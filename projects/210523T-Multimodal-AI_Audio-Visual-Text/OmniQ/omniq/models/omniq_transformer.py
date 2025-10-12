import torch, torch.nn as nn, timm
from .fusion_transformer import FusionTransformer
from .text_embed import TextTokenEmbedder

class OmniQTransformer(nn.Module):
    def __init__(self,
                 backbone_name: str = "swin_tiny_patch4_window7_224",
                 num_classes: int = 101,
                 max_frames: int = 64,
                 fusion_depth: int = 2,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 use_text: bool = False,
                 text_model_name: str = "bert-base-uncased",
                 text_max_len: int = 64,
                 text_use_pretrained: bool = False,
                 text_trainable: bool = False):
        super().__init__()

        self.vision = timm.create_model(backbone_name, pretrained=True,
                                        num_classes=0, global_pool='avg')
        self.d_model = self.vision.num_features
        self.num_classes = num_classes
        self.max_frames = max_frames

        self.cls_vis = nn.Parameter(torch.zeros(1,1,self.d_model))
        self.cls_txt = nn.Parameter(torch.zeros(1,1,self.d_model))
        self.type_embed = nn.Embedding(2, self.d_model)      # 0=video, 1=text
        self.time_embed = nn.Embedding(max_frames, self.d_model)
        nn.init.normal_(self.cls_vis, std=0.02)
        nn.init.normal_(self.cls_txt, std=0.02)
        nn.init.normal_(self.time_embed.weight, std=0.02)

        self.use_text = use_text
        if use_text:
            self.text = TextTokenEmbedder(model_name=text_model_name,
                                          d_model=self.d_model,
                                          max_len=text_max_len,
                                          use_pretrained=text_use_pretrained,
                                          trainable=text_trainable)
            self.text_max_len = text_max_len

        self.fusion = FusionTransformer(d_model=self.d_model,
                                        depth=fusion_depth,
                                        n_heads=n_heads,
                                        mlp_ratio=4,
                                        dropout=dropout)

        self.head = nn.Linear(self.d_model, num_classes)

    @torch.no_grad()
    def _frames_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        feats = self.vision(x)                 # (B*T, D)
        return feats.view(B, T, -1)            # (B,T,D)

    def forward(self, video: torch.Tensor, text_input: dict = None):
        B, _, T, _, _ = video.shape
        T = min(T, self.max_frames)

        tokens_v = self._frames_to_tokens(video[:, :, :T])                  # (B,T,D)
        time_ids = torch.arange(T, device=video.device).clamp(max=self.max_frames-1)
        tokens_v = tokens_v + self.time_embed(time_ids).unsqueeze(0)        # (B,T,D)
        tokens_v = tokens_v + self.type_embed.weight[0].view(1,1,-1)        # type 0

        seq = torch.cat([self.cls_vis.expand(B,1,-1), tokens_v], dim=1)     # (B,1+T,D)

        if self.use_text and text_input is not None and "input_ids" in text_input:
            ids = text_input["input_ids"].to(video.device)
            txt = self.text(ids)                                            # (B,L,D)
            txt = txt + self.type_embed.weight[1].view(1,1,-1)
            seq = torch.cat([seq, self.cls_txt.expand(B,1,-1), txt], dim=1) # (B,1+T+1+L,D)

        fused = self.fusion(seq)                                            # (B,Ltot,D)
        pooled = fused[:, 0]
        return self.head(pooled)
