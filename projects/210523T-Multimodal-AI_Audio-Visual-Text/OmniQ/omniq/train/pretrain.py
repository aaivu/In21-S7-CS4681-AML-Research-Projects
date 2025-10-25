import argparse, yaml, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from omniq.data.ucf101 import UCF101Clips
from omniq.data.masking import video_mask_indices, mlm_mask
from omniq.models.omniq_mamba import OmniQMamba
from omniq.models.lora import loraize_linear_modules, mark_only_lora_and_head_trainable

def load_id2name(classind_path: str):
    id2name = {}
    with open(classind_path, "r") as f:
        for ln in f:
            if not ln.strip(): continue
            i, name = ln.strip().split()
            id2name[int(i)-1] = name
    return id2name

def make_loader(cfg):
    ds = UCF101Clips(
        root=cfg["data"]["root"],
        split_txt=cfg["data"]["trainlist"],
        classind=cfg["data"]["classind"],
        frames=cfg["frames"], stride=cfg["stride"], size=cfg["size"], train=True
    )
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                      num_workers=cfg["train"]["num_workers"], pin_memory=True)

def main(args):
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ld = make_loader(cfg)

    # --- model with text enabled ---
    txt = cfg.get("text", {})
    model = OmniQMamba(
        backbone_name=cfg.get("vision_backbone", "swin_tiny_patch4_window7_224"),
        num_classes=cfg.get("num_classes", 101),
        max_frames=cfg.get("frames", 32),
        fusion_depth=cfg.get("fusion", {}).get("depth", 2),
        d_state=cfg.get("fusion", {}).get("d_state", 128),
        expand=cfg.get("fusion", {}).get("expand", 2),
        dropout=cfg.get("fusion", {}).get("dropout", 0.1),
        use_text=True,
        text_model_name=txt.get("model_name", "bert-base-uncased"),
        text_max_len=txt.get("max_len", 16),
        text_use_pretrained=txt.get("use_pretrained", False),
        text_trainable=txt.get("trainable", False),
    ).to(device)

    # Freeze vision during pretrain (predict its features)
    for p in model.vision.parameters(): p.requires_grad = False

    # LoRA on fusion
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        loraize_linear_modules(model.fusion, r=lora_cfg.get("r", 8),
                               alpha=lora_cfg.get("alpha", 16),
                               dropout=lora_cfg.get("dropout", 0.05))
        mark_only_lora_and_head_trainable(model, head_attr="head")  # we'll add our own heads below; this freezes fusion base

    # Heads for masked prediction
    d = model.d_model
    proj_video = nn.Linear(d, d).to(device)     # predict frozen vision features (feature-level MIM)
    lm_head = nn.Linear(d, model.text.vocab_size, bias=False).to(device)  # weight tie to text emb
    lm_head.weight = model.text.emb.weight      # tie

    # Loss weights
    w_v = cfg["loss"].get("video_mse", 0.5)
    w_t = cfg["loss"].get("text_ce", 0.5)

    # Optimizer (LoRA + heads)
    params = [p for p in list(proj_video.parameters()) + list(lm_head.parameters()) if p.requires_grad]
    for m in model.modules():
        # collect LoRA params (identified by 'lora_' names)
        for n, p in m.named_parameters(recurse=False):
            if "lora_" in n and p.requires_grad: params.append(p)
    opt = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    scaler = GradScaler("cuda", enabled=cfg["train"].get("amp", True))

    id2name = load_id2name(cfg["data"]["classind"])
    mask_token_id = model.text.tok.mask_token_id or model.text.tok.unk_token_id

    save_dir = Path(cfg["train"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)

    model.train(); proj_video.train(); lm_head.train()
    mse = nn.MSELoss(); ce = nn.CrossEntropyLoss(ignore_index=-100)

    steps = 0
    for epoch in range(1, cfg["train"]["epochs"]+1):
        t0 = time.time()
        running = 0.0
        for vids, labels in ld:
            vids = vids.to(device)            # (B,C,T,H,W)
            B, _, T, _, _ = vids.shape

            # --- video tokens & mask ---
            with torch.no_grad():
                tokens_v = model._frames_to_tokens(vids)            # (B,T,D) frozen target
            mask_v = torch.stack([video_mask_indices(T, cfg["mask"].get("video_ratio", 0.4)) for _ in range(B)]).to(device)  # (B,T)
            tokens_v_in = tokens_v.clone()
            tokens_v_tgt = tokens_v.clone().detach()
            tokens_v_in[mask_v] = 0.0                                # zero masked tokens at input

            # --- synthesize a short text from class names, then MLM ---
            texts = [f"a video of {id2name[int(y.item())].replace('_', ' ').lower()}" for y in labels]
            input_ids, attn = model.text.tokenize(texts)
            input_ids = input_ids.to(device)
            input_ids_masked, labels_mlm = mlm_mask(input_ids, mask_token_id, model.text.vocab_size, prob=cfg["mask"].get("text_ratio", 0.15))
            labels_mlm = labels_mlm.to(device)

            # --- build sequence: [VIS_CLS] + V_masked + [TXT_CLS] + E(text_masked) ---
            time_ids = torch.arange(T, device=device).clamp(max=model.max_frames-1)
            v = tokens_v_in + model.time_embed(time_ids).unsqueeze(0)
            v = v + model.type_embed.weight[0].view(1,1,-1)
            seq = torch.cat([model.cls_vis.expand(B,1,-1), v], dim=1)   # (B,1+T,D)

            txt_emb = model.text(input_ids_masked.to(device))           # (B,L,D)
            L = txt_emb.shape[1]
            txt_emb = txt_emb + model.type_embed.weight[1].view(1,1,-1)
            seq = torch.cat([seq, model.cls_txt.expand(B,1,-1), txt_emb], dim=1)  # (B,1+T+1+L,D)

            # --- fuse & compute losses on masked positions ---
            with autocast("cuda", enabled=cfg["train"].get("amp", True)):
                fused = model.fusion(seq)                                # (B,1+T+1+L,D)

                # video MSE on masked positions (positions 1..T in fused)
                vid_states = fused[:, 1:1+T, :]                          # (B,T,D)
                pred_video = proj_video(vid_states[mask_v])              # (Nmask,D)
                loss_v = mse(pred_video, tokens_v_tgt[mask_v])

                # text CE on masked token positions (after 1+T+1 offset)
                txt_states = fused[:, 1+T+1:, :]                         # (B,L,D) (skip TXT_CLS)
                logits = lm_head(txt_states)                             # (B,L,V)
                loss_t = ce(logits.reshape(-1, logits.size(-1)), labels_mlm.reshape(-1))

                loss = w_v * loss_v + w_t * loss_t

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            running += loss.item(); steps += 1
            if steps % 20 == 0:
                print(f"epoch {epoch} step {steps}  loss {running/20:.4f}  (v={loss_v.item():.4f} t={loss_t.item():.4f})")
                running = 0.0

        dt = time.time()-t0
        print(f"[epoch {epoch}] done in {dt:.1f}s")

        # save small checkpoint of fusion + heads
        torch.save({
            "epoch": epoch,
            "fusion": model.fusion.state_dict(),
            "proj_video": proj_video.state_dict(),
            "lm_head": lm_head.state_dict(),
            "cfg": cfg
        }, save_dir / f"pretrain_epoch{epoch}.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args())
