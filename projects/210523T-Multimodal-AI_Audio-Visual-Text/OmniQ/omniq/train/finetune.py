import argparse, os, yaml, time, csv
from pathlib import Path
from omniq.models.omniq_mamba import OmniQMamba
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from omniq.models.lora import loraize_linear_modules, mark_only_lora_and_head_trainable, collect_lora_params
from omniq.data.ucf101 import UCF101Clips
from omniq.models.omnivore_backbone import Swin2DTemporalAvg
from omniq.models.omniq_transformer import OmniQTransformer


def make_loaders(cfg):
    root = cfg["data"]["root"]
    split = cfg["data"]["split"]
    frames, stride, size = cfg["frames"], cfg["stride"], cfg["size"]

    trainlist = cfg["data"]["trainlist"]
    testlist  = cfg["data"]["testlist"]
    classind  = cfg["data"]["classind"]

    train_ds = UCF101Clips(root, trainlist, classind, frames, stride, size, train=True)
    val_ds   = UCF101Clips(root, testlist,  classind, frames, stride, size, train=False)

    train_ld = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=max(1, cfg["train"]["batch_size"]//2), shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True
    )
    return train_ld, val_ld


def accuracy(logits, targets, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        B = targets.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k / B * 100.0).item())
        return res


def main(args):
    cfg = yaml.safe_load(open(args.config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ld, val_ld = make_loaders(cfg)
    txt = cfg.get("text", {})
    if cfg["model"] == "swin_tiny_2d_temporalavg":
        model = Swin2DTemporalAvg(num_classes=cfg["num_classes"])

    elif cfg["model"] == "omniq_mamba":
        model = OmniQMamba(
            backbone_name=cfg.get("vision_backbone", "swin_tiny_patch4_window7_224"),
            num_classes=cfg["num_classes"],
            max_frames=cfg.get("frames", 32),
            fusion_depth=cfg.get("fusion", {}).get("depth", 2),
            d_state=cfg.get("fusion", {}).get("d_state", 128),
            expand=cfg.get("fusion", {}).get("expand", 2),
            dropout=cfg.get("fusion", {}).get("dropout", 0.1),
            use_text=txt.get("enabled", False),
            text_model_name=txt.get("model_name", "bert-base-uncased"),
            text_max_len=txt.get("max_len", 64),
            text_use_pretrained=txt.get("use_pretrained", False),
            text_trainable=txt.get("trainable", False),
        )

    elif cfg["model"] == "omniq_transformer":
        txt = cfg.get("text", {})
        model = OmniQTransformer(
            backbone_name=cfg.get("vision_backbone", "swin_tiny_patch4_window7_224"),
            num_classes=cfg["num_classes"],
            max_frames=cfg.get("frames", 32),
            fusion_depth=cfg.get("fusion", {}).get("depth", 2),
            n_heads=cfg.get("fusion", {}).get("n_heads", 8),
            dropout=cfg.get("fusion", {}).get("dropout", 0.1),
            use_text=txt.get("enabled", False),
            text_model_name=txt.get("model_name", "bert-base-uncased"),
            text_max_len=txt.get("max_len", 64),
            text_use_pretrained=txt.get("use_pretrained", False),
            text_trainable=txt.get("trainable", False),
    )

    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    model = model.to(device)

    # --- LoRA enable (fusion only) ---
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        # Apply LoRA to Linear layers inside the fusion stack only
        if hasattr(model, "fusion"):
            # You can filter to specific sublayers by name; here we include all nn.Linear in fusion
            loraize_linear_modules(model.fusion,
                                   r=lora_cfg.get("r", 8),
                                   alpha=lora_cfg.get("alpha", 16),
                                   dropout=lora_cfg.get("dropout", 0.05),
                                   name_filter=lora_cfg.get("name_filter", ()))
        # Freeze vision backbone if requested
        if lora_cfg.get("freeze_backbone", True) and hasattr(model, "vision"):
            for p in model.vision.parameters():
                p.requires_grad = False
        # Only train LoRA + head
        mark_only_lora_and_head_trainable(model, head_attr="head")
        # --- end LoRA enable ---

        # --- Warm start: load fusion from masked-pretrain checkpoint  ---
        warm_ckpt = cfg.get("init_from_pretrain", None)
        if warm_ckpt:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(warm_ckpt, map_location=device)
            if "fusion" in ckpt:
                load_out = model.fusion.load_state_dict(ckpt["fusion"], strict=False)
                missing, unexpected = load_out.missing_keys, load_out.unexpected_keys
                print(f"[warmstart] loaded fusion from {warm_ckpt}")
                if missing:   print(f"[warmstart] missing: {len(missing)} keys (ok)")
                if unexpected:print(f"[warmstart] unexpected: {len(unexpected)} keys (ok)")
            else:
                print(f"[warmstart] No 'fusion' key in {warm_ckpt}; skipping.")
            # --- end warm start ---

    criterion = nn.CrossEntropyLoss()

    # different LR for head/backbone
    param_groups = []
    if lora_cfg.get("enabled", False):
        # LoRA + head at lr_head
        lora_params = collect_lora_params(model)
        if lora_params:
            param_groups.append({"params": lora_params, "lr": cfg["optim"]["lr_head"]})
        if hasattr(model, "head"):
            param_groups.append({"params": model.head.parameters(), "lr": cfg["optim"]["lr_head"]})
        # Optionally include backbone if not frozen
        if not lora_cfg.get("freeze_backbone", True):
            if hasattr(model, "vision"):
                param_groups.append({"params": model.vision.parameters(), "lr": cfg["optim"]["lr_backbone"]})
    else:
        # previous behavior: backbone at lr_backbone, fusion+head at lr_head
        if hasattr(model, "vision"):
            param_groups.append({"params": model.vision.parameters(), "lr": cfg["optim"]["lr_backbone"]})
        rest = []
        if hasattr(model, "fusion"): rest.append(model.fusion.parameters())
        if hasattr(model, "head"):   rest.append(model.head.parameters())
        if rest:
            from itertools import chain
            param_groups.append({"params": chain(*rest), "lr": cfg["optim"]["lr_head"]})

    opt = torch.optim.AdamW(param_groups, weight_decay=cfg["optim"]["weight_decay"])

    scaler = GradScaler('cuda', enabled=cfg["train"].get("amp", True))

    save_dir = Path(cfg["train"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(cfg["train"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(save_dir))                       # <— add
    csv_path = save_dir / "metrics.csv"                         # <— add
    if not csv_path.exists():                                   # <— add
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_top1","val_top5"])

    best_top1 = 0.0
    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (vid, y) in enumerate(train_ld, 1):
            vid = vid.to(device, non_blocking=True)   # (B,C,T,H,W)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=cfg["train"].get("amp", True)):
                logits = model(vid)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item()

            if i % 20 == 0:
                print(f"epoch {epoch} iter {i}/{len(train_ld)}  loss {running/i:.4f}")

        avg_train = running / max(1, i)

        # eval
        model.eval()
        top1s, top5s = [], []
        with torch.no_grad():
            for vid, y in val_ld:
                vid = vid.to(device); y = y.to(device)
                logits = model(vid)
                t1, t5 = accuracy(logits, y, topk=(1,5))
                top1s.append(t1); top5s.append(t5)

        top1 = sum(top1s)/len(top1s); top5 = sum(top5s)/len(top5s)
        dt = time.time()-t0
        print(f"[epoch {epoch}] val top1={top1:.2f} top5={top5:.2f}  ({dt:.1f}s)")
        writer.add_scalar("train/loss", avg_train, epoch)
        writer.add_scalar("val/top1", top1, epoch)
        writer.add_scalar("val/top5", top5, epoch)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_train:.6f}", f"{top1:.2f}", f"{top5:.2f}"])

    # simple checkpoint
        if top1 > best_top1:
            best_top1 = top1
            ckpt = save_dir / "best.pt"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "top1": best_top1, "cfg": cfg}, ckpt)
            print(f"saved {ckpt}  (top1={best_top1:.2f})")
        writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args())
