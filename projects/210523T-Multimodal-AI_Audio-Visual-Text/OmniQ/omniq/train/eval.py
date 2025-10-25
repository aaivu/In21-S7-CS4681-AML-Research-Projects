import argparse, json, time, csv
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

from omniq.data.ucf101 import UCF101Clips
from omniq.models.omnivore_backbone import Swin2DTemporalAvg
from omniq.models.omniq_mamba import OmniQMamba
from omniq.models.omniq_transformer import OmniQTransformer


def accuracy(logits, targets, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        B = targets.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        out = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            out.append((correct_k / B * 100.0).item())
        return out

def build_model(cfg):
    if cfg["model"] == "swin_tiny_2d_temporalavg":
        return Swin2DTemporalAvg(num_classes=cfg["num_classes"])
    elif cfg["model"] == "omniq_mamba":
        txt = cfg.get("text", {})
        return OmniQMamba(
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
        return OmniQTransformer(
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

def bytes_to_gb(x): return x / (1024**3)

def eval_once(cfg, ckpt_path, batch_size=8, num_workers=4, amp=True, out_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    val_ds = UCF101Clips(
        root=cfg["data"]["root"],
        split_txt=cfg["data"]["testlist"],
        classind=cfg["data"]["classind"],
        frames=cfg["frames"], stride=cfg["stride"], size=cfg["size"], train=False
    )
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # param counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # latency + VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    top1s, top5s = [], []
    timings = []
    with torch.no_grad():
        for vids, y in val_ld:
            vids = vids.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast("cuda", enabled=amp):
                logits = model(vids)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) / vids.size(0) * 1000.0)  # ms per clip

            t1k, t5k = accuracy(logits, y, topk=(1,5))
            top1s.append(t1k); top5s.append(t5k)

    top1 = sum(top1s)/len(top1s) if top1s else 0.0
    top5 = sum(top5s)/len(top5s) if top5s else 0.0
    avg_lat_ms = sum(timings)/len(timings) if timings else 0.0
    peak_gb = bytes_to_gb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0

    result = {
        "model": cfg["model"],
        "vision_backbone": cfg.get("vision_backbone", "swin_tiny_patch4_window7_224"),
        "fusion_depth": cfg.get("fusion", {}).get("depth", 0),
        "lora_enabled": cfg.get("lora", {}).get("enabled", False),
        "dataset": "UCF101",
        "split": cfg["data"]["split"],
        "frames": cfg["frames"], "stride": cfg["stride"], "size": cfg["size"],
        "batch_eval": batch_size,
        "top1": round(top1, 2), "top5": round(top5, 2),
        "params_M": round(total_params/1e6, 2),
        "trainable_params_M": round(trainable_params/1e6, 2),
        "peak_vram_GB": round(peak_gb, 2),
        "avg_latency_ms": round(avg_lat_ms, 2),
        "ckpt": str(ckpt_path),
    }

    out_dir = Path(out_dir or cfg["train"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"eval.json").write_text(json.dumps(result, indent=2))

    # append to a global summary CSV
    res_dir = Path("results"); res_dir.mkdir(exist_ok=True)
    csv_path = res_dir / "summary.csv"
    header = ["model","vision_backbone","fusion_depth","lora_enabled","dataset","split",
              "frames","stride","size","batch_eval","top1","top5","params_M",
              "trainable_params_M","peak_vram_GB","avg_latency_ms","ckpt"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([result[k] for k in header])

    print(json.dumps(result, indent=2))
    print(f"\nWrote per-run eval to: {out_dir/'eval.json'}")
    print(f"Appended summary row to: {csv_path}")
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    eval_once(cfg, Path(args.ckpt), batch_size=args.batch,
              num_workers=args.workers, amp=not args.no_amp)

if __name__ == "__main__":
    main()
