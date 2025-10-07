import argparse, os, yaml, time, json
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from omniq.data.ucf101 import UCF101Clips
from omniq.models.omnivore_backbone import Swin2DTemporalAvg


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


class TrainingLogger:
    """Logger to track training metrics for visualization."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'epochs': [],
            'train_losses': [],
            'train_losses_per_iter': [],
            'val_top1': [],
            'val_top5': [],
            'train_times': [],
            'learning_rates': [],
            'iteration_times': []
        }
        
        self.current_epoch = 0
        self.current_iter = 0
    
    def log_epoch_start(self, epoch: int):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.epoch_losses = []
    
    def log_iteration(self, loss: float, lr: float = None):
        self.current_iter += 1
        self.epoch_losses.append(loss)
        self.metrics['train_losses_per_iter'].append({
            'epoch': self.current_epoch,
            'iteration': self.current_iter,
            'loss': loss,
            'lr': lr
        })
    
    def log_epoch_end(self, val_top1: float, val_top5: float):
        epoch_time = time.time() - self.epoch_start_time
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0.0
        
        self.metrics['epochs'].append(self.current_epoch)
        self.metrics['train_losses'].append(avg_loss)
        self.metrics['val_top1'].append(val_top1)
        self.metrics['val_top5'].append(val_top5)
        self.metrics['train_times'].append(epoch_time)
        
        # Save metrics after each epoch
        self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        log_file = self.save_dir / "training_logs.json"
        with open(log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_final_summary(self, config: dict, best_top1: float):
        """Save final training summary."""
        summary = {
            'config': config,
            'best_top1_accuracy': best_top1,
            'total_epochs': len(self.metrics['epochs']),
            'total_training_time': sum(self.metrics['train_times']),
            'final_metrics': {
                'train_loss': self.metrics['train_losses'][-1] if self.metrics['train_losses'] else None,
                'val_top1': self.metrics['val_top1'][-1] if self.metrics['val_top1'] else None,
                'val_top5': self.metrics['val_top5'][-1] if self.metrics['val_top5'] else None,
            }
        }
        
        summary_file = self.save_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def main(args):
    cfg = yaml.safe_load(open(args.config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_ld, val_ld = make_loaders(cfg)

    if cfg["model"] == "swin_tiny_2d_temporalavg":
        model = Swin2DTemporalAvg(num_classes=cfg["num_classes"])
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # different LR for head/backbone
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg["optim"]["lr_backbone"]},
        {"params": model.head.parameters(),     "lr": cfg["optim"]["lr_head"]},
    ], weight_decay=cfg["optim"]["weight_decay"])

    scaler = GradScaler('cuda', enabled=cfg["train"].get("amp", True))

    save_dir = Path(cfg["train"]["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(save_dir)

    best_top1 = 0.0
    print(f"Starting training for {cfg['train']['epochs']} epochs...")
    
    for epoch in range(1, cfg["train"]["epochs"]+1):
        logger.log_epoch_start(epoch)
        
        model.train()
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
            
            # Log iteration metrics
            current_lr = opt.param_groups[0]['lr']  # backbone lr
            logger.log_iteration(loss.item(), current_lr)

            if i % 20 == 0:
                avg_loss = running / i
                print(f"epoch {epoch} iter {i}/{len(train_ld)}  loss {avg_loss:.4f}")

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
        
        # Log epoch metrics
        logger.log_epoch_end(top1, top5)
        
        print(f"[epoch {epoch}] val top1={top1:.2f} top5={top5:.2f}  ({logger.metrics['train_times'][-1]:.1f}s)")

        # simple checkpoint
        if top1 > best_top1:
            best_top1 = top1
            ckpt = save_dir / "best.pt"
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "top1": best_top1, "cfg": cfg}, ckpt)
            print(f"🔥 saved {ckpt}  (top1={best_top1:.2f})")

    # Save final summary
    logger.save_final_summary(cfg, best_top1)
    print(f"\nTraining completed! Logs saved to: {save_dir}")
    print(f"Best Top-1 Accuracy: {best_top1:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args())
