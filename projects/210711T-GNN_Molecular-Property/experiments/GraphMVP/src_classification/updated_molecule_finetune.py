from os.path import join
import os, random, csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import args
from models import GNN, GNN_graphpred
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    roc_auc_score,
)
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader
from util import get_num_task

from datasets import MoleculeDataset


# -------------------------- utils & losses --------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """Binary focal loss that works with logits and supports per-task pos_weight."""
    def __init__(self, pos_weight=None, gamma=1.5, reduction='none'):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE per element with logits; include pos_weight if provided
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p)
        loss = (1.0 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_pos_weight(loader, num_tasks, device):
    """Compute pos_weight = #neg / #pos for each task, using TRAIN split only."""
    pos = torch.zeros(num_tasks, dtype=torch.float32)
    neg = torch.zeros(num_tasks, dtype=torch.float32)
    for batch in loader:
        y = batch.y.view(-1, num_tasks)
        valid = (y.abs() > 0)
        pos += ((y == 1) & valid).sum(0).float()
        neg += ((y == -1) & valid).sum(0).float()
    # avoid div-by-zero; default to 1.0 when no positives observed
    denom = pos.clone()
    denom[denom == 0] = 1.0
    pw = torch.where(pos > 0, neg / denom, torch.ones_like(neg))
    return pw.to(device)


def make_balanced_sampler(dataset):
    """WeightedRandomSampler for single-task binary labels (-1, 1, 0=missing)."""
    labels = []
    for i in range(len(dataset)):
        # dataset[i].y is scalar tensor for HIV
        labels.append(int(dataset[i].y.item()))
    labels = np.asarray(labels)

    mask = labels != 0
    pos = (labels[mask] == 1).sum()
    neg = (labels[mask] == -1).sum()
    if pos == 0 or neg == 0:
        return None  # fall back to shuffling if dataset is degenerate

    # inverse-frequency weights on observed (non-missing) examples
    w_pos = neg / (pos + neg)
    w_neg = pos / (pos + neg)
    weights = np.where(labels == 1, w_pos, np.where(labels == -1, w_neg, 0.0))
    weights = torch.tensor(weights, dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=int(mask.sum()), replacement=True
    )
    return sampler


# -------------------------- train / eval --------------------------

def train(model, device, loader, optimizer, criterion, use_amp=False, scaler=None):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape).float()

            # mask out missing targets (0 encoded as -1/1 in raw, shifted to 0/1)
            is_valid = (y ** 2) > 0
            loss_mat = criterion(pred, (y + 1) / 2)  # targets -> {0,1}
            loss = torch.sum(torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))) / is_valid.sum()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item())

    return total_loss / max(1, len(loader))


def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list, pr_list = [], []
    for i in range(y_true.shape[1]):
        # valid if both pos and neg exist
        if (y_true[:, i] == 1).sum() > 0 and (y_true[:, i] == -1).sum() > 0:
            mask = (y_true[:, i] ** 2) > 0
            yt = (y_true[mask, i] + 1) / 2  # -> {0,1}
            yp = y_scores[mask, i]
            roc_list.append(roc_auc_score(yt, yp))
            pr_list.append(average_precision_score(yt, yp))
        else:
            print(f'{i} is invalid')

    if len(roc_list) == 0:
        return 0.0, 0.0, y_true, y_scores

    return float(np.mean(roc_list)), float(np.mean(pr_list)), y_true, y_scores


# -------------------------- main --------------------------

if __name__ == '__main__':
    # robust seeding + device
    set_seed(getattr(args, 'runseed', 0))
    device = torch.device('cuda:' + str(getattr(args, 'device', 0))) \
        if torch.cuda.is_available() else torch.device('cpu')

    # dataset
    num_tasks = get_num_task(args.dataset)
    dataset_root = getattr(args, 'input_data_dir', None) or '../datasets/molecule_datasets'
    dataset = MoleculeDataset(join(dataset_root, args.dataset), dataset=args.dataset)
    print(dataset)

    # splits
    if args.split == 'scaffold':
        smiles_csv = join(dataset_root, args.dataset, 'processed', 'smiles.csv')
        smiles_list = pd.read_csv(smiles_csv, header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1
        )
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=getattr(args, 'seed', 42)
        )
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_csv = join(dataset_root, args.dataset, 'processed', 'smiles.csv')
        smiles_list = pd.read_csv(smiles_csv, header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=getattr(args, 'seed', 42)
        )
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])

    # loaders (with optional balanced sampler)
    train_sampler = make_balanced_sampler(train_dataset) if getattr(args, 'balance_sampler', False) else None
    if train_sampler is None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)

    val_loader  = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    molecule_model = GNN(
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type
    )
    model = GNN_graphpred(args=args, num_tasks=num_tasks, molecule_model=molecule_model)
    if getattr(args, 'input_model_file', '') != '':
        model.from_pretrained(args.input_model_file)
    model.to(device)
    print(model)

    # optimizer (separate LR for pred head)
    model_param_group = [
        {'params': model.molecule_model.parameters()},
        {'params': model.graph_pred_linear.parameters(), 'lr': args.lr * args.lr_scale},
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    # criterion: pos_weight from TRAIN set only; optional focal
    pos_weight = compute_pos_weight(train_loader, num_tasks, device)
    use_focal = getattr(args, 'focal', False)
    gamma = float(getattr(args, 'focal_gamma', 1.5))
    if use_focal:
        criterion = FocalLoss(pos_weight=pos_weight, gamma=gamma, reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    # AMP + scheduler + early stopping
    use_amp = bool(getattr(args, 'amp', False) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=int(getattr(args, 'lr_patience', 5)), verbose=True
    )
    patience = int(getattr(args, 'patience', 10))

    # prepare output dir & CSV
    if getattr(args, 'output_model_dir', '') != '':
        os.makedirs(args.output_model_dir, exist_ok=True)
        log_csv = join(args.output_model_dir, 'log.csv')
        if not os.path.exists(log_csv):
            with open(log_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch','loss','train_roc','train_pr','val_roc','val_pr','test_roc','test_pr','lr'])

    # training loop
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    best_val_roc, best_val_idx = -1.0, 0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer, criterion, use_amp=use_amp, scaler=scaler)
        print(f'Epoch: {epoch}\nLoss: {loss_acc}')

        if getattr(args, 'eval_train', True):
            train_roc, train_pr, train_target, train_pred = eval(model, device, train_loader)
        else:
            train_roc, train_pr = 0.0, 0.0
        val_roc, val_pr, val_target, val_pred   = eval(model, device, val_loader)
        test_roc, test_pr, test_target, test_pred = eval(model, device, test_loader)

        # choose threshold on val by maximizing F1 (HIV is single-task)
        val_thr = 0.5
        try:
            vy = (val_target + 1) / 2
            vp = val_pred
            p, r, thr = precision_recall_curve(vy[:, 0], vp[:, 0])
            f1s = 2 * p * r / (p + r + 1e-8)
            j = int(np.nanargmax(f1s))
            if j < len(thr):  # last point in PR curve has no threshold
                val_thr = float(thr[j])
        except Exception:
            pass
        ty = (test_target + 1) / 2
        that = (test_pred[:, 0] >= val_thr).astype(np.float32)
        test_f1 = f1_score(ty[:, 0], that)

        train_roc_list.append(train_roc)
        val_roc_list.append(val_roc)
        test_roc_list.append(test_roc)

        print(f'train: ROC {train_roc:.6f}  PR {train_pr:.6f} | '
              f'val: ROC {val_roc:.6f}  PR {val_pr:.6f} | '
              f'test: ROC {test_roc:.6f}  PR {test_pr:.6f} | '
              f'F1@valThr {test_f1:.4f} (thr={val_thr:.3f})\n')

        # CSV log
        if getattr(args, 'output_model_dir', '') != '':
            with open(join(args.output_model_dir, 'log.csv'), 'a', newline='') as f:
                csv.writer(f).writerow([epoch, loss_acc, train_roc, train_pr, val_roc, val_pr, test_roc, test_pr, optimizer.param_groups[0]['lr']])

        # scheduler + early stopping on val ROC
        scheduler.step(val_roc)
        improved = (val_roc > best_val_roc + 1e-4)
        if improved:
            best_val_roc, best_val_idx, epochs_no_improve = val_roc, epoch - 1, 0
            if getattr(args, 'output_model_dir', '') != '':
                torch.save(
                    {'molecule_model': molecule_model.state_dict(), 'model': model.state_dict()},
                    join(args.output_model_dir, 'model_best.pth')
                )
                np.savez(
                    join(args.output_model_dir, 'evaluation_best.npz'),
                    val_target=val_target, val_pred=val_pred,
                    test_target=test_target, test_pred=test_pred
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch} (best val ROC={best_val_roc:.4f}).')
                break

    print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
        train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]
    ))

    if getattr(args, 'output_model_dir', '') != '':
        output_model_path = join(args.output_model_dir, 'model_final.pth')
        torch.save(
            {'molecule_model': molecule_model.state_dict(), 'model': model.state_dict()},
            output_model_path
        )
