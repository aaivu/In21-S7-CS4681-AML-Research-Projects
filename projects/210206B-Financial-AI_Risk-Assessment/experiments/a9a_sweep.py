#!/usr/bin/env python3
import os
import time
import json
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib
from qhoptim.pyt import QHAdam


def timestamp():
    return time.strftime('%Y.%m.%d_%H-%M-%S')


def build_model(in_features, device):
    model = nn.Sequential(
        lib.DenseBlock(in_features, 128, num_layers=8, tree_dim=3, depth=6, flatten_output=False,
                       choice_function=lib.entmax15, bin_function=lib.entmoid15),
        lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),
    ).to(device)
    # data-aware init will happen in trainer via first batches
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def load_a9a(data_dir, random_state=1337):
    from sklearn.datasets import load_svmlight_file
    from sklearn.preprocessing import QuantileTransformer

    train_path = os.path.join(data_dir, 'a9a')
    test_path = os.path.join(data_dir, 'a9a.t')
    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=123)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=123)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    # convert {-1,1} -> {0,1} if needed
    if set(np.unique(y_train)) == {-1.0, 1.0}:
        y_train = (y_train + 1) // 2
        y_test = (y_test + 1) // 2

    train_idx_path = os.path.join(data_dir, 'stratified_train_idx.txt')
    valid_idx_path = os.path.join(data_dir, 'stratified_valid_idx.txt')
    train_indices = np.loadtxt(train_idx_path, dtype=int)
    valid_indices = np.loadtxt(valid_idx_path, dtype=int)

    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_valid = X_train[valid_indices]
    y_valid = y_train[valid_indices]

    qt = QuantileTransformer(output_distribution='normal', random_state=random_state)
    X_train_split = qt.fit_transform(X_train_split)
    X_valid = qt.transform(X_valid)
    X_test = qt.transform(X_test)

    class Data:
        def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
            self.X_train = X_train
            self.y_train = y_train
            self.X_valid = X_valid
            self.y_valid = y_valid
            self.X_test = X_test
            self.y_test = y_test

    return Data(X_train_split, y_train_split, X_valid, y_valid, X_test, y_test)


def steps_per_epoch(num_samples, batch_size):
    return (num_samples + batch_size - 1) // batch_size


def train_and_eval(data, device, experiment_name, trainer_kwargs, batch_size=256, max_epochs=100):
    model = build_model(data.X_train.shape[1], device)

    optimizer_params = {
        'lr': trainer_kwargs.pop('lr', 0.01),
        'nus': (0.7, 1.0),
        'betas': (0.95, 0.998)
    }

    trainer = lib.Trainer(
        model=model,
        loss_function=F.binary_cross_entropy_with_logits,
        experiment_name=experiment_name,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        verbose=True,
        n_last_checkpoints=5,
        **trainer_kwargs
    )

    best_auc = -1.0
    best_step = 0
    for epoch in range(max_epochs):
        model.train()
        for batch in lib.iterate_minibatches(data.X_train, data.y_train, batch_size=batch_size, shuffle=True):
            trainer.train_on_batch(*batch, device=device)

        if (epoch % 5) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            auc = trainer.evaluate_auc(data.X_valid, data.y_valid, device=device, batch_size=1024, use_amp=True)
            if auc > best_auc:
                best_auc = auc
                best_step = trainer.step
                trainer.save_checkpoint(tag='best')
            trainer.save_checkpoint()
        # simple early stopping when no improvement for 50 eval steps
        if trainer.step > best_step + 50 * steps_per_epoch(len(data.X_train), batch_size):
            break

    trainer.load_checkpoint(tag='best')
    test_auc = trainer.evaluate_auc(data.X_test, data.y_test, device=device, batch_size=1024, use_amp=True)
    test_err = trainer.evaluate_classification_error(data.X_test, data.y_test, device=device, batch_size=1024, use_amp=True)
    return {'test_auc': float(test_auc), 'test_error': float(test_err), 'best_step': int(best_step)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'A9A'))
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help='Path to baseline log dir, e.g. E:/Hari/node/notebooks/logs/a9a_node_8layers_2025.10.01_14-57')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out_csv', type=str, default='a9a_sweep_results.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (args.device if args.device != 'auto' else 'cpu')
    data = load_a9a(args.data_dir)

    num_train = len(data.X_train)
    steps_ep = steps_per_epoch(num_train, args.batch_size)
    total_steps = steps_ep * args.epochs

    # Define sweep grid (small, controlled)
    losses = [
        {'loss_type': None, 'label_smoothing': 0.0},
        {'loss_type': None, 'label_smoothing': 0.05},
        {'loss_type': 'focal', 'focal_gamma': 1.5},
        {'loss_type': 'focal', 'focal_gamma': 2.0},
    ]
    lrs = [0.005, 0.01]
    warmup_ratios = [0.05, 0.1]

    results = []
    for loss_cfg, lr, wr in itertools.product(losses, lrs, warmup_ratios):
        trainer_kwargs = {
            'scheduler_type': 'warmup_cosine',
            'warmup_steps': int(wr * total_steps),
            'total_steps': int(total_steps),
            'min_lr': 1e-5,
            'notes_baseline_path': args.baseline_dir,
            'lr': lr
        }
        trainer_kwargs.update(loss_cfg)

        tag = [
            f"lr{lr}",
            f"wr{int(wr*100)}",
        ]
        if loss_cfg.get('loss_type') == 'focal':
            tag.append(f"focal{loss_cfg.get('focal_gamma', 2.0)}")
        else:
            tag.append(f"ls{loss_cfg.get('label_smoothing', 0.0)}")

        experiment_name = f"a9a_sweep_{'-'.join(tag)}_{timestamp()}"
        print(f"\n=== Running {experiment_name} ===")
        metrics = train_and_eval(data, device, experiment_name, trainer_kwargs, batch_size=args.batch_size, max_epochs=args.epochs)
        row = {
            'experiment': experiment_name,
            'lr': lr,
            'warmup_ratio': wr,
            **{k: v for k, v in loss_cfg.items()},
            **metrics
        }
        results.append(row)
        # write incremental csv
        try:
            import pandas as pd
            pd.DataFrame(results).to_csv(args.out_csv, index=False)
        except Exception:
            pass

    # Final print
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        print(df.sort_values('test_auc', ascending=False).to_string(index=False))
        df.to_csv(args.out_csv, index=False)
        with open('a9a_sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()




