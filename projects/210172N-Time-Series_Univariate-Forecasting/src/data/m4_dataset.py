"""
M4 Competition Dataset Loader.

This module provides data loading functionality for the M4 Competition benchmark,
handling variable-length time series and proper train/test splits.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


class M4Dataset(Dataset):
    """
    M4 Competition dataset with variable-length series support.

    The M4 dataset consists of 100,000 univariate time series with varying
    lengths. This loader handles:
    - Variable-length series (42 to 2,794 observations)
    - Proper train/test splits from separate files
    - Padding for shorter series
    - Series ID tracking

    Args:
        train_file: Path to M4 training data CSV
        seq_len: Input sequence length (look-back window)
        pred_len: Prediction horizon (forecast length)
        test_file: Optional path to M4 test data CSV (for evaluation)
        mode: Dataset mode ('train' or 'test')

    M4 File Format:
        - Train file: [series_id, value1, value2, ..., valueN, ...]
        - Test file: [series_id, future1, future2, ..., futureH]
          where H is the forecast horizon

    Examples:
        >>> # Training
        >>> train_dataset = M4Dataset(
        ...     train_file='Monthly-train.csv',
        ...     seq_len=72,
        ...     pred_len=18
        ... )
        >>>
        >>> # Testing
        >>> test_dataset = M4Dataset(
        ...     train_file='Monthly-train.csv',
        ...     seq_len=72,
        ...     pred_len=18,
        ...     test_file='Monthly-test.csv',
        ...     mode='test'
        ... )
    """

    def __init__(
        self,
        train_file: str,
        seq_len: int,
        pred_len: int,
        test_file: Optional[str] = None,
        mode: str = 'train'
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode

        # Load training data
        df_train = pd.read_csv(train_file)
        self.series_ids = df_train.iloc[:, 0].values
        train_data = df_train.iloc[:, 1:].values

        # Load test data if provided (for evaluation)
        test_data = None
        if test_file is not None:
            if not Path(test_file).exists():
                raise FileNotFoundError(f"Test file not found: {test_file}")
            df_test = pd.read_csv(test_file)
            test_data = df_test.iloc[:, 1:].values

        # Process and validate series
        self.valid_samples = []
        invalid_count = 0

        for i in range(len(train_data)):
            # Get training series and remove NaN
            train_series = train_data[i]
            train_series = train_series[~np.isnan(train_series)]

            if mode == 'train':
                # Training mode: need seq_len + pred_len
                if len(train_series) >= seq_len + pred_len:
                    self.valid_samples.append((i, train_series, None))
                else:
                    invalid_count += 1
            else:
                # Testing mode: need seq_len from train, pred_len from test
                if test_file is None:
                    raise ValueError("test_file required for mode='test'")

                if len(train_series) >= seq_len:
                    # Get test series and remove NaN
                    test_series = test_data[i]
                    test_series = test_series[~np.isnan(test_series)]

                    if len(test_series) >= pred_len:
                        self.valid_samples.append((i, train_series, test_series))
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1

        print(f"   M4Dataset ({mode}): {len(self.valid_samples)}/{len(train_data)} "
              f"valid series ({invalid_count} skipped)")

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Returns:
            Tuple of (input_sequence, target_sequence, series_id)
                - input_sequence: (seq_len, 1)
                - target_sequence: (pred_len, 1)
                - series_id: int
        """
        series_idx, train_series, test_series = self.valid_samples[idx]

        if self.mode == 'train':
            # Training: Use training data for both X and Y
            total_len = self.seq_len + self.pred_len

            # Take the last total_len points
            if len(train_series) > total_len:
                train_series = train_series[-total_len:]

            seq_x = train_series[:self.seq_len]
            seq_y = train_series[self.seq_len:self.seq_len + self.pred_len]

        else:
            # Testing: X from train (last seq_len), Y from test (first pred_len)
            seq_x = train_series[-self.seq_len:]
            seq_y = test_series[:self.pred_len]

        # Reshape to (seq_len, 1) for univariate series
        seq_x = seq_x.reshape(-1, 1)
        seq_y = seq_y.reshape(-1, 1)

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            series_idx
        )

    def get_series_id(self, idx: int) -> str:
        """Get the series ID for a given index."""
        series_idx, _, _ = self.valid_samples[idx]
        return self.series_ids[series_idx]


def create_m4_dataloaders(
    train_file: str,
    test_file: str,
    seq_len: int,
    pred_len: int,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create M4 train and test data loaders.

    Args:
        train_file: Path to M4 training CSV
        test_file: Path to M4 test CSV
        seq_len: Input sequence length
        pred_len: Prediction horizon
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for loading
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, test_loader)

    Examples:
        >>> train_loader, test_loader = create_m4_dataloaders(
        ...     train_file='Monthly-train.csv',
        ...     test_file='Monthly-test.csv',
        ...     seq_len=72,
        ...     pred_len=18,
        ...     batch_size=64
        ... )
    """
    # Create datasets
    train_dataset = M4Dataset(
        train_file=train_file,
        seq_len=seq_len,
        pred_len=pred_len,
        mode='train'
    )

    test_dataset = M4Dataset(
        train_file=train_file,
        seq_len=seq_len,
        pred_len=pred_len,
        test_file=test_file,
        mode='test'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
