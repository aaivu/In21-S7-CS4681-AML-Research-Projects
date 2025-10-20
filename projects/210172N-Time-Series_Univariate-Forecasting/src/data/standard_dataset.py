"""
Standard Dataset Loader for Long-Term Time Series Forecasting.

This module provides data loading for standard LTSF benchmarks like
Weather, Traffic, Electricity, ETT, etc.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


class StandardDataset(Dataset):
    """
    Standard time series dataset for LTSF benchmarks.

    This loader handles standard CSV-format time series datasets with:
    - Fixed-length series
    - Standard train/val/test splits (70%/10%/20%)
    - Optional date column
    - Multiple features/channels

    Args:
        data_file: Path to dataset CSV file
        seq_len: Input sequence length
        pred_len: Prediction horizon
        c_in: Number of input features/channels
        flag: Dataset split ('train', 'val', or 'test')
        target_feature: Name of target feature (if univariate prediction)
        scale: Whether to standardize the data

    File Format:
        CSV with columns: [date (optional), feature1, feature2, ..., featureN]

    Examples:
        >>> train_dataset = StandardDataset(
        ...     data_file='weather.csv',
        ...     seq_len=336,
        ...     pred_len=96,
        ...     c_in=21,
        ...     flag='train'
        ... )
    """

    def __init__(
        self,
        data_file: str,
        seq_len: int,
        pred_len: int,
        c_in: int,
        flag: str = 'train',
        target_feature: Optional[str] = None,
        scale: bool = True
    ):
        if flag not in ['train', 'val', 'test']:
            raise ValueError(f"flag must be 'train', 'val', or 'test', got '{flag}'")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.flag = flag
        self.scale = scale

        # Load data
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        df_raw = pd.read_csv(data_file)

        # Remove date column if present
        if 'date' in df_raw.columns:
            df_raw = df_raw.drop(columns=['date'])

        # Reorder columns if target feature specified
        if target_feature and target_feature in df_raw.columns:
            cols = list(df_raw.columns)
            cols.remove(target_feature)
            df_raw = df_raw[cols + [target_feature]]

        # Define split borders (70% train, 10% val, 20% test)
        n = len(df_raw)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        # Split boundaries
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]

        border1s = [0, train_end - seq_len, n - int(n * 0.2) - seq_len]
        border2s = [train_end, val_end, n]

        border1 = border1s[set_type]
        border2 = border2s[set_type]

        # Extract data
        data = df_raw.values

        # Standardization (using training statistics)
        if scale:
            train_data = data[:train_end]
            self.mean = train_data.mean(axis=0)
            self.std = train_data.std(axis=0)
            data = (data - self.mean) / self.std
        else:
            self.mean = np.zeros(data.shape[1])
            self.std = np.ones(data.shape[1])

        # Extract split
        self.data = data[border1:border2]

        print(f"   StandardDataset ({flag}): {len(self)} samples from {data_file}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Tuple of (input_sequence, target_sequence)
                - input_sequence: (seq_len, c_in)
                - target_sequence: (pred_len, c_in)
        """
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse standardization transform.

        Args:
            data: Standardized data

        Returns:
            Original scale data
        """
        if self.scale:
            return data * self.std + self.mean
        return data


def create_standard_dataloaders(
    data_file: str,
    seq_len: int,
    pred_len: int,
    c_in: int,
    batch_size: int = 128,
    num_workers: int = 0,
    target_feature: Optional[str] = None,
    scale: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for standard datasets.

    Args:
        data_file: Path to dataset CSV file
        seq_len: Input sequence length
        pred_len: Prediction horizon
        c_in: Number of input features
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        target_feature: Name of target feature (optional)
        scale: Whether to standardize the data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Examples:
        >>> train_loader, val_loader, test_loader = create_standard_dataloaders(
        ...     data_file='weather.csv',
        ...     seq_len=336,
        ...     pred_len=96,
        ...     c_in=21,
        ...     batch_size=128
        ... )
    """
    # Create datasets
    train_dataset = StandardDataset(
        data_file=data_file,
        seq_len=seq_len,
        pred_len=pred_len,
        c_in=c_in,
        flag='train',
        target_feature=target_feature,
        scale=scale
    )

    val_dataset = StandardDataset(
        data_file=data_file,
        seq_len=seq_len,
        pred_len=pred_len,
        c_in=c_in,
        flag='val',
        target_feature=target_feature,
        scale=scale
    )

    test_dataset = StandardDataset(
        data_file=data_file,
        seq_len=seq_len,
        pred_len=pred_len,
        c_in=c_in,
        flag='test',
        target_feature=target_feature,
        scale=scale
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    return train_loader, val_loader, test_loader
