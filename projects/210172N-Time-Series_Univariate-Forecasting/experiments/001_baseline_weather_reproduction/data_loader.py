"""
Weather dataset loader matching PatchTST original implementation
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import config


class WeatherDataset(Dataset):
    def __init__(self, flag='train'):
        """
        flag: 'train', 'val', or 'test'
        Matches Dataset_Custom from PatchTST data_provider/data_loader.py
        """
        self.flag = flag
        self.scaler = StandardScaler()

        # Load data
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = os.path.join(project_root, 'data', 'secondary', 'weather', 'weather.csv')
        df_raw = pd.read_csv(data_path)

        # Reorganize columns: ['date', ...other features, target]
        cols = list(df_raw.columns)
        cols.remove(config.TARGET)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [config.TARGET]]

        # Calculate borders (70% train, 10% val, 20% test) - matching PatchTST lines 234-240
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[flag]

        border1s = [0, num_train - config.SEQ_LEN, len(df_raw) - num_test - config.SEQ_LEN]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[set_type]
        border2 = border2s[set_type]

        # Select features
        if config.FEATURES == 'M' or config.FEATURES == 'MS':
            cols_data = df_raw.columns[1:]  # All except date
            df_data = df_raw[cols_data]
        elif config.FEATURES == 'S':
            df_data = df_raw[[config.TARGET]]

        # Scale using training data statistics
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        # Extract split data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __len__(self):
        return len(self.data_x) - config.SEQ_LEN - config.PRED_LEN + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + config.SEQ_LEN
        r_begin = s_end
        r_end = r_begin + config.PRED_LEN

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def get_dataloaders():
    """Get train, val, test dataloaders"""
    train_dataset = WeatherDataset('train')
    val_dataset = WeatherDataset('val')
    test_dataset = WeatherDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.scaler


if __name__ == "__main__":
    # Test
    train_loader, val_loader, test_loader, scaler = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check one batch
    for x, y in train_loader:
        print(f"\nBatch shapes:")
        print(f"  x: {x.shape}  # (batch, seq_len, features)")
        print(f"  y: {y.shape}  # (batch, pred_len, features)")
        break
