import os
import urllib.request

import numpy as np
import torch
from typing import Optional

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _get_transform(dataset_name: str):
    """Return dataset-specific transforms."""
    name = dataset_name.lower()
    if name == 'cifar-10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if name in ('mnist', 'emnist', 'femnist'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    if name == 'shakespeare':
        return None
    raise ValueError(f"Unsupported dataset for transforms: {dataset_name}")


def _split_dataset(
    dataset,
    num_clients: int,
    non_iid: bool,
    shards_per_client: int = 2,
    seed: Optional[int] = None,
):
    """Split a dataset into subsets for each client."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if num_clients <= 1:
        return [dataset]

    indices = np.arange(len(dataset))
    targets = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)

    if non_iid:
        # Sort by labels and divide into shards
        sorted_indices = indices[np.argsort(targets)]
        shards = np.array_split(sorted_indices, num_clients * shards_per_client)
        np.random.shuffle(shards)
        client_indices = [
            np.concatenate(shards[i * shards_per_client:(i + 1) * shards_per_client])
            for i in range(num_clients)
        ]
    else:
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)

    return [Subset(dataset, idx) for idx in client_indices]


def load_data(
    dataset_name: str,
    batch_size: int,
    num_clients: int = 1,
    non_iid: bool = False,
    shards_per_client: int = 2,
    seed: Optional[int] = None,
):
    """Load dataset and return DataLoaders for clients and test set."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    transform = _get_transform(dataset_name)
    name = dataset_name.upper()

    if name == 'CIFAR-10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif name in ('EMNIST', 'FEMNIST'):
        train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)
    elif name == 'SHAKESPEARE':
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        data_dir = './data'
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, 'shakespeare.txt')
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

        split_idx = int(0.9 * len(encoded))
        train_data = encoded[:split_idx]
        test_data = encoded[split_idx:]
        seq_len = 80

        class ShakespeareDataset(Dataset):
            def __init__(self, data, seq_len):
                self.data = data
                self.seq_len = seq_len
                self.targets = data[seq_len:]

            def __len__(self):
                return len(self.data) - self.seq_len

            def __getitem__(self, idx):
                x = self.data[idx:idx + self.seq_len]
                # Use the next character as the target rather than a full sequence
                y = self.data[idx + self.seq_len]
                return x, y

        train_dataset = ShakespeareDataset(train_data, seq_len)
        test_dataset = ShakespeareDataset(test_data, seq_len)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    subsets = _split_dataset(
        train_dataset,
        num_clients,
        non_iid,
        shards_per_client=shards_per_client,
        seed=seed,
    )
    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets
    ]
    if num_clients == 1:
        train_loaders = train_loaders[0]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loaders, test_loader
