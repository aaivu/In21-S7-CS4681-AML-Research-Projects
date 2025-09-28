import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

# Ensure the project root is on the Python path when running tests
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

from models.architectures import CIFARCNN
from training.client_update import client_update
from training.client_update_baseline import client_update_baseline
from training.server_aggregation import server_aggregation
from utils.data_loader import load_data
from utils.evaluation import evaluate_model
from utils.loss_function import cross_entropy_loss


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(seed: int, batch_size: int,
                 num_clients: int = 2,
                 subset_train: int = 100,
                 subset_test: int = 200):
    """Load CIFAR-10 and keep only small subsets for quick tests."""
    train_loaders, test_loader = load_data(
        'CIFAR-10',
        batch_size,
        num_clients=num_clients,
        non_iid=True,
        shards_per_client=1,
        seed=seed,
    )

    new_train_loaders = []
    for loader in train_loaders:
        indices = list(range(subset_train))
        subset = Subset(loader.dataset, indices)
        new_train_loaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=True)
        )
    subset = Subset(test_loader.dataset, range(subset_test))
    test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return new_train_loaders, test_loader


def run_fedavg(train_loaders, test_loader, epochs: int, lr: float, device, local_epochs: int):
    model = CIFARCNN().to(device)
    for _ in range(epochs):
        client_weights, client_sizes = [], []
        for loader in train_loaders:
            weights = client_update_baseline(model, loader, lr, device, local_epochs)
            client_weights.append(weights)
            client_sizes.append(len(loader.dataset))
        aggregated = server_aggregation(client_weights, client_sizes)
        model.load_state_dict(aggregated)
    acc, _ = evaluate_model(model, test_loader, cross_entropy_loss, device)
    return acc


def run_fedavg_kd(train_loaders, test_loader, epochs: int,
                  lr: float, lambda_: float, T: float,
                  tau: float, device, local_epochs: int):
    model = CIFARCNN().to(device)
    for _ in range(epochs):
        client_weights, client_sizes = [], []
        for loader in train_loaders:
            weights = client_update(
                model, loader, lambda_, T, tau, lr, device, local_epochs
            )
            client_weights.append(weights)
            client_sizes.append(len(loader.dataset))
        aggregated = server_aggregation(client_weights, client_sizes)
        model.load_state_dict(aggregated)
    acc, _ = evaluate_model(model, test_loader, cross_entropy_loss, device)
    return acc


def test_kd_improves_over_baseline():
    seed = 0
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = CONFIG["batch_size"]
    train_loaders, test_loader = prepare_data(seed, batch_size=batch_size)
    baseline_acc = run_fedavg(
        train_loaders, test_loader, epochs=2, lr=0.01, device=device, local_epochs=1
    )

    # Re-create loaders so that shuffling order matches baseline run
    train_loaders, test_loader = prepare_data(seed, batch_size=batch_size)
    kd_acc = run_fedavg_kd(
        train_loaders,
        test_loader,
        epochs=2,
        lr=0.01,
        lambda_=0.5,
        T=2.0,
        tau=0.9,
        device=device,
        local_epochs=1,
    )

    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    print(f"KD accuracy: {kd_acc:.2f}%")
    assert kd_acc >= baseline_acc + 1.0
