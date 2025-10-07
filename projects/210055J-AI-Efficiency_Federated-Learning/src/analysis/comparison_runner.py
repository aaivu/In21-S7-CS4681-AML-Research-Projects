from __future__ import annotations

from typing import Any, Dict
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from utils.data_loader import load_data
from .metadata import create_run_dir
from .reporting import save_results

from utils.experiment_runner import train_algorithm, create_model


def _seed_everything(seed: int | None) -> None:
    """Seed all random number generators for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_comparison(
    config: Dict[str, Any],
    base_path: Path,
    tag: str | None = None,
) -> Dict[str, Any]:
    """Compare baseline FedAvg with knowledge distillation variant.

    The function trains two separate models using the provided configuration:
    the standard Federated Averaging baseline (``fedavg``) and the knowledge
    distillation enhanced version (``fedavg_kd``). Per-epoch accuracy and loss
    are collected for each algorithm along with the final model weights.

    Parameters
    ----------
    config: dict
        Experiment configuration containing dataset and hyperparameters.
    base_path: Path
        Root directory where the ``results`` folder will be created.
    tag: str, optional
        Optional identifier for the comparison run. If not provided a timestamp
        based tag is generated.

    Returns
    -------
    dict
        Dictionary holding metrics and final model states for both algorithms.
    """
    if tag is not None:
        config.setdefault("tag", tag)

    run_dir = create_run_dir(base_path, config)
    tag = run_dir.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train_loaders, test_loader = load_data(
            config["dataset_name"],
            config["batch_size"],
            num_clients=config.get("num_clients", 1),
            non_iid=config.get("non_iid", True),
            shards_per_client=config.get("shards_per_client", 2),
            seed=config.get("seed"),
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return an empty dictionary to avoid the AttributeError
        return {}

    if isinstance(train_loaders, DataLoader):
        train_loaders = [train_loaders]

    results: Dict[str, Any] = {"tag": tag}
    for algorithm in ["fedavg", "fedavg_kd"]:
        print(f"Running algorithm: {algorithm}")
        _seed_everything(config.get("seed"))
        global_model = create_model(config)
        
        # FIX: Wrap the train_algorithm call in a try-except block
        try:
            algorithm_results = train_algorithm(
                algorithm, config, global_model, train_loaders, test_loader, device
            )
            # Check if train_algorithm returned a valid dictionary
            if isinstance(algorithm_results, dict):
                results[algorithm] = algorithm_results
            else:
                print(f"Warning: train_algorithm for {algorithm} did not return a dictionary. Got: {type(algorithm_results)}")
                # Assign an empty dictionary to prevent the AttributeError in reporting.py
                results[algorithm] = {"metrics": {}, "model_state": None}
                
        except Exception as e:
            print(f"Error running {algorithm} algorithm: {e}")
            # Ensure the results dictionary is still populated, even on failure
            results[algorithm] = {"metrics": {}, "model_state": None}

    # Now that results is guaranteed to be a dictionary, we can save them
    save_results(results, run_dir)

    return results
