"""
Standard dataset configuration (Weather, Traffic, Electricity, etc.).

This module provides configuration for standard long-term time series forecasting
datasets commonly used in benchmarking.
"""

from dataclasses import dataclass
from typing import Dict, List
from src.config.base_config import BaseConfig


@dataclass
class StandardConfig(BaseConfig):
    """
    Configuration for standard LTSF datasets.

    This configuration is optimized for long-term time series forecasting
    datasets like Weather, Traffic, Electricity, ETT, etc.
    """

    # Standard dataset specifications
    DATASET_SPECS: Dict[str, Dict] = None

    def __init__(self, dataset: str = 'weather', pred_len: int = 96, **kwargs):
        """
        Initialize standard dataset configuration.

        Args:
            dataset: Dataset name ('weather', 'traffic', 'electricity', 'illness',
                    'exchange_rate', 'etth1', 'etth2', 'ettm1', 'ettm2')
            pred_len: Prediction horizon (96, 192, 336, 720)
            **kwargs: Additional configuration parameters to override defaults
        """
        self.DATASET_SPECS = {
            'weather': {
                'file': 'weather.csv',
                'c_in': 21,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/weather'
            },
            'traffic': {
                'file': 'traffic.csv',
                'c_in': 862,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/traffic'
            },
            'electricity': {
                'file': 'electricity.csv',
                'c_in': 321,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/electricity'
            },
            'illness': {
                'file': 'national_illness.csv',
                'c_in': 7,
                'seq_len': 36,
                'pred_lens': [24, 36, 48, 60],
                'data_path': 'data/secondary/illness'
            },
            'exchange_rate': {
                'file': 'exchange_rate.csv',
                'c_in': 8,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/exchange_rate'
            },
            'etth1': {
                'file': 'ETTh1.csv',
                'c_in': 7,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/ETT-small'
            },
            'etth2': {
                'file': 'ETTh2.csv',
                'c_in': 7,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/ETT-small'
            },
            'ettm1': {
                'file': 'ETTm1.csv',
                'c_in': 7,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/ETT-small'
            },
            'ettm2': {
                'file': 'ETTm2.csv',
                'c_in': 7,
                'seq_len': 336,
                'pred_lens': [96, 192, 336, 720],
                'data_path': 'data/secondary/ETT-small'
            },
        }

        dataset = dataset.lower()
        if dataset not in self.DATASET_SPECS:
            raise ValueError(
                f"Invalid dataset '{dataset}'. Must be one of: "
                f"{list(self.DATASET_SPECS.keys())}"
            )

        spec = self.DATASET_SPECS[dataset]

        # Validate prediction length
        if pred_len not in spec['pred_lens']:
            raise ValueError(
                f"Invalid pred_len {pred_len} for {dataset}. "
                f"Valid options: {spec['pred_lens']}"
            )

        self.dataset = dataset
        self.pred_len = pred_len

        # Dataset-specific defaults
        dataset_defaults = {
            'c_in': spec['c_in'],
            'seq_len': spec['seq_len'],
            'patch_len': 16,
            'stride': 8,

            # Standard architecture (full-size for maximum accuracy)
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,

            # Training parameters
            'batch_size': 128,
            'epochs': 20,
            'learning_rate': 1e-4,
            'patience': 5,

            # Data paths
            'data_root': spec['data_path'],
            'checkpoint_dir': f'checkpoints/{dataset}_pred{pred_len}',
            'results_dir': f'results/{dataset}_pred{pred_len}',
        }

        # Merge dataset defaults with any provided kwargs
        config_params = {**dataset_defaults, **kwargs}

        # Initialize parent class
        super().__init__(**config_params)

        # Store dataset file path
        self.data_file = spec['file']

    def get_data_file(self) -> str:
        """
        Get path to dataset file.

        Returns:
            Path to the dataset CSV file
        """
        return str(self.data_root / self.data_file)

    @staticmethod
    def get_available_datasets() -> List[str]:
        """Get list of available datasets."""
        config = StandardConfig.__new__(StandardConfig)
        config.DATASET_SPECS = StandardConfig(dataset='weather').DATASET_SPECS
        return list(config.DATASET_SPECS.keys())

    def __repr__(self) -> str:
        """String representation of standard dataset configuration."""
        config_str = f"Standard Configuration ({self.dataset.upper()}):\n"
        config_str += "=" * 80 + "\n"

        sections = {
            "Dataset Settings": ['dataset', 'pred_len', 'data_file'],
            "Model Architecture": ['d_model', 'n_heads', 'e_layers', 'd_ff', 'dropout'],
            "Patching": ['patch_len', 'stride', 'patch_num'],
            "Time Series": ['seq_len', 'c_in'],
            "Training": ['batch_size', 'epochs', 'learning_rate', 'patience'],
            "System": ['device', 'seed']
        }

        for section, params in sections.items():
            config_str += f"\n{section}:\n"
            config_str += "-" * 40 + "\n"
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    config_str += f"  {param:20s}: {value}\n"

        config_str += "=" * 80
        return config_str
