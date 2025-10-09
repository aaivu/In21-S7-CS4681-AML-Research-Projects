"""
M4 Competition specific configuration.

This module provides configuration tailored for the M4 Competition benchmark,
including frequency-specific parameters and M4 evaluation metrics.
"""

from dataclasses import dataclass
from typing import Dict
from src.config.base_config import BaseConfig


@dataclass
class M4Config(BaseConfig):
    """
    Configuration for M4 Competition forecasting.

    This configuration is optimized for the M4 Competition benchmark with
    frequency-specific parameters and adjusted architecture for shorter series.
    """

    # M4 Competition forecast horizons by frequency
    M4_HORIZONS: Dict[str, int] = None

    def __init__(self, frequency: str = 'Monthly', **kwargs):
        """
        Initialize M4 configuration.

        Args:
            frequency: M4 frequency ('Yearly', 'Quarterly', 'Monthly', 'Weekly',
                      'Daily', 'Hourly')
            **kwargs: Additional configuration parameters to override defaults
        """
        # Define M4 official horizons
        self.M4_HORIZONS = {
            'Yearly': 6,
            'Quarterly': 8,
            'Monthly': 18,
            'Weekly': 13,
            'Daily': 14,
            'Hourly': 48
        }

        if frequency not in self.M4_HORIZONS:
            raise ValueError(
                f"Invalid frequency '{frequency}'. Must be one of: "
                f"{list(self.M4_HORIZONS.keys())}"
            )

        self.frequency = frequency
        self.pred_len = self.M4_HORIZONS[frequency]

        # M4-optimized architecture (smaller for faster inference)
        m4_defaults = {
            'd_model': 64,  # Reduced from 128
            'n_heads': 8,   # Reduced from 16
            'e_layers': 2,  # Reduced from 3
            'd_ff': 128,    # Reduced from 256
            'dropout': 0.1,
            'c_in': 1,      # Univariate series

            # Frequency-dependent patching
            'patch_len': self._get_patch_len(frequency),
            'seq_len': None,  # Will be computed based on patch_len

            # Training parameters (adjusted for M4)
            'batch_size': 64,
            'epochs': 10,
            'learning_rate': 1e-3,
            'patience': 3,

            # Data paths
            'data_root': 'data/m4',
            'checkpoint_dir': f'checkpoints/m4_{frequency.lower()}',
            'results_dir': f'results/m4_{frequency.lower()}',
        }

        # Merge M4 defaults with any provided kwargs
        config_params = {**m4_defaults, **kwargs}

        # Compute seq_len if not provided
        if config_params['seq_len'] is None:
            config_params['seq_len'] = config_params['patch_len'] * 6  # 6 cycles lookback

        # Compute stride if not provided
        if 'stride' not in config_params:
            config_params['stride'] = config_params['patch_len'] // 2  # 50% overlap

        # Initialize parent class
        super().__init__(**config_params)

    @staticmethod
    def _get_patch_len(frequency: str) -> int:
        """
        Get optimal patch length for M4 frequency.

        Args:
            frequency: M4 frequency

        Returns:
            Optimal patch length
        """
        patch_lengths = {
            'Yearly': 3,
            'Quarterly': 4,
            'Monthly': 12,
            'Weekly': 13,
            'Daily': 14,
            'Hourly': 24
        }
        return patch_lengths.get(frequency, 12)

    def get_data_files(self) -> tuple:
        """
        Get paths to M4 train and test files.

        Returns:
            Tuple of (train_file_path, test_file_path)
        """
        train_file = self.data_root / f'{self.frequency}-train.csv'
        test_file = self.data_root / f'{self.frequency}-test.csv'
        return str(train_file), str(test_file)

    def __repr__(self) -> str:
        """String representation of M4 configuration."""
        config_str = f"M4 Configuration ({self.frequency}):\n"
        config_str += "=" * 80 + "\n"

        sections = {
            "M4 Settings": ['frequency', 'pred_len'],
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
