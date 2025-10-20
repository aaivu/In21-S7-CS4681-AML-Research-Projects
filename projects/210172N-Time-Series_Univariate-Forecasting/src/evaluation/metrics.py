"""
Evaluation metrics for time series forecasting.

Includes standard metrics (MSE, MAE, RMSE, MAPE) and M4 Competition metrics
(sMAPE, MASE, OWA).
"""

import numpy as np
from typing import Dict, Optional


# ============================================================================
# Standard Metrics
# ============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


# ============================================================================
# M4 Competition Metrics
# ============================================================================

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (M4 primary metric).

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        sMAPE value (0-100)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)

    # Avoid division by zero
    smape_val = np.where(denominator == 0, 0, diff / denominator)

    return 100 * np.mean(smape_val)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonal_period: int = 1
) -> float:
    """
    Mean Absolute Scaled Error (M4 metric).

    Args:
        y_true: Ground truth test values
        y_pred: Predicted values
        y_train: Training values for computing scaling factor
        seasonal_period: Seasonal period for naive forecast

    Returns:
        MASE value
    """
    if y_train is None:
        # If no training data, use MAE
        return mae(y_true, y_pred)

    # Compute MAE of predictions
    mae_pred = np.mean(np.abs(y_true - y_pred))

    # Compute MAE of naive seasonal forecast on training data
    n = len(y_train)
    if n < seasonal_period + 1:
        seasonal_period = 1

    mae_naive = np.mean(
        np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    )

    # Avoid division by zero
    if mae_naive == 0:
        return 0 if mae_pred == 0 else np.inf

    return mae_pred / mae_naive


def owa(
    smape_val: float,
    mase_val: float,
    smape_benchmark: float,
    mase_benchmark: float
) -> float:
    """
    Overall Weighted Average (M4 ranking metric).

    Args:
        smape_val: sMAPE of the model
        mase_val: MASE of the model
        smape_benchmark: sMAPE of benchmark (e.g., Naive2)
        mase_benchmark: MASE of benchmark

    Returns:
        OWA value (lower is better, 1.0 = benchmark performance)
    """
    smape_ratio = smape_val / smape_benchmark
    mase_ratio = mase_val / mase_benchmark

    # Weighted average (50-50 split)
    return 0.5 * smape_ratio + 0.5 * mase_ratio


# ============================================================================
# Batch Calculation Functions
# ============================================================================

def calculate_standard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all standard forecasting metrics.

    Args:
        y_true: Ground truth values (batch, seq_len, features)
        y_pred: Predicted values (batch, seq_len, features)

    Returns:
        Dictionary of metric values
    """
    return {
        'MSE': mse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }


def calculate_m4_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonal_period: int = 1
) -> Dict[str, float]:
    """
    Calculate M4 Competition metrics.

    Args:
        y_true: Ground truth test values
        y_pred: Predicted values
        y_train: Training values (for MASE)
        seasonal_period: Seasonal period

    Returns:
        Dictionary of M4 metric values
    """
    metrics = {
        'sMAPE': smape(y_true, y_pred),
        'MASE': mase(y_true, y_pred, y_train, seasonal_period),
        'MAE': mae(y_true, y_pred),
        'MSE': mse(y_true, y_pred)
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metric values
        title: Title for the metrics table
    """
    print()
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)

    for name, value in metrics.items():
        print(f"{name:>15s}: {value:>10.4f}")

    print("=" * 60)
