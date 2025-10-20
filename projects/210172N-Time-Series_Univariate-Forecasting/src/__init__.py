"""
PatchTST for Real-Time M4 Forecasting
======================================

Production-ready implementation of PatchTST adapted for the M4 Competition
with real-time optimization using ONNX and quantization.

Author: Galappaththi A. S.
Project: CS4681 - Advanced Machine Learning Research
"""

__version__ = "1.0.0"
__author__ = "Galappaththi A. S."

from src.models.patchtst import PatchTSTModel
from src.training.trainer import Trainer
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator

__all__ = [
    'PatchTSTModel',
    'Trainer',
    'Predictor',
    'Evaluator',
]
