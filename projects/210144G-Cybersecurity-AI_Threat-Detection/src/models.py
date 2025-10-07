"""
Model architecture definitions for Network Intrusion Detection System
This file contains model architectures used in the individual training notebooks
"""

from typing import Dict, Any

# CNN-LSTM Architecture Configuration
CNNLSTM_ARCHITECTURE = {
    'description': 'CNN-LSTM model with focal loss for intrusion detection',
    'layers': [
        {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
        {'type': 'Conv1D', 'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
        {'type': 'MaxPooling1D', 'pool_size': 2},
        {'type': 'Conv1D', 'filters': 128, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
        {'type': 'Conv1D', 'filters': 128, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'},
        {'type': 'MaxPooling1D', 'pool_size': 2},
        {'type': 'BatchNormalization'},
        {'type': 'LSTM', 'units': 100, 'dropout': 0.1},
        {'type': 'Dropout', 'rate': 0.5},
        {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
    ],
    'loss': 'focal_loss',
    'metrics': ['accuracy', 'precision', 'recall', 'auc'],
    'optimizer': 'Adam'
}

# Traditional ML Models Configuration
TRADITIONAL_ML_MODELS = {
    'RandomForest': {
        'description': 'Random Forest classifier for Probe attack detection',
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    },
    'GradientBoosting': {
        'description': 'Gradient Boosting classifier for Probe attack detection',
        'hyperparameters': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
    },
    'LogisticRegression': {
        'description': 'Logistic Regression classifier for Probe attack detection',
        'hyperparameters': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear',
            'random_state': 42
        }
    }
}

def get_model_info() -> Dict[str, Any]:
    """
    Get information about model architectures used in training
    
    Returns:
        Dictionary containing model architecture information
    """
    return {
        'deep_learning': CNNLSTM_ARCHITECTURE,
        'traditional_ml': TRADITIONAL_ML_MODELS,
        'training_note': 'Models are trained in individual notebooks with specific configurations per attack type'
    }
