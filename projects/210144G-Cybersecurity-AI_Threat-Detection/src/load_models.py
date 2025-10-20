"""
Model loading utilities for pre-trained Network Intrusion Detection models
"""

import sys
import tensorflow as tf
from tensorflow import keras
import joblib
from pathlib import Path
from typing import Dict, Any

# Add current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_PATHS


def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss function for handling class imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(tf.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_t * tf.pow(focal_weight, gamma)
        cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(cls_loss)
    return focal_loss_fixed


def load_pretrained_models() -> Dict[str, Any]:
    """
    Load all pre-trained models from the Models directory
    
    Returns:
        Dictionary of loaded models
    """
    models = {}
    
    # Load deep learning models
    attack_types = ['DoS', 'Probe', 'R2L', 'U2R']
    
    for attack_type in attack_types:
        model_path = MODEL_PATHS[attack_type]
        if model_path.exists():
            try:
                # Try different loading approaches for robustness
                try:
                    # Try loading with custom objects for focal loss
                    custom_objects = {'focal_loss_fixed': focal_loss()}
                    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
                    models[f'{attack_type}_dl'] = model
                    print(f"Loaded {attack_type} deep learning model")
                except Exception as e1:
                    try:
                        # Try loading with compile=False
                        model = tf.keras.models.load_model(str(model_path), compile=False)
                        models[f'{attack_type}_dl'] = model
                        print(f"Loaded {attack_type} deep learning model (compile=False)")
                    except Exception as e2:
                        try:
                            # Last resort: try loading with all custom objects disabled
                            model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects={})
                            models[f'{attack_type}_dl'] = model
                            print(f"Loaded {attack_type} deep learning model (minimal)")
                        except Exception as e3:
                            print(f"Warning: Could not load {attack_type} model:")
                            print(f"         Primary: {e1}")
                            print(f"         Secondary: {e2}")
                            print(f"         Tertiary: {e3}")
            except Exception as e:
                print(f"Warning: Could not load {attack_type} model: {e}")
        else:
            print(f"Model not found: {model_path}")
    
    # Load traditional ML models for Probe
    ml_models = ['rf', 'gb', 'lr']
    for model_type in ml_models:
        model_key = f'probe_{model_type}'
        model_path = MODEL_PATHS[model_key]
        if model_path.exists():
            try:
                models[model_key] = joblib.load(str(model_path))
                print(f"Loaded Probe {model_type} model")
            except Exception as e:
                print(f"Warning: Could not load Probe {model_type} model: {e}")
    
    return models


def verify_model_compatibility() -> bool:
    """
    Verify that all required models are available and compatible
    
    Returns:
        True if all models are available, False otherwise
    """
    required_models = ['DoS', 'Probe', 'R2L', 'U2R']
    missing_models = []
    
    for attack_type in required_models:
        model_path = MODEL_PATHS[attack_type]
        if not model_path.exists():
            missing_models.append(attack_type)
    
    if missing_models:
        print(f"Missing required models: {missing_models}")
        print("Please train models using the individual notebooks first.")
        return False
    
    return True
