"""
Centralized Configuration File
All hyperparameters and settings in one place for easy tuning
"""

import torch


class Config:
    """Configuration class for CAFE project."""
    
    # ============================================================================
    # Model Configuration
    # ============================================================================
    MODEL_NAME = 'microsoft/deberta-v3-large'
    MAX_LENGTH = 512
    DROPOUT_RATE = 0.1
    
    # ============================================================================
    # Training Configuration
    # ============================================================================
    BATCH_SIZE = 8  # Reduce to 4 or 2 if OOM on GPU
    EPOCHS = 3  # Increase to 4-5 for better performance
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    
    # Cross-validation
    N_FOLDS = 5
    RANDOM_SEED = 42
    
    # ============================================================================
    # Loss Weights Configuration
    # ============================================================================
    LOSS_WEIGHTS = {
        'toxicity': 1.0,  # Primary target
        'severe_toxicity': 0.5,
        'obscene': 0.5,
        'identity_attack': 0.5,
        'insult': 0.5,
        'threat': 0.5,
        'sexual_explicit': 0.5
    }
    
    # ============================================================================
    # Sample Weighting Configuration (Fairness-Aware)
    # ============================================================================
    BASE_WEIGHT = 1.0
    IDENTITY_WEIGHT = 1.5  # Samples mentioning any identity
    BPSN_WEIGHT = 2.0  # Non-toxic + identity (Background Positive, Subgroup Negative)
    BNSP_WEIGHT = 2.0  # Toxic + no identity (Background Negative, Subgroup Positive)
    
    # ============================================================================
    # Ensemble Configuration
    # ============================================================================
    ENSEMBLE_POWER = 3.5  # Power for weighted ensemble
    
    # ============================================================================
    # Data Configuration
    # ============================================================================
    TRAIN_FILE = 'jigsaw_train_preprocessed.parquet'
    TEST_FILE = 'jigsaw_test_preprocessed.parquet'
    TEXT_COLUMN = 'comment_text_cleaned'
    TARGET_COLUMN = 'target'
    
    # Identity columns for bias evaluation
    IDENTITY_COLUMNS = [
        'male', 'female', 'transgender', 'other_gender',
        'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
        'other_sexual_orientation', 'christian', 'jewish', 'muslim',
        'hindu', 'buddhist', 'atheist', 'other_religion',
        'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
        'physical_disability', 'intellectual_or_learning_disability',
        'psychiatric_or_mental_illness', 'other_disability'
    ]
    
    # ============================================================================
    # Output Paths
    # ============================================================================
    MODEL_DIR = 'models'
    RESULTS_DIR = 'results'
    FIGURES_DIR = 'figures'
    
    # ============================================================================
    # Hardware Configuration
    # ============================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2  # For DataLoader
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Mixed precision training (for faster training on newer GPUs)
    USE_AMP = False  # Set to True for automatic mixed precision
    
    # ============================================================================
    # RealToxicityPrompts Configuration
    # ============================================================================
    RTP_SAMPLE_SIZE = None  # None = use all data, or set to integer for subset
    RTP_BATCH_SIZE = 16
    
    # Perspective API (if using)
    PERSPECTIVE_API_KEY = None  # Add your key here
    PERSPECTIVE_RATE_LIMIT = 0.1  # Seconds between requests
    
    # ============================================================================
    # Logging and Checkpointing
    # ============================================================================
    SAVE_BEST_ONLY = True
    VERBOSE = True
    LOG_INTERVAL = 100  # Log every N batches
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("="*80)
        print("Configuration")
        print("="*80)
        for key, value in cls.to_dict().items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("="*80)
    
    @classmethod
    def get_training_config(cls):
        """Get configuration dict for training."""
        return {
            'model_name': cls.MODEL_NAME,
            'max_length': cls.MAX_LENGTH,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'warmup_ratio': cls.WARMUP_RATIO,
            'dropout': cls.DROPOUT_RATE,
            'n_folds': cls.N_FOLDS,
            'loss_weights': cls.LOSS_WEIGHTS,
            'random_seed': cls.RANDOM_SEED
        }


# Create a global config instance
config = Config()


if __name__ == "__main__":
    # Print configuration when run directly
    Config.print_config()