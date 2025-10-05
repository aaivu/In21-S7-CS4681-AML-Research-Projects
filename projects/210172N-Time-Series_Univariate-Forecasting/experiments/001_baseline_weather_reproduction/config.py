"""
Configuration for Experiment 001: Baseline Weather Reproduction
Based on PatchTST paper weather.sh script
"""

# Data paths
DATA_ROOT = "../../data/secondary/weather"
DATA_FILE = "weather.csv"

# Data settings
SEQ_LEN = 336  # Look-back window
PRED_LEN = 96  # Forecast horizon (change to 192, 336, 720 for other experiments)
LABEL_LEN = 48  # Decoder input overlap (not used in PatchTST but kept for compatibility)
TARGET = 'OT'  # Target column
FEATURES = 'M'  # S: univariate, M: multivariate (weather.sh uses M with 21 channels)

# Model settings
ENC_IN = 21  # Number of input channels (weather features)
D_MODEL = 128
N_HEADS = 16
E_LAYERS = 3
D_FF = 256
PATCH_LEN = 16
STRIDE = 8
DROPOUT = 0.2
FC_DROPOUT = 0.2
HEAD_DROPOUT = 0.0
INDIVIDUAL = False  # Channel-independent (False means shared weights across channels)
REVIN = True  # Reversible Instance Normalization
AFFINE = True  # Learnable affine in RevIN
SUBTRACT_LAST = False  # Use mean instead of last value in RevIN

# Training settings
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 100
PATIENCE = 20
DEVICE = 'cuda'  # Will check torch.cuda.is_available() in train.py

# Experiment settings
SEED = 2021
CHECKPOINT_DIR = "./results/checkpoints"
LOG_DIR = "./results/logs"
