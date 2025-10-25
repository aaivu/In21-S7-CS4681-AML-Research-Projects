import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "base_models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_DATA_PATH = DATA_DIR / "KDDTest+.txt"
TEST_DATA_21_PATH = DATA_DIR / "KDDTest-21.txt"  # Additional test set for generalizability

# Available test datasets
TEST_DATASETS = {
    'KDDTest+': TEST_DATA_PATH,
    'KDDTest-21': TEST_DATA_21_PATH
}

# Model paths
MODEL_PATHS = {
    'DoS': MODELS_DIR / "DoS.h5",
    'Probe': MODELS_DIR / "Probe.h5",
    'R2L': MODELS_DIR / "R2L.h5",
    'U2R': MODELS_DIR / "U2R.h5",
    'probe_rf': MODELS_DIR / "rf_probe_model.pkl",
    'probe_gb': MODELS_DIR / "gb_probe_model.pkl",
    'probe_lr': MODELS_DIR / "lr_probe_model.pkl"
}

# Attack type mappings
ATTACK_MAPPING = {
    'normal': 0,
    'neptune': 1, 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3, 'spy': 3,
    'warezclient': 3, 'warezmaster': 3, 'sendmail': 3, 'named': 3, 'snmpgetattack': 3,
    'snmpguess': 3, 'xlock': 3, 'xsnoop': 3, 'httptunnel': 3,
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'ps': 4,
    'sqlattack': 4, 'xterm': 4
}

ATTACK_CATEGORIES = {
    0: 'Normal',
    1: 'DoS',
    2: 'Probe',
    3: 'R2L',
    4: 'U2R'
}

# Feature names
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
]

# Model hyperparameters
MODEL_CONFIG = {
    'random_seed': 42,
    'test_size': 0.2,
    'n_features_rfe': 123,  # Use full feature set to match trained models
    'smote_sampling_strategy': 'auto',
    'smote_k_neighbors': 5,
    'focal_loss_gamma': 2.0,
    'cnn_lstm': {
        'conv_filters': [64, 64, 128, 128],
        'conv_kernel_size': 3,
        'lstm_units': 100,
        'dropout_rate': 0.5,
        'l2_reg': 0.01,
        'initial_lr': 0.001,
        'batch_size': 64,
        'epochs': 50,
        'early_stopping_patience': 15
    }
}

# Ensemble parameters
ENSEMBLE_CONFIG = {
    'attack_thresholds': {
        'DoS': 0.5,
        'Probe': 0.47,
        'R2L': 0.004,
        'U2R': 0.942
    },
    'probe_ensemble_weights': {
        'cnn_lstm': 0.4,
        'rf': 0.3,
        'gb': 0.2,
        'lr': 0.1
    },
    'confidence_threshold': 0.3,
    'voting_weights': {'DoS': 1.0, 'Probe': 1.0, 'R2L': 1.0, 'U2R': 1.0}
}

# Create directories
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
