# config.py
# Centralized config defaults. Override via CLI arguments in scripts.

MODEL_NAME = "microsoft/wavlm-large"  # pretrained backbone
OUTPUT_DIR = "./outputs"
RESULTS_DIR = "../results"
DATA_MANIFEST_DIR = "../data/manifest"  # optional manifest directory
KENLM_PATH_DEFAULT = "../models/4gram.arpa"  # replace with your LM path
SAMPLE_RATE = 16000