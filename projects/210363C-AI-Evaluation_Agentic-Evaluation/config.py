import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# API & Model Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

# Data Config
DATA_PATH = os.getenv("DATA_PATH")

# DATA_PATH=data/math_easy_int_120.jsonl
# OUT_PATH=results/results.csv

# Configuration for the application
SEED = 42
TEMPERATURE = 0.7
MAX_TOKENS = 1024
CODE_EXEC_ENABLED = True

SEED_VALUES = [42, 123, 999]
TEMPERATURE_VALUES = [0.5, 0.7, 0.9]

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env file")
