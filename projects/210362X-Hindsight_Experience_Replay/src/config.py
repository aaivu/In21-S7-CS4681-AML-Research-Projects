# config.py
import os

PROJECT_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Environment
ENV_NAME = "LunarLander-v3"  # fallback handled in train.py if missing
SEED = 42
ENV_NAME = "LunarLanderContinuous-v3"

# Training
MAX_EPISODES = 150
MAX_STEPS = 500
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005  # for soft update
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
REWARD_MODEL_LR = 1e-3

# Replay / PER / HER
BUFFER_SIZE = int(1e6)
PRIORITIZED_ALPHA = 0.6
PRIORITIZED_BETA0 = 0.4
PRIORITIZED_EPS = 1e-6
HER_K = 4  # number of HER substitutions per episode

# Saving
SAVE_DIR = os.path.join(PROJECT_DIR, "checkpoints-v2")
os.makedirs(SAVE_DIR, exist_ok=True)
ACTOR_PATH = os.path.join(SAVE_DIR, "actor.pt")
CRITIC_PATH = os.path.join(SAVE_DIR, "critic.pt")
REWARD_MODEL_PATH = os.path.join(SAVE_DIR, "reward_model.pt")
BUFFER_PATH = os.path.join(SAVE_DIR, "replay_buffer.pkl")

# Plotting frequency
PLOT_FREQ = 30  # plot every N episodes
LOG_INTERVAL = 5

# Misc
USE_LEARNED_REWARD = True   # set True to use learned reward for training the policy
HER_ON = True
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


# TD3 / advanced
POLICY_DELAY = 2
TARGET_POLICY_NOISE = 0.2
TARGET_POLICY_NOISE_CLIP = 0.5
TAU = 0.005

# RND
USE_RND = True
RND_PRED_LR = 1e-4
RND_INT_SCALE = 1.0

# Reward ensemble
REWARD_ENSEMBLE_SIZE = 3
REWARD_ENSEMBLE_LR = 3e-4
USE_REWARD_ENSEMBLE = True
UNCERTAINTY_BONUS_SCALE = 0.5

# mixing weights
W_ENV = 0.4
W_LEARNED = 0.4
W_INTRINSIC = 0.15
W_UNCERT = 0.05

# gradient clipping
MAX_GRAD_NORM = 0.5



# config.py - Complete configuration for TD3 training with advanced features

import torch

# Environment
ENV_NAME = "LunarLanderContinuous-v3"
SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
MAX_EPISODES = 600
MAX_STEPS = 1000  # Max steps per episode
BATCH_SIZE = 256
GAMMA = 0.97  # Discount factor

# Replay buffer
BUFFER_SIZE = 1_000_000

# Prioritized Experience Replay (PER)
PRIORITIZED_ALPHA = 0.6  # How much prioritization to use (0 = uniform, 1 = full prioritization)
PRIORITIZED_BETA0 = 0.4  # Initial importance sampling weight (increases to 1.0 over training)
PRIORITIZED_EPS = 1e-6   # Small constant to prevent zero priority

# Network learning rates
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
REWARD_ENSEMBLE_LR = 3e-4
RND_PRED_LR = 1e-4

# TD3 specific parameters
TAU = 0.005  # Soft update parameter for target networks
POLICY_DELAY = 2  # How often to update actor relative to critic (TD3 feature)
TARGET_POLICY_NOISE = 0.2  # Noise added to target policy for smoothing
TARGET_POLICY_NOISE_CLIP = 0.5  # Clip range for target policy noise

# Exploration
EXPLORATION_NOISE = 0.1  # Std of Gaussian noise added to actions during training for exploration
# This noise is added to the actor's output to encourage exploration
# Set to 0 to disable exploration noise (rely only on policy learning)

# Gradient clipping
MAX_GRAD_NORM = 1.0  # Max gradient norm for clipping (None to disable)

# Hindsight Experience Replay (HER)
HER_ON = True
HER_K = 4  # Number of HER transitions to generate per episode

# Reward Ensemble
USE_REWARD_ENSEMBLE = True
REWARD_ENSEMBLE_SIZE = 5  # Number of reward models in ensemble
REWARD_ENSEMBLE_WEIGHT = 0.5  # Weight for mixing learned reward with env reward

# Random Network Distillation (RND) for intrinsic motivation
USE_RND = True
RND_PRED_LR = 1e-4  # Learning rate for RND predictor
RND_INTRINSIC_WEIGHT = 0.01  # Weight for intrinsic reward in total reward

# Learned reward (backward compatibility)
USE_LEARNED_REWARD = USE_REWARD_ENSEMBLE  # Use ensemble mean as learned reward

# Logging and saving
LOG_INTERVAL = 10  # Print stats every N episodes
PLOT_FREQ = 50     # Generate plots every N episodes
SAVE_FREQ = 100    # Save checkpoint every N episodes

# Directories
PLOTS_DIR = "plots-v2"
CHECKPOINTS_DIR = "checkpoints-v2"

# Create directories if they don't exist
import os
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Model architecture (optional, can be used in models.py)
ACTOR_HIDDEN = [400, 300]
CRITIC_HIDDEN = [400, 300]
REWARD_MODEL_HIDDEN = [256, 256]
RND_HIDDEN = [128, 128]

# Print configuration on import
def print_config():
    print("="*60)
    print("Configuration")
    print("="*60)
    print(f"Environment: {ENV_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Max Episodes: {MAX_EPISODES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Buffer Size: {BUFFER_SIZE}")
    print(f"Gamma: {GAMMA}")
    print(f"Actor LR: {ACTOR_LR}")
    print(f"Critic LR: {CRITIC_LR}")
    print(f"TAU: {TAU}")
    print(f"Policy Delay: {POLICY_DELAY}")
    print(f"Exploration Noise: {EXPLORATION_NOISE}")
    print(f"HER: {HER_ON}")
    print(f"Reward Ensemble: {USE_REWARD_ENSEMBLE}")
    print(f"RND: {USE_RND}")
    print("="*60)

# Uncomment to print config on import
# print_config()