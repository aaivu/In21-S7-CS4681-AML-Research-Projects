import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv

def train_model(algo='PPO', train_csv='processed_data/train.csv', save_path=None, timesteps=100000, **kwargs):
    """
    Train a DRL model (PPO, A2C) on the StockTradingEnv.
    """
    df = pd.read_csv(train_csv)
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # Select algorithm
    if algo.upper() == 'PPO':
        model_cls = PPO
    elif algo.upper() == 'A2C':
        model_cls = A2C
    elif algo.upper() == 'DDPG':
        model_cls = DDPG
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    save_path = save_path or f'{algo.lower()}_model'
    model = model_cls('MlpPolicy', env, verbose=1, **kwargs)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"{algo} model saved to {save_path}")

    return model
