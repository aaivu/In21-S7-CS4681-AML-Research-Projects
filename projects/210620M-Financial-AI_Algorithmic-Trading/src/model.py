# model.py
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv

def train_model(train_csv='processed_data/train.csv', save_path='ppo_model', timesteps=100_000, **kwargs):
    df = pd.read_csv(train_csv)
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=1, **kwargs)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model
