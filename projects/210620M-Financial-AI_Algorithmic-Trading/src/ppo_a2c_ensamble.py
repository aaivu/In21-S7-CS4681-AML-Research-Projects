# ensemble.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv


def train_models(df, total_timesteps=50_000, verbose=0):
    """
    Train both PPO and A2C models on the same environment.
    Returns trained models and environment.
    """
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # PPO model
    ppo_model = PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=256,
        gamma=0.99,
        ent_coef=0.01,
        verbose=verbose
    )
    ppo_model.learn(total_timesteps=total_timesteps)

    # A2C model
    a2c_model = A2C(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=256,
        gamma=0.99,
        ent_coef=0.01,
        verbose=verbose
    )
    a2c_model.learn(total_timesteps=total_timesteps)

    return ppo_model, a2c_model, env


def ensemble_predict(obs, ppo_model, a2c_model, w_ppo=0.5, w_a2c=0.5):
    """
    Predict ensemble action as a weighted average of PPO and A2C predictions.
    """
    action_ppo, _ = ppo_model.predict(obs, deterministic=True)
    action_a2c, _ = a2c_model.predict(obs, deterministic=True)
    action = w_ppo * action_ppo + w_a2c * action_a2c
    return action


def evaluate_ensemble(df, total_timesteps=50000, w_ppo=0.5, w_a2c=0.5):
    """
    Train PPO + A2C, then evaluate ensemble performance.
    """
    ppo_model, a2c_model, env = train_models(df, total_timesteps)

    obs = env.reset()
    total_reward = 0
    done = False

    while True:
        action = ensemble_predict(obs, ppo_model, a2c_model, w_ppo, w_a2c)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    return float(total_reward)


if __name__ == '__main__':
    df = pd.read_csv('processed_data/train.csv')
    reward = evaluate_ensemble(df)
    print(f"Ensemble (PPO + A2C) Reward: {reward:.2f}")
