# tune.py
import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv

def objective(trial):
    df = pd.read_csv('processed_data/train.csv')
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # Hyperparameters to tune
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)

    model = PPO('MlpPolicy', env,
                n_steps=n_steps,
                gamma=gamma,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                clip_range=clip_range,
                verbose=0)
    model.learn(total_timesteps=50_000)

    # Evaluate reward on train set
    obs = env.reset()
    total_reward = 0
    for _ in range(len(df)):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_params)
