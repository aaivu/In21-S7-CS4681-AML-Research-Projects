# tune.py
import optuna
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv
import os
import torch

np.random.seed(42)
torch.manual_seed(42)


def get_model_class(algo_name):
    algo_name = algo_name.upper()
    if algo_name == 'PPO':
        return PPO
    elif algo_name == 'A2C':
        return A2C
    elif algo_name == 'DDPG':
        return DDPG
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")


def objective(trial, algo_name):
    """
    Optuna objective for single models (PPO, A2C, DDPG).
    """
    df = pd.read_csv('processed_data/train.csv')
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    Model = get_model_class(algo_name)

    # Suggest hyperparameters
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    if algo_name in ['PPO', 'A2C']:
        n_steps = trial.suggest_categorical('n_steps', [128, 256, 512])
        ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
        model = Model(
            'MlpPolicy', env,
            n_steps=n_steps,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            verbose=0
        )
    elif algo_name == 'DDPG':
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        tau = trial.suggest_float('tau', 0.001, 0.05)
        buffer_size = trial.suggest_categorical('buffer_size', [10000, 50000, 100000])

        model = Model(
            'MlpPolicy',
            env,
            gamma=gamma,
            learning_rate=learning_rate,
            tau=tau,
            buffer_size=buffer_size,
            action_noise=action_noise,
            verbose=0
        )

    model.learn(total_timesteps=50000)

    # Evaluate in a fresh environment
    eval_env = DummyVecEnv([lambda: StockTradingEnv(df)])
    obs = eval_env.reset()
    total_reward = 0.0
    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        if done:
            break

    return float(total_reward)


def objective_ensemble(trial):
    """
    Optuna objective for tuning PPO + A2C ensemble.
    Uses separate environments for training and evaluation for comparability.
    """
    df = pd.read_csv('processed_data/train.csv')

    gamma = trial.suggest_float('gamma', 0.9, 0.9999)

    # PPO hyperparams
    ppo_lr = trial.suggest_float('ppo_learning_rate', 1e-5, 1e-3, log=True)
    ppo_n_steps = trial.suggest_categorical('ppo_n_steps', [128, 256, 512])
    ppo_ent_coef = trial.suggest_float('ppo_ent_coef', 0.0, 0.1)

    # A2C hyperparams
    a2c_lr = trial.suggest_float('a2c_learning_rate', 1e-5, 1e-3, log=True)
    a2c_n_steps = trial.suggest_categorical('a2c_n_steps', [128, 256, 512])
    a2c_ent_coef = trial.suggest_float('a2c_ent_coef', 0.0, 0.1)

    # Ensemble weights
    w_ppo = trial.suggest_float('ppo_weight', 0.0, 1.0)
    w_a2c = 1.0 - w_ppo

    ppo_env = DummyVecEnv([lambda: StockTradingEnv(df)])
    ppo_model = PPO(
        "MlpPolicy",
        ppo_env,
        learning_rate=ppo_lr,
        n_steps=ppo_n_steps,
        ent_coef=ppo_ent_coef,
        gamma=gamma,
        verbose=0
    )
    ppo_model.learn(total_timesteps=50000)

    a2c_env = DummyVecEnv([lambda: StockTradingEnv(df)])
    a2c_model = A2C(
        "MlpPolicy",
        a2c_env,
        learning_rate=a2c_lr,
        n_steps=a2c_n_steps,
        ent_coef=a2c_ent_coef,
        gamma=gamma,
        verbose=0
    )
    a2c_model.learn(total_timesteps=50000)

    eval_env = DummyVecEnv([lambda: StockTradingEnv(df)])
    obs = eval_env.reset()
    total_reward = 0.0
    done = False
    while True:
        action_ppo, _ = ppo_model.predict(obs, deterministic=True)
        action_a2c, _ = a2c_model.predict(obs, deterministic=True)

        combined_action = w_ppo * action_ppo + w_a2c * action_a2c

        obs, reward, done, info = eval_env.step(combined_action)
        total_reward += reward
        if done:
            break

    return float(total_reward)


def tune_model(algo_name='PPO', n_trials=10):
    algo_name = algo_name.upper()
    os.makedirs('tuning_results', exist_ok=True)

    print(f"Starting tuning for {algo_name}...")

    study = optuna.create_study(direction='maximize')

    if algo_name == 'ENSEMBLE':
        study.optimize(objective_ensemble, n_trials=n_trials)
    else:
        study.optimize(lambda trial: objective(trial, algo_name), n_trials=n_trials)

    # Save results
    results = []
    for t in study.trials:
        result = t.params.copy()
        result['reward'] = t.value
        results.append(result)

    df_results = pd.DataFrame(results)
    result_path = f'tuning_results/{algo_name.lower()}_tuning_results.csv'
    df_results.to_csv(result_path, index=False)

    print(f"\nTuning complete for {algo_name}")
    print(f"Best reward: {study.best_value:.2f}")
    print("Best hyperparameters:", study.best_params)
    print(f"Results saved to: {result_path}")

    return study


if __name__ == '__main__':
    algorithms = ['PPO', 'A2C', 'ENSEMBLE']

    for algo in algorithms: 
        tune_model(algo_name=algo, n_trials=20)
