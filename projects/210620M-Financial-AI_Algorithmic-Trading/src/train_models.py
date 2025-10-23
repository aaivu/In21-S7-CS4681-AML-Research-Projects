import argparse
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockTradingEnv

def train_model(algo='PPO', train_csv='processed_data/train.csv', test_csv='processed_data/test.csv', save_path=None, timesteps=100000, **kwargs):
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
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    save_path = save_path or f'{algo.lower()}_model'
    model = model_cls('MlpPolicy', env, verbose=1, **kwargs)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"{algo} model saved to {save_path}")

    # --- Evaluation ---
    print("Evaluating on test data...")
    test_df = pd.read_csv(test_csv)
    eval_env = DummyVecEnv([lambda: StockTradingEnv(test_df)])

    obs = eval_env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward
        if done:
            break

    avg_reward = float(total_reward)
    print(f"🏁 Evaluation complete. Total Reward on test set: {avg_reward:.2f}")

    return model, avg_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DRL model (PPO or A2C) for stock trading.')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'A2C'],
                        help='Algorithm to use: PPO or A2C (default: PPO)')
    parser.add_argument('--train_csv', type=str, default='processed_data/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Where to save the trained model')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Number of timesteps to train (default: 100000)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for the optimizer (default: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor gamma (default: 0.99)')

    args = parser.parse_args()

    train_model(
        algo=args.algo,
        train_csv=args.train_csv,
        save_path=args.save_path,
        timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma
    )
