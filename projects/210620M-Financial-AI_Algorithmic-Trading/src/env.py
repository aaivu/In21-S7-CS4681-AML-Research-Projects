# env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        self.initial_balance = initial_balance

        # Observation: [balance, shares_held, close, macd, rsi_30, boll_ub, boll_lb, close_30_sma, close_60_sma]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # Action: continuous -1 to 1 (sell/buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}  # Gymnasium expects obs, info tuple

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            self.balance,
            self.shares_held,
            row['close'],
            row['macd'],
            row['rsi_30'],
            row['boll_ub'],
            row['boll_lb'],
            row['close_30_sma'],
            row['close_60_sma']
        ], dtype=np.float32)
        return obs

    def step(self, action):
        action = action[0]
        row = self.df.iloc[self.current_step]
        price = row['close']

        # Buy/sell logic
        if action > 0:
            buy_amount = self.balance * action
            shares_bought = buy_amount / price
            self.shares_held += shares_bought
            self.balance -= buy_amount
        elif action < 0:
            sell_amount = self.shares_held * (-action)
            self.balance -= min(self.shares_held, -action) * price
            self.shares_held -= min(self.shares_held, -action)

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # Reward = change in portfolio
        portfolio_value = self.balance + self.shares_held * price
        reward = portfolio_value - self.initial_balance

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Shares: {self.shares_held}')
