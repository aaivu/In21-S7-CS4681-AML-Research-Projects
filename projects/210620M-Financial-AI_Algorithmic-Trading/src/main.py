# main.py
# from data_acquisition import get_stock_data
from model import train_model
# train, test = get_stock_data(ticker='AAPL')
import pandas as pd

# 1. Download and preprocess
# train, test = get_stock_data(ticker='AAPL')
train = pd.read_csv('processed_data/train.csv')
test = pd.read_csv('processed_data/test.csv')

# 2. Train PPO model with default or tuned hyperparameters
model = train_model(timesteps=100_000)
