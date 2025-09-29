# data_acquisition.py
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import os

def get_stock_data(ticker='AAPL', start='2010-01-01', end='2023-01-01', save_path='processed_data'):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)

    data = data.reset_index()
    data['day'] = np.arange(len(data))
    data.rename(columns={'Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'}, inplace=True)

    # Ensure 1D series
    close_series = data['close'].astype(float).squeeze()

    # Indicators
    data['macd'] = ta.trend.MACD(close_series).macd()
    data['rsi_30'] = ta.momentum.RSIIndicator(close_series, window=30).rsi()
    bb = ta.volatility.BollingerBands(close_series)
    data['boll_ub'] = bb.bollinger_hband()
    data['boll_lb'] = bb.bollinger_lband()
    data['close_30_sma'] = close_series.rolling(30).mean()
    data['close_60_sma'] = close_series.rolling(60).mean()
    
    data.fillna(method='bfill', inplace=True)

    # Train/test split
    train_ratio = 0.8
    train_cutoff = int(len(data) * train_ratio)
    train = data.iloc[:train_cutoff].copy()
    test = data.iloc[train_cutoff:].copy()

    os.makedirs(save_path, exist_ok=True)
    train.to_csv(f'{save_path}/train.csv', index=False)
    test.to_csv(f'{save_path}/test.csv', index=False)

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test
