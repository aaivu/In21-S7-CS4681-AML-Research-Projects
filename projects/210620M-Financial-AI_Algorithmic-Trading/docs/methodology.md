# Methodology: Financial AI:Algorithmic Trading

**Student:** 210620M
**Research Area:** Financial AI:Algorithmic Trading
**Date:** 2025-09-01

## 1. Overview
This project implements an end-to-end reinforcement learning pipeline for single-instrument algorithmic trading. The implementation in `src/` contains routines for:

- downloading historical OHLCV data and computing technical indicators (`src/data_acquisition.py`),
- simulating a single-instrument trading environment compatible with Gymnasium (`src/env.py`),
- training and saving DRL agents using Stable-Baselines3 (`src/model.py`),
- constructing a simple PPO+A2C ensemble (`src/ppo_a2c_ensamble.py`), and
- hyperparameter tuning using Optuna (`src/tune.py`).

The methodology focuses on reproducibility and modularity: data generation and feature engineering are separated from environment logic and model training so that experiments can be repeated and components swapped independently.

## 2. Research Design
This research adopts an empirical, experiment-driven approach. Core elements are:

- Hypothesis: A reinforcement-learning agent trained on technical indicators for a single equity can learn trading behaviors that yield positive cumulative returns compared to simple baselines (e.g., buy-and-hold).
- Controlled experiments: use chronological train/test splits to avoid look-ahead bias. Train on the first 80% of the time series and test on the final 20%.
- Ablations and comparisons: evaluate single algorithms (PPO, A2C, DDPG), and an ensemble of PPO+A2C.
- Hyperparameter search: use Optuna to explore learning rate, gamma, n_steps, ent_coef, and ensemble weights.

The design emphasises reproducible configuration: model selection, timesteps, and data paths are exposed through function arguments or top-level scripts to make experiment runs and comparisons straightforward.

## 3. Data Collection

### 3.1 Data Sources
[List your data sources]
Data are sourced from Yahoo Finance via the `yfinance` Python library. The default code example downloads a single ticker's historical OHLCV data (Open, High, Low, Close, Volume) for a specified date range using `get_stock_data()` in `src/data_acquisition.py`.

For reproducibility and research comparisons, the code saves processed CSVs and uses them as canonical inputs for training and tuning.

### 3.2 Data Description
[Describe your datasets]
Each processed dataset is a chronological table containing per-period (daily by default) OHLCV and engineered indicator columns. Typical columns produced by `data_acquisition.py` are:

- date / index (reset by the function),
- day (integer time index),
- open, high, low, close, volume,
- macd,
- rsi_30,
- boll_ub (Bollinger upper band),
- boll_lb (Bollinger lower band),
- close_30_sma,
- close_60_sma.

The training and testing splits are simple chronological partitions (first 80% rows -> training, remaining 20% -> testing). Files are written into `processed_data/` by default as `train.csv` and `test.csv`.

### 3.3 Data Preprocessing
[Explain preprocessing steps]
Preprocessing steps implemented in `src/data_acquisition.py`:

1. Download OHLCV with `yfinance` and reset index to make date a column if present.
2. Add a `day` integer index and rename columns to lowercase for consistency.
3. Compute indicators using `ta` and pandas rolling operations: MACD, RSI (30), Bollinger Bands, 30-day and 60-day SMAs.
4. Fill missing indicator values with backward fill (`DataFrame.fillna(method='bfill')`) to avoid NaNs during training.
5. Split chronologically into train/test using an 80/20 ratio and persist as CSVs.

Notes and caveats:
- Short date ranges can lead to NaNs for long-window indicators; ensure the date range exceeds the largest indicator window.
- The current pipeline targets single-instrument experiments; multi-symbol processing would require changes to data shapes and environment logic.

## 4. Model Architecture

[Describe your proposed model/algorithm]
The project uses off-the-shelf deep reinforcement learning algorithms from Stable-Baselines3. The architecture choices are:

- Algorithms: PPO, A2C and DDPG are supported via `src/model.py`.
- Policy: default MLP policy (`MlpPolicy`) provided by Stable-Baselines3. Network architecture and policy kwargs can be passed through `train_model`.

Model training flow (`train_model`):

1. Load the training CSV into a pandas DataFrame.
2. Wrap `StockTradingEnv(df)` in a `DummyVecEnv` for compatibility with Stable-Baselines3.
3. Instantiate the requested algorithm class with passed hyperparameters (learning rate, gamma, n_steps, ent_coef, etc.).
4. Call `model.learn(total_timesteps=...)` to train and then `model.save(save_path)` to persist the agent.

Rationale: using standard, well-tested RL implementations allows focusing experiments on data, environment design and hyperparameters rather than bespoke RL algorithm engineering.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
[List evaluation metrics you'll use]
Primary metric used in the code and tuning objective:

- Cumulative episode return (sum of per-step rewards) computed over an evaluation episode. This is the metric used as Optuna's objective value by default.

Recommended additional metrics (to implement for rigorous evaluation):

- Sharpe ratio (mean excess return / std of returns, annualized),
- Maximum drawdown (peak-to-trough loss),
- Annualized return / CAGR, 
- Win rate (percentage of profitable episodes or steps),
- Risk-adjusted return measures (Sortino ratio, Calmar ratio).

The current code logs cumulative returns; computing the extra metrics requires storing per-step portfolio values during evaluation and aggregating post-hoc.

### 5.2 Baseline Models
[Describe baseline comparisons]
Baselines to compare against (easy to implement given current code structure):

- Buy-and-hold (long the asset across the test period),
- Momentum/simple rule-based strategies using the same indicators (e.g., buy when close > 30-day SMA, sell otherwise),
- Single-agent baseline comparisons (PPO vs A2C vs DDPG) to measure algorithmic differences.

Implement baselines by computing returns on the `test.csv` data and comparing cumulative returns and the recommended financial metrics.

### 5.3 Hardware/Software Requirements
[List computational requirements]
Minimum software requirements (from code usage):

- Python 3.8+,
- pandas, numpy,
- yfinance,
- ta (technical analysis library),
- gymnasium,
- stable-baselines3,
- torch (PyTorch),
- optuna (for tuning).

Hardware:

- CPU-only is sufficient for small experiments and development. For larger timesteps and faster training, a CUDA-enabled GPU with compatible PyTorch and drivers is recommended.

Practical notes:
- Install package versions compatible with each other (PyTorch and stable-baselines3 are sensitive to mismatched versions). Use the repository `requirements.txt` as a starting point.

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data acquisition & preprocessing: implement `get_stock_data()`, compute indicators, produce `processed_data/train.csv` and `processed_data/test.csv` | 1–2 weeks | Processed CSVs, data validation scripts |
| Phase 2 | Environment & model prototypes: implement `StockTradingEnv`, create training helper `train_model`, run small-scale experiments | 2–3 weeks | Working environment, trained toy models |
| Phase 3 | Tuning & ensembles: run Optuna experiments for single models and PPO+A2C ensemble; refine hyperparameters | 2–4 weeks | Tuning results CSVs, best hyperparameter sets |
| Phase 4 | Robust evaluation & reporting: implement evaluation scripts, compute financial metrics, run final experiments and collect results | 1–2 weeks | Evaluation reports, plots, final trained models |

Timing depends on compute capacity and the size of timesteps used for training/tuning. The plan is iterative: early phases use smaller timesteps and fewer trials to narrow promising configurations before running full-scale experiments.

## 7. Risk Analysis

[Identify potential risks and mitigation strategies]
Key risks and mitigations observed during implementation:

- Data leakage / look-ahead bias: Using chronological splits and avoiding indicator calculations that use future data mitigates look-ahead. Always verify that feature calculations and splits are strictly causal.
- Overfitting to train split: Use a held-out test split (the code already separates test data), and prefer multi-episode evaluation or cross-validation (rolling windows) when tuning hyperparameters.
- Noisy reward signal and unstable training: DRL agents can be unstable with sparse or noisy rewards. Mitigate by shaping rewards carefully, normalizing inputs, running multiple training seeds, and using ensemble or averaging strategies.
- Compute/time constraints: Tuning and training are expensive. Use smaller timesteps during exploration, checkpointing for long runs, and run parallel studies if resources allow.
- Library / API changes: Stable-Baselines3, Gymnasium and other libs evolve; pin versions in `requirements.txt` and run small smoke tests after upgrades.

Operational mitigations:
- Log all experiment configuration and random seeds.
- Save model checkpoints and intermediate metrics.
- Automate dataset generation with reproducible scripts.

## 8. Expected Outcomes

[Describe expected results and contributions]
From the implemented pipeline we expect the following outcomes:

- A reproducible data pipeline that generates cleaned, indicator-rich CSVs suitable for RL training.
- Working Gymnasium-compatible `StockTradingEnv` that can be used with Stable-Baselines3 algorithms.
- Trained agents (PPO/A2C/DDPG) and a baseline ensemble (PPO+A2C) with saved models for reproducible evaluation.
- Tuning results (Optuna trials) that identify promising hyperparameter regions for each algorithm and the ensemble weight.

Academic and practical contributions:
- A modular experimental framework to compare DRL algorithms for single-instrument trading, and
- A demonstration of a simple ensemble approach that can be extended for robustness.

Success criteria
- Agents that consistently outperform simple baselines (e.g., buy-and-hold) on the test split under one or more metrics (cumulative return, Sharpe) would indicate success; otherwise use the experiments to identify failure modes and refine environment design.

---

**Note:** Update this document as your methodology evolves during implementation.