## src — Source modules for the Financial RL project

This folder contains the main Python modules used for data acquisition, environment definition, training models, tuning hyperparameters and evaluating an ensemble of RL agents.

Files
- `data_acquisition.py` — download price history (via `yfinance`), compute indicators (using `ta`), split into train/test and save CSVs (default folder: `processed_data/`).
- `env.py` — `StockTradingEnv` Gymnasium environment used by the agents. Observation and action spaces are documented in the source.
- `model.py` — `train_model(...)` helper to train and save a DRL model (supports PPO, A2C, DDPG via stable-baselines3).
- `ppo_a2c_ensamble.py` — utilities to train PPO and A2C and run a simple weighted ensemble prediction/evaluation.
- `tune.py` — Optuna-based hyperparameter tuning for individual algorithms and the PPO+A2C ensemble; outputs are saved to `tuning_results/`.
- `main.py` — a tiny example entrypoint that loads `processed_data` and calls `train_model` (adjust as needed).

Quick setup

1. Create and activate a virtual environment (recommended).

   PowerShell example:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Generate processed data (if not present)

   Open a Python REPL (or run a small script) and call:

   ```python
   from data_acquisition import get_stock_data
   # Default saves into `processed_data/train.csv` and `processed_data/test.csv`
   get_stock_data('AAPL', start='2015-01-01', end='2020-12-31')
   ```

   Or modify `get_stock_data(..., save_path='my_data_folder')` to write somewhere else.

Usage examples

- Train a model (using the example `main.py`):

  ```powershell
  python src\main.py
  ```

  `main.py` reads `processed_data/train.csv` and calls `train_model(...)`. Edit `main.py` or call `train_model` directly to change algorithm, timesteps or save path.

- Train programmatically:

  ```python
  from model import train_model
  # train_model(algo='A2C'|'PPO'|'DDPG', train_csv='processed_data/train.csv', timesteps=100000)
  train_model(algo='PPO', timesteps=100000)
  ```

- Run tuning (Optuna)

  ```powershell
  python src\tune.py
  ```

  Tuning results will be saved under `tuning_results/` (CSV per algorithm).

- Evaluate the PPO+A2C ensemble (quick run):

  ```powershell
  python src\ppo_a2c_ensamble.py
  ```

Notes & tips

- Data paths: the repository currently expects processed data in `processed_data/` (see `data_acquisition.py` and other modules). If you prefer a single top-level `data/` folder, either pass a custom `save_path` to `get_stock_data()` or update the modules to read from `data/` instead of `processed_data/`.
- If you need GPU support for training, ensure PyTorch and stable-baselines3 are installed with CUDA support on your system.
- Reproducibility: `tune.py` sets seeds for NumPy and PyTorch. Consider setting seeds in other modules if you want deterministic runs.
- Long runs: training and tuning use non-trivial timesteps. For quick experiments reduce `timesteps` or `n_trials`.

Troubleshooting

- Missing packages: install with `pip install -r requirements.txt`.
- Missing data files: run `get_stock_data(...)` to generate `processed_data/train.csv` and `processed_data/test.csv`.
- Gym/env errors: ensure `gymnasium` version matches the code expectations and `stable-baselines3` is compatible with your PyTorch version.

Suggested next steps

- Decide whether you want `processed_data/` or a top-level `data/` folder and update the code or calls accordingly.
- Add a small script `scripts/fetch_train_data.py` that calls `get_stock_data(...)` with command-line args for reproducible data pulls.

If you'd like, I can create the `scripts/` helper, or update the modules to use a repo-level `data/` directory instead of `processed_data/`.
