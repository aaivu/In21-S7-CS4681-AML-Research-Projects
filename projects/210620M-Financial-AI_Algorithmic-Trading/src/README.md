## src — quick file summary & CLI

Files in this folder (short):

- `data_acquisition.py` — download OHLCV, compute indicators, bfill and save `train.csv`/`test.csv`.
- `env.py` — `StockTradingEnv` Gymnasium environment used by agents.
- `model.py` — `train_model(...)` wrapper to train and save PPO/A2C/DDPG models.
- `ppo_a2c_ensamble.py` — functions to train PPO+A2C and evaluate a weighted ensemble.
- `tune.py` — Optuna tuning utilities and `tune_model` entrypoint.
- `main.py` — small example that loads `processed_data` and calls `train_model`.

Create a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```

Quick note: the project expects `processed_data/` by default. Use `--out` or `--save_path` to override.


1) Data acquisition — `data_acquisition.py`

Arguments (suggested):
- `--ticker` (default: AAPL)
- `--start` (YYYY-MM-DD)
- `--end` (YYYY-MM-DD)
- `--out` / `--save_path` (output folder)

Example (from `src`):

```powershell
python data_acquisition.py --ticker MSFT --start 2015-01-01 --end 2020-12-31 --out processed_data
```


2) Train a model — `model.py` (wraps `train_model`)

Arguments (suggested):
- `--algo` (PPO|A2C|DDPG)
- `--train_csv` (default: processed_data/train.csv)
- `--timesteps` (int)
- `--save_path`

Example:

```powershell
python model.py --algo PPO --train_csv processed_data/train.csv --timesteps 50000 --save_path ppo_model
```

3) Train/evaluate ensemble — `ppo_a2c_ensamble.py`

Arguments (suggested):
- no arguments

Example (train + evaluate):

```powershell
python ppo_a2c_ensamble.py 
```

4) Hyperparameter tuning — `tune.py` (Optuna)

Arguments (suggested):
- no arguments

Example:

```powershell
python tune.py
```

