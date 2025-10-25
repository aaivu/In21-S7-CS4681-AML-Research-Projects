# Experiments Guide: Folder Structure and How to Run

This guide summarizes the project layout and how to run the main training scripts for both OurTD3 and the TD3 baseline.

## 1) Folder Structure (key paths)

```
projects/210043V-RL_Model-Free-Control/
├─ README.md
├─ requirements.txt                 # Project-specific dependencies (Gymnasium, MuJoCo, Torch, etc.)
├─ data/                            # (placeholder)
├─ docs/
│  ├─ usage_instructions.md         # Extra usage notes and hyperparameters
│  ├─ research_proposal.md
│  ├─ literature_review.md
│  ├─ methodology.md
│  └─ progress_reports/
├─ experiments/
│  └─ experiments.md                # This file
├─ results/
│  ├─ graphs/                       # Generated comparison plots
│  ├─ halfcheetah-v5/               # Saved curves (.npy) per env/method
│  ├─ hopper-v5/
│  └─ Walker2d-v5/
└─ src/
   ├─ graphs/
   │  ├─ plotter.py
   │  └─ plot_compare.py
   ├─ methods/
   │  ├─ OurMethod/
   │  │  ├─ train_ourTD3.py         # MAIN: OurTD3 training runner
   │  │  └─ our_td3/
   │  │     ├─ policy.py            # OurTD3 agent + config
   │  │     ├─ td3_core.py          # Actor/Critic networks and utils
   │  │     ├─ replay_per.py        # Prioritized replay (optional)
   │  │     ├─ replay_vmfer.py      # Agreement-weighted replay
   │  │     └─ vi_td3.py            # VI regularizer helpers
   │  └─ Others/
   │     ├─ main.py                 # MAIN: TD3 baseline runner (Gymnasium v5)
   │     ├─ TD3/TD3.py              # TD3 implementation
   │     └─ TD3-BC/TD3_BC.py        # TD3-BC reference implementation
   └─ models/                       # Saved model checkpoints (.pt)
```

Notes:
- OurTD3-related code lives under `src/methods/OurMethod/`.
- Baseline TD3 (and TD3-BC reference) are under `src/methods/Others/`.
- Plot scripts under `src/graphs/` can aggregate `.npy` files in `results/` to reproduce comparison figures.

## 2) Environment Setup (one-time)

From `projects/210043V-RL_Model-Free-Control/`:

```
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

If using GPU, install a CUDA-enabled PyTorch wheel matching your CUDA version.

## 3) How To Run — OurTD3 (recommended)

Entry point: `src/methods/OurMethod/train_ourTD3.py`

Common flags:
- `--env` Gymnasium MuJoCo env id (e.g., `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`).
- `--replay` one of `uniform`, `per`, `vmfer` (agreement-weighted).
- `--use-vi` enable Value-Improvement regularizer.
- `--steps` total environment steps (default 1,000,000).
- `--eval-every` evaluation frequency in steps.
- `--batch-size`, `--actor-delay`, `--policy-noise`, `--noise-clip`, `--tau`.
- `--outdir` output directory for curves and checkpoints.
- `--tag` optional label used in filenames (otherwise auto-inferred).

Examples:

```
# OurTD3 + VMFER + VI on Hopper-v5
python src/methods/OurMethod/train_ourTD3.py ^
  --env Hopper-v5 ^
  --replay vmfer ^
  --use-vi ^
  --steps 1000000 ^
  --eval-every 5000 ^
  --seed 0 ^
  --outdir results/hopper-v5/OurTD3

# Vanilla TD3 (uniform replay, no VI)
python src/methods/OurMethod/train_ourTD3.py ^
  --env Walker2d-v5 ^
  --replay uniform ^
  --steps 1000000 ^
  --eval-every 5000 ^
  --seed 0 ^
  --outdir results/Walker2d-v5/TD3 ^
  --tag TD3

# Evaluate a saved actor checkpoint
python src/methods/OurMethod/train_ourTD3.py ^
  --env HalfCheetah-v5 ^
  --eval-only ^
  --checkpoint src/models/TD3HalfCheetah-v50_final.pt
```

Outputs (per run):
- Curves: `{outdir}/{tag}{env}{seed}_train_returns.npy`, `{outdir}/{tag}{env}{seed}_train_steps.npy`, `{outdir}/{tag}{env}{seed}.npy`
- Final model: `{outdir}/{tag}{env}{seed}_final.pt`

## 4) How To Run — TD3 Baseline

Entry point: `src/methods/Others/main.py`

Common flags:
- `--policy TD3` (default), `--env Hopper-v5`, `--seed 0`
- `--max_timesteps 1000000`, `--start_timesteps 25000`, `--eval_freq 5000`
- `--batch_size 256`, `--discount 0.99`, `--tau 0.005`, `--policy_noise 0.2`, `--noise_clip 0.5`, `--policy_freq 2`
- `--save_model` to write checkpoints under `./models`

Example:

```
python src/methods/Others/main.py ^
  --policy TD3 ^
  --env Hopper-v5 ^
  --seed 0 ^
  --max_timesteps 1000000 ^
  --eval_freq 5000 ^
  --save_model
```

Outputs:
- Curves: `./results/TD3_{env}_{seed}.npy`
- Model (if `--save_model`): `./models/TD3_{env}_{seed}`

Note:
- The TD3-BC reference implementation is in `src/methods/Others/TD3-BC/TD3_BC.py`. If you need to run TD3-BC, adapt `main.py` to import and instantiate it similarly to `TD3`.

## 5) Plotting (optional)

Use the scripts in `src/graphs/` to compare runs and export figures, e.g.:

```
python src/graphs/plot_compare.py
```

Point the script(s) at the directories where your `.npy` curves are saved (e.g., `results/hopper-v5/OurTD3`, `results/Walker2d-v5/TD3`).

---

Quick tip: `docs/usage_instructions.md` contains additional notes on hyperparameters, multi-seed experiments, and troubleshooting.

