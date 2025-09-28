# Student Usage Instructions

These instructions explain how to **run, evaluate, and reproduce** the results for **OurTD3** (online, model-free continuous control) on MuJoCo tasks. They match the methodology and terminology used in your Overleaf paper.

---

## 1) Prerequisites

- **Python:** 3.10 or 3.11
- **Packages:** `torch`, `numpy`, `gymnasium`, `gymnasium[mujoco]`, `mujoco` (>= 3.x), `matplotlib`
- **GPU (optional):** CUDA-capable GPU recommended for faster training

### Install (example)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install numpy matplotlib gymnasium gymnasium[mujoco] mujoco
```

### Quick checks

```python
python -c "import mujoco, gymnasium as gym; print('MuJoCo OK'); gym.make('Hopper-v4'); print('Gym OK')"
```

If you see `No module named 'mujoco'` or env creation fails, reinstall `mujoco` and `gymnasium[mujoco]`, then retry.

---

## 2) Repository layout (expected)

````
src/methods/OurMethod/our_td3
├─ our_td3.py              # OurTD3 algorithm (agreement weights, VI, grad-clip)
├─ td3_core.py             # Actor/Critic modules and utilities
├─ replay_per.py           # Prioritized replay (optional)
├─ replay_vmfer.py         # Agreement-weighted replay (VMFER)
├─ vi_td3.py               # VI regularizer helpers (if factored)

src/methods/OurMethod
├─train_ourTd3.py           # training script (OurTD3 runner)

src/methods/Others
├─TD3                       TD3 baseline Algorithm
├─TD3-BC                    TD3 + BC Algorithm
├─ main.py                  # training script (TD3/TD3BC)



## 3) Quick start (single run)

Train **OurTD3** with **agreement-weighted replay (VMFER)** and **VI regularizer** on **Hopper-v4**:

```bash
python train_td3.py \
  --env Hopper-v4 \
  --algo ourtd3 \
  --replay vmfer \
  --kappa 5 \
  --use_vi true \
  --vi_coef 0.01 \
  --grad_clip 10.0 \
  --total_steps 1000000 \
  --eval_every 5000 \
  --seed 0 \
  --logdir runs/hopper/ourtd3_vmfer_vi001_s0
````

Common toggles:

- **Disable VI:** `--use_vi false` or `--vi_coef 0.0`
- **Enable PER:** `--replay per --per_alpha 0.6 --per_beta_start 0.4 --per_beta_end 1.0`
- **Vanilla TD3 baseline:** `--algo td3 --replay uniform`
- **TD3-BC (online reference only):** `--algo td3bc --bc_alpha 2.5` (still interacting with env)

> If your script uses different flag names, map them accordingly (e.g., `--policy_noise`, `--noise_clip`, `--actor_delay`, `--tau`, `--batch`).

---

## 4) Recommended hyperparameters (match the paper)

- **Shared (TD3/OurTD3):**
  `policy_noise=0.2`, `noise_clip=0.5`, `actor_delay=2`, `tau=0.005`, `batch=256`, `buffer_size=1e6`
- **VMFER (agreement weighting):**
  `kappa=5`, normalize weights to mean 1, clip to `[0.5, 2.0]`
- **PER (optional):**
  `alpha=0.6`, `beta` anneal `0.4 → 1.0` over training
- **VI regularizer:**
  `vi_coef=0.01` (keep small; raise cautiously to `0.05` max)
- **Gradient clipping:**
  `clip=10.0` for both actor and critics
- **Observation normalization:** running mean/std per state dim (recommended)

---

## 5) Multi-seed evaluation (reproducibility)

Run **10 seeds** per method per environment:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python train_td3.py --env Hopper-v4 --algo td3 --replay uniform \
    --total_steps 1000000 --seed $s --logdir runs/hopper/td3_s${s}
done

for s in 0 1 2 3 4 5 6 7 8 9; do
  python train_td3.py --env Hopper-v4 --algo ourtd3 --replay vmfer \
    --kappa 5 --use_vi true --vi_coef 0.01 --grad_clip 10.0 \
    --total_steps 1000000 --seed $s --logdir runs/hopper/ourtd3_vmfer_vi001_s${s}
done
```

Repeat for **Walker2d-v4** and **HalfCheetah-v4**.

---

## 6) Metrics & plots (match paper/README)

- **Metrics:** Final return (mean±std across seeds), **AUC** of learning curves, **steps-to-threshold**, **divergence %**.
- **Plots:** learning curves with mean ± shaded std.

Typical export targets (as in your figures):

- `results/graph/hopper_compareALL.png`
- `results/graph/walker_compareAll.png`
- `results/graph/halfcheetah_compareAll.png`

If your script supports a `--plot` or a separate `plot_results.py`, run it pointing to `runs/` to generate the above images.

---

## 7) Troubleshooting

- **`No module named 'mujoco'`**
  `pip install mujoco gymnasium[mujoco]` and re-run the quick check.
- **Env creation fails (version mismatch)**
  Use Gymnasium v1.x and **MuJoCo v5 env IDs** (e.g., `Hopper-v5`).
- **Training collapse/NaNs**
  Lower `vi_coef` (e.g., 0.005→0.01), tighten weight clipping for VMFER (e.g., `[0.7,1.5]`), ensure gradient clipping is enabled, and verify observation normalization.
- **PER over-focus**
  Mix in uniform replay (e.g., 20%) or anneal `alpha`/`beta` more gently.

---

## 8) What to report (for your write-up)

- **Setup:** env IDs, total steps (1M), eval cadence (every 5k), eval episodes per check (e.g., 10), seeds (≥10), network sizes (256×256), shared hyperparams.
- **Baselines:** TD3; OurTD3 (+VMFER, +PER, +VI); TD3-BC (online ref).
- **Results:** Final return ± std; **AUC**; **steps-to-threshold**; **divergence %**; the three plots listed above.
- **Ablation takeaways:** VMFER → stability & sample-efficiency; VI (0.01) → final return (esp. HalfCheetah); grad-clip → fewer collapses.

---

**That’s it.** Follow the steps above to reproduce the paper’s continuous-control results and to create the exact plots referenced in your Overleaf document.
