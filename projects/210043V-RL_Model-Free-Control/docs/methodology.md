# Methodology: RL:Model-Free Control

**Student:** 210043V
**Research Area:** RL:Model-Free Control
**Date:** 2025-09-01

## 1. Overview

We study **online, model-free continuous control** and propose **OurTD3**—a lightweight extension of TD3 that keeps the standard backbone (twin critics, target-policy smoothing, delayed actor updates, Polyak averaging) and improves the **quality of learning signals** via:

- **Agreement-weighted replay (VMFER):** up-weights transitions where the two critics’ TD errors agree (proxy for reliable targets) and down-weights high-disagreement samples.
- **Optional PER:** standard prioritized replay as a comparative knob.
- **Critic Value-Improvement (VI) regularizer:** a small auxiliary loss (e.g., λ=0.01) that softly pulls Q toward a greedy (max) target to counter excess pessimism from TD3’s min backup.
- **Gradient-norm clipping:** stability guard (e.g., clip=10) for actor and critics.

## 2. Research Design

- **Setting:** Online/off-policy RL with on-the-fly data collection in MuJoCo locomotion tasks.
- **Design:** Compare **TD3** vs **OurTD3** under identical hyperparameters; run **ablations** (+PER, +VMFER, +VI, grad-clip on/off).
- **Hypotheses:** (H1) Agreement-weighted replay improves sample-efficiency and reduces collapse rate; (H2) a small VI term increases final return on smooth dynamics (e.g., HalfCheetah) without destabilizing training.

## 3. Data Collection

### 3.1 Data Sources

- **Environment interactions (online):** Hopper-v4, Walker2d-v4, HalfCheetah-v4 (Gymnasium + MuJoCo).
- **No fixed offline datasets** used for training; TD3-BC only referenced when run online for parity.

### 3.2 Data Description

- Transitions ((s,a,r,s',d)) collected by the current policy with Gaussian exploration.
- Replay buffer size: (1\text{–}2\times 10^6) transitions (per env).
- Evaluation rollouts are deterministic (no exploration noise).

### 3.3 Data Preprocessing

- **Observation normalization:** running mean/std per state dimension.
- **Action bounds:** tanh-squashed actor with env-specific scaling.
- **Replay weighting:**

  - **VMFER:** weight (w \propto \exp(\kappa \cdot \cos(\delta*1,\delta_2))); normalize to mean 1; clip to ([w*{\min}, w\_{\max}]).
  - **PER (optional):** α≈0.6, β annealed → 1.0.

- **Seeding:** ≥10 seeds per method for statistical reliability.

## 4. Model Architecture

- **Actor:** MLP (256!-!256), ReLU, deterministic (\pi(s)) with tanh output, target-policy smoothing during target computation.
- **Twin critics (Q1,Q2):** separate MLPs (256!-!256), ReLU.
- **Targets:** Polyak averaging (τ=0.005).
- **Losses:**

  - **TD3 critic loss** with **min** target (y).
  - **Agreement-weighted** MSE on critic TD errors (weights from VMFER).
  - **VI regularizer (λ small):** additional MSE to (y\_{\max}) (greedy target).
  - **Actor:** maximize (Q_1(s,\pi(s))) every `actor_delay=2` steps.

- **Stability:** gradient-norm clipping (e.g., 10.0) on actor/critics.

## 5. Experimental Setup

### 5.1 Evaluation Metrics

- **Final return** (mean ± std across seeds).
- **AUC** of learning curves up to 1M steps (sample-efficiency).
- **Steps-to-threshold** (first time reaching a fixed score per env).
- **Divergence rate** (% runs that collapse or NaN).

### 5.2 Baseline Models

- **TD3 (vanilla).**
- **OurTD3 variants:** +VMFER; +PER; +PER+VMFER; (+VI on/off); grad-clip on/off.
- **TD3-BC (online reference only):** included to show that our gains arise without BC.

### 5.3 Hardware/Software Requirements

- **Software:** Python 3.10/3.11, PyTorch 2.x, Gymnasium, `gymnasium[mujoco]`, MuJoCo ≥3.x, NumPy.
- **Typical hyperparameters:** policy_noise=0.2, noise_clip=0.5, actor_delay=2, τ=0.005, batch=256, buffer=1–2M.
- **Hardware:** 1× mid-range GPU (e.g., RTX 3060+), 8–16 CPU cores, 16–32GB RAM.
- **Note:** ensure `mujoco` installed and importable to avoid runtime errors.

## 6. Implementation Plan

| Phase   | Tasks                                                               | Duration | Deliverables                          |
| ------- | ------------------------------------------------------------------- | -------- | ------------------------------------- |
| Phase 1 | Env setup; replay buffer; obs norm; logging                         | 1 week   | Reproducible training scaffold        |
| Phase 2 | Implement VMFER weights; integrate PER; add VI and grad-clip        | 2 week   | OurTD3 training code                  |
| Phase 3 | Run baselines + ablations on Hopper/Walker/Cheetah (≥10 seeds each) | 2 weeks  | Curves, AUC, tables, divergence stats |
| Phase 4 | Analysis & write-up; finalize plots for paper/README                | 2 week   | Results section + figures             |

## 7. Risk Analysis

- **MuJoCo/Gym install issues** → _Mitigation:_ lock versions; smoke tests (`import mujoco`, `gym.make('Hopper-v4')`).
- **Over-weighting easy states (VMFER)** → _Mitigation:_ normalize/clip weights; mix a fraction of uniform replay.
- **Over-optimism from VI** → _Mitigation:_ keep λ small (0.01), monitor Q-value scale; disable if Q explodes.
- **High variance across seeds** → _Mitigation:_ ≥10 seeds; report mean±std and AUC; enable grad-clip.

## 8. Expected Outcomes

- **Higher AUC** and **fewer collapses** vs TD3 across Hopper/Walker2d/HalfCheetah.
- **Small but consistent final-return gains** (esp. HalfCheetah) with VI=0.01.
