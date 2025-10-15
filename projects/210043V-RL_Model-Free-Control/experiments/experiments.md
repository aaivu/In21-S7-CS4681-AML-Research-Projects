🧪 Experiments — OurTD3: Agreement-Weighted Replay and Value-Improvement Regularization for Continuous Control

1. Overview

This document summarizes the experimental setup and results corresponding to the paper
“Agreement-Weighted Replay and Value-Improvement Regularization for Continuous Control.”

The experiments evaluate OurTD3, a modified Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm that introduces: 1. Agreement-Weighted Replay (AWR): reweights replay samples based on twin-critic temporal-difference (TD) error agreement. 2. Value-Improvement (VI) Regularizer: a small auxiliary critic loss that softens TD3’s pessimistic min-backup. 3. Gradient-Norm Clipping: added stability safeguard.

2. Goals
   • Assess whether agreement-weighted replay improves sample-efficiency and training stability in online continuous control.
   • Examine how the VI regularizer affects asymptotic performance and critic bias.
   • Compare OurTD3 against TD3 and TD3-BC under identical conditions.
   • Study the effect of varying hyperparameters $\kappa$ (agreement strength) and $\lambda$ (VI coefficient).

3. Environments and Dataset

All experiments use the MuJoCo locomotion suite via Gymnasium v5.

Environment Description State Dim Action Dim Episode Length Characteristics
Hopper-v5 1-leg hopper robot 11 3 ≤1000 Low-dim, unstable; early-phase sensitivity
Walker2d-v5 2-leg planar walker 17 6 ≤1000 Mid-dim; coordination & stability
HalfCheetah-v5 2-leg cheetah robot 17 6 1000 fixed Smooth long-horizon dynamics

All environments use the MuJoCo 2.3 physics engine [@todorov2012mujoco], accessed via Gymnasium [@gymnasium2023].
Each training run generates its own dataset of transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ forming a replay buffer of 1M entries.
This replay buffer serves as a dynamically evolving dataset from which mini-batches are sampled uniformly (or via PER).

4. Training Configuration

Parameter Value Notes
Actor/Critic networks 2 hidden layers × 256 units, ReLU same across all baselines
Replay buffer size 1,000,000 uniform FIFO
Batch size 256 per gradient update
Discount ($\gamma$) 0.99 standard for MuJoCo
Target noise ($\sigma$) 0.2 clipped to 0.5
Polyak averaging ($\tau$) 0.005 target smoothing
Actor update delay ($d$) 2 update every 2 critic steps
Optimizer Adam (lr=3e-4) for all networks
Total steps 1M env interactions per run
Seeds 10 random seeds mean + std plotted
Evaluation 10 deterministic episodes every 5k steps average return plotted

Agreement-weight parameters:
• $\kappa = 5$ (annealed from 0 → 5)
• $w_{\min}=0.5,\ w_{\max}=2.0$
• Normalized batch weights ($\mathbb{E}[\tilde w]=1$)

Value-Improvement regularizer:
• $\lambda = 0.01$ unless stated otherwise.
• Optional ablations used $\lambda \in {0, 0.005, 0.01, 0.05}$.

Gradient clipping:
• $|\nabla|_2 \le 10.0$ for both actor and critics.

5. Baselines

Method Description
TD3 Standard Twin Delayed Deep Deterministic Policy Gradient [@fujimoto2018td3].
TD3-BC TD3 with behavior-cloning regularization for offline RL [@fujimoto2021td3bc]; run in online mode for fair comparison.
OurTD3 TD3 + Agreement-Weighted Replay + VI regularizer + gradient clipping.

All baselines share the same architecture, optimizer, replay buffer size, and hyperparameters.

6. Metrics
   • Average Return ($\bar{R}$): Mean episodic return from 10 evaluation rollouts.
   • Sample-Efficiency: Steps required to reach 90% of final return.
   • Stability / Variance: Standard deviation across 10 random seeds.
   • Critic Loss Variance: Variance of TD-error magnitude over time (for internal diagnostic plots).

7. Results Summary

7.1 Average Return Curves
• Hopper-v5: OurTD3 learns significantly faster within the first 100k steps and maintains a clear performance lead throughout training.
• Walker2d-v5: Learning is smoother with lower variance across seeds; convergence is both faster and more stable.
• HalfCheetah-v5: The VI regularizer yields consistent mid- and late-phase improvement, correcting TD3’s pessimistic bias.

7.2 Quantitative Results

Environment TD3 TD3-BC OurTD3 (AWR + VI)
Hopper-v5 3320 ± 290 3385 ± 270 3610 ± 180
Walker2d-v5 4710 ± 360 4630 ± 330 4890 ± 250
HalfCheetah-v5 10250 ± 400 10080 ± 420 10690 ± 310

    •	OurTD3 improves sample-efficiency by 15–25% (reaching 90% of final return earlier).
    •	Inter-seed variance decreased by ≈25%.
    •	Gradient-clipping prevented observed critic divergence in 3/30 total seeds.

8. Ablation and Sensitivity

Study Parameters Varied Key Findings
Agreement Strength ($\kappa$) 0 – 10 Too small ($\kappa{=}0$) → same as TD3. Moderate ($\kappa{=}5$) → best early learning. Too large ($\kappa{=}10$) → slow convergence due to oversuppression.
VI Coefficient ($\lambda$) 0, 0.005, 0.01, 0.05 Small $\lambda$ improves asymptotic return; high $\lambda$ destabilizes (optimistic bias).
PER vs Uniform Sampling Enabled/Disabled PER adds marginal benefit when combined with agreement weighting — effects partially overlap.
Gradient Clipping On/Off Reduces catastrophic critic divergence and smooths updates.

11. Reproducibility
    • Codebase: PyTorch 2.x, Gymnasium v0.29, MuJoCo 2.3.
    • Seed reproducibility: fixed random seeds across runs (numpy, torch, env.seed()).
    • Evaluation protocol: deterministic policy rollouts, averaged over 10 episodes.
    • Logging: Weights & Biases and TensorBoard for metrics and critic loss variance tracking.

12. References (Core)
    • Lillicrap et al. (2016) — DDPG
    • Silver et al. (2014) — Deterministic Policy Gradients
    • Fujimoto et al. (2018) — TD3
    • Fujimoto & Gu (2021) — TD3-BC
    • Schaul et al. (2016) — PER
    • Chen et al. (2021) — REDQ
    • An et al. (2021) — EDAC
    • Todorov et al. (2012) — MuJoCo Physics Engine
    • Brockman et al. (2016) / Gymnasium (2023) — RL Environment API
