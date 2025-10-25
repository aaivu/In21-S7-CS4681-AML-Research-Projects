# Research Proposal: RL:Model-Free Control

**Student:** 210043V
**Research Area:** RL:Model-Free Control
**Date:** 2025-09-01

## Abstract

This proposal targets **online, model-free continuous control** and investigates a compute-light extension of TD3, called **OurTD3**. The method preserves TD3’s backbone (twin critics, target-policy smoothing, delayed actor updates, Polyak averaging) and improves the **quality of learning signals** via two additions: (i) **agreement-weighted replay** that uses the native twin-critic TD-error agreement to reweight samples; and (ii) a small **critic value-improvement (VI) regularizer** that softly pulls Q toward a greedy (max) backup to counter TD3’s conservatism from the min backup. We evaluate on MuJoCo locomotion (Hopper-v4, Walker2d-v4, HalfCheetah-v4) under identical hyper-parameters to TD3, reporting final return, area-under-curve (AUC), steps-to-threshold, and divergence rate over ≥10 seeds. We hypothesize improved **sample-efficiency** and **stability** without behavior cloning or added networks, and provide ablations (+PER, +VMFER, +VI, grad-clip on/off) to isolate contributions.

## 1. Introduction

Model-free RL has achieved strong performance in continuous-action control using deterministic actor–critic methods (DPG/DDPG) and stabilizers introduced by **TD3** [1–3]. **SAC** extends this line with entropy regularization [4]. Despite progress, TD3’s outcomes remain sensitive to (a) **replay distribution** (which transitions are emphasized) and (b) **value-target bias/variance** (min backup can be overly pessimistic). This work proposes **OurTD3**, a minimal TD3 extension that improves signal quality via **twin-critic agreement-weighted replay** and a **small VI regularizer**, aiming for better learning curves and fewer collapses in standard MuJoCo tasks.

## 2. Problem Statement

How can we **improve sample-efficiency and stability** of TD3 in **online** continuous control **without** behavior cloning or extra ensembles? Specifically:
**P1.** Does replay reweighting by **twin-critic agreement** reduce noisy targets and variance?
**P2.** Does a **small VI** term counter TD3’s pessimism and improve final return while remaining stable?

## 3. Literature Review Summary

DDPG and DPG establish deterministic policy gradients for continuous control [1,2]. TD3 addresses overestimation via **clipped double Q (min backup), action smoothing, delayed actor updates** [3]. SAC maximizes return with entropy for robustness [4]. **PER** improves sample-efficiency by sampling high-TD-error items [5], while ensemble ideas (e.g., Bootstrapped DQN; **REDQ**) reduce target variance with multiple Qs [6,7]. Offline RL (e.g., **TD3-BC**) relies on behavior cloning to stay near dataset support [9]. **Gap:** Using **native twin-critic agreement** (already present in TD3) to **reweight replay** in online control is underexplored; combining this with a **tiny VI** term offers a simple, compute-light alternative to ensembles.

## 4. Research Objectives

### Primary Objective

Design, implement, and evaluate **OurTD3** to achieve **higher AUC**, **lower divergence**, and **equal or better final return** than TD3 on Hopper/Walker2d/HalfCheetah with minimal computational overhead.

### Secondary Objectives

- O1: Quantify the effect of **agreement-weighted replay (VMFER)** vs uniform and **PER**.
- O2: Study **VI coefficient** sensitivity (λ ∈ {0, 0.01, 0.05}).
- O3: Measure **variance across seeds** and **collapse rate** with/without gradient clipping.
- O4: Report **overhead** (wall-clock, GPU hours) relative to TD3.
- O5 (stretch): Check portability of agreement-weighting to **SAC** (pilot run).

## 5. Methodology

- **Algorithm:** TD3 backbone; critics trained with **weighted MSE** where weights come from **cosine agreement** between the two critics’ TD errors; add **VI auxiliary** MSE to a greedy (max) target with small λ; **actor** unchanged (maximize Q1). Apply **gradient-norm clipping** (e.g., 10.0).
- **Environments:** Hopper-v4, Walker2d-v4, HalfCheetah-v4 (Gymnasium + MuJoCo).
- **Setup:** 1M steps, eval every 5k (10 episodes), networks 256×256, batch 256, τ=0.005, policy_noise=0.2, noise_clip=0.5, actor_delay=2, buffer 1–2M, observation normalization.
- **Baselines/Ablations:** TD3; TD3+PER; **OurTD3** (+VMFER; +PER+VMFER; +VI on/off; grad-clip on/off); TD3-BC run **online** only for reference (no fixed dataset).
- **Metrics:** **Final return** (mean±std over ≥10 seeds), **AUC** of learning curves, **steps-to-threshold**, **divergence %**.
- **Analysis:** Compare learning curves and metric tables; sensitivity to κ (agreement strength) and λ (VI); report runtime overhead.

## 6. Expected Outcomes

- **Improved sample-efficiency** (higher AUC) and **lower divergence** vs TD3 on Hopper/Walker2d;
- **Small but consistent final-return gains** on HalfCheetah with λ≈0.01;
- **Negligible compute overhead**, as no extra critics/ensembles or BC are used;
- Reproducible code, plots, and ablation tables aligned with the Overleaf paper.

## 7. Timeline

| Week  | Task                                                |
| ----- | --------------------------------------------------- |
| 1–2   | Literature review & design finalization             |
| 3–4   | Implement VMFER + VI + grad-clip                    |
| 5–8   | Run baselines & ablations (all envs, ≥10 seeds)     |
| 9–12  | Aggregate metrics, plot curves, sensitivity studies |
| 13–15 | Analysis & write-up (paper + README)                |
| 16    | Final submission & artifact packaging               |

## 8. Resources Required

- **Software:** Python 3.10/3.11; PyTorch 2.x; Gymnasium 1.x; `gymnasium[mujoco]`; **MuJoCo** ≥3.x; NumPy; plotting (Matplotlib).
- **Hardware:** 1× CUDA GPU (e.g., RTX 3060+), 8–16 CPU cores, 16–32GB RAM.
- **Codebase:** TD3 baseline, OurTD3 implementation (VMFER, VI, PER, grad-clip), logging/plotting scripts.
- **Versioning:** Git repo with seeds, configs, and figure generation scripts.

## References

[1] Silver, D. et al. “Deterministic Policy Gradient Algorithms.” ICML, 2014.
[2] Lillicrap, T. P. et al. “Continuous Control with Deep Reinforcement Learning.” arXiv:1509.02971, 2016.
[3] Fujimoto, S., van Hoof, H., Meger, D. “Addressing Function Approximation Error in Actor-Critic Methods (TD3).” ICML, 2018.
[4] Haarnoja, T. et al. “Soft Actor-Critic.” ICML, 2018.
[5] Schaul, T. et al. “Prioritized Experience Replay.” ICLR, 2016.
[6] Osband, I. et al. “Deep Exploration via Bootstrapped DQN.” NeurIPS, 2016.
[7] Chen, X. et al. “Randomized Ensembled Double Q-Learning (REDQ).” NeurIPS, 2021.
[8] Watkins, C. J. C. H., Dayan, P. “Q-learning.” Machine Learning, 1992.
[9] Fujimoto, S., Gu, S. “A Minimalist Approach to Offline Reinforcement Learning (TD3-BC).” NeurIPS, 2021.
