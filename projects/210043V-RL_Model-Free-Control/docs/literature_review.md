# Literature Review: RL: Model-Free Control

**Student:** 210043V
**Research Area:** RL: Model-Free Control
**Date:** 2025-09-01

## Abstract

This review surveys model-free reinforcement learning (RL) for **continuous control**, centering on deterministic actor–critic methods (DDPG/TD3), entropy-regularized SAC, replay/data-quality techniques, and ensemble-style uncertainty. It motivates your method (**OurTD3**) that keeps the TD3 backbone but improves **signal quality** via (i) **agreement-weighted replay** using native twin-critic agreement and (ii) a small **critic value-improvement (VI)** regularizer that counteracts pessimism from the clipped double-Q target. These choices align with Prioritized Experience Replay (PER) and ensemble-based uncertainty while remaining **compute-light** for **online** learning. Experiments on Hopper, Walker2d, and HalfCheetah indicate improved sample-efficiency and stability relative to TD3 (and TD3-BC when trained online as a reference).

## 1. Introduction

Model-free continuous control learns policies that output real-valued actions (e.g., torques) without learning a dynamics model. Deterministic policy gradients (DPG/DDPG) enable efficient off-policy learning, while **TD3** stabilizes training via twin critics (min backup), target-policy smoothing, and delayed actor updates. **SAC** introduces entropy regularization for robustness and exploration. Despite these advances, performance is sensitive to (a) the **replay distribution** and (b) **bias/variance** of value targets—precisely where **OurTD3** intervenes by improving the _quality_ of the critic’s learning signal.

## 2. Search Methodology

### Search Terms Used

- model-free continuous control; deterministic policy gradient; DDPG; TD3; SAC
- clipped double Q; overestimation bias; target policy smoothing
- experience replay; prioritized replay; replay weighting
- ensemble critics; uncertainty; REDQ; Bootstrapped DQN
- offline RL; TD3-BC; conservative targets

### Databases Searched

- [x] IEEE Xplore
- [x] ACM Digital Library
- [x] Google Scholar
- [x] ArXiv / OpenReview
- [ ] Other: —

### Time Period

2016–2024 (with earlier seminal theory where needed).

## 3. Key Areas of Research

### 3.1 Deterministic Actor–Critic for Continuous Control

**Scope.** Off-policy learning with a deterministic policy and twin critics.
**Consensus.** DPG/DDPG establish the framework; **TD3** reduces overestimation via clipped double-Q (min), action smoothing, and delayed policy updates.
**Relevance to OurTD3.** We **retain** TD3’s stabilizers and modify only how critic targets are trained/weighted.

**Key Papers:**

- Silver et al., 2014 — _Deterministic Policy Gradient (DPG)_: formalizes deterministic policy gradients [1].
- Lillicrap et al., 2016 — _DDPG_: scalable continuous control baseline [2].
- Fujimoto et al., 2018 — _TD3_: clipped double-Q, policy smoothing, delayed updates [3].

### 3.2 Replay & Uncertainty

**Problem.** Uniform replay can be sample-inefficient. Noisy targets amplify gradient variance.
**Directions.**

- **PER** samples high-TD-error transitions with importance correction [5].
- **Ensemble signals** estimate uncertainty from multiple Qs—Bootstrapped DQN [6].**REDQ** uses many lightweight Q-heads to reduce target variance [7].
  **Relevance to OurTD3.**
- Optional **PER** for efficiency.
- New **agreement-weighted replay**: use TD3’s _existing_ twin critics to compute a cosine-agreement weight; up-weight reliable samples (high agreement), down-weight contentious ones—no extra networks.

### 3.3 Offline RL (for contrast)

**Idea.** Train from a fixed dataset with **no interaction**; must avoid out-of-distribution actions.
**Representative.** **TD3-BC** adds an adaptive behavior-cloning penalty to the actor for stability in the offline regime [9].
**Relevance.** OurTD3 targets **online** control (no BC term). TD3-BC is included only as a reference when run online for comparison.

## 4. Research Gaps and Opportunities

### Gap 1 — Replay prioritization using **native twin-critic agreement** in TD3 is underexplored

**Why it matters:** PER focuses on TD-error magnitude, which can overweight noise.Ensemble approaches help but often add compute.
**How your project addresses it:** **Agreement-weighted replay** leverages TD3’s two critics to form a lightweight reliability weight—no additional networks or heads.

### Gap 2 — TD3’s **min** backup can be overly pessimistic

**Why it matters:** Excess pessimism slows learning and can cap performance in smooth locomotion tasks.
**How your project addresses it:** Add a small **VI** auxiliary loss that pulls Q toward a greedy (max) target (grounded in Q-learning) while keeping the main TD3 min-target unchanged.

## 5. Theoretical Framework

- **Deterministic Policy Gradient** (DPG) [1]:

- **TD3 target** [3]:

- **Agreement-weighted critic loss (ours):**

- **VI regularizer (ours):**

- **Actor update:** [3].

## 6. Methodology Insights

- **Typical setups:** MuJoCo locomotion (Hopper/Walker2d/HalfCheetah), 1M steps, MLP 256×256, batch 256, (\tau=0.005), policy noise 0.2, noise clip 0.5, actor delay 2 [3].
- **Metrics:** Final return (mean±std across seeds), **AUC** of learning curves, steps-to-threshold, divergence %.
- **Our findings summary:** Agreement-weighted replay improves sample-efficiency and stability; a small **VI** (e.g., 0.01) boosts final return on smoother dynamics—both are **drop-in** and retain TD3’s compute profile.

## 7. Conclusion

The literature establishes TD3 as a strong online baseline and highlights the importance of **replay quality** and **uncertainty handling**. **OurTD3** advances this line with **compute-light** mechanisms agreement-weighted replay and a tiny VI regularizer grounded in Q-learning—that raise sample-efficiency and stability in continuous control, without behavior cloning.

## References

1. Silver, D., Lever, G., Heess, N., et al. (2014). _Deterministic Policy Gradient Algorithms_ (ICML).
2. Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2016). _Continuous Control with Deep Reinforcement Learning_ (arXiv:1509.02971).
3. Fujimoto, S., van Hoof, H., Meger, D. (2018). _Addressing Function Approximation Error in Actor-Critic Methods (TD3)_ (ICML).
4. Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). _Soft Actor-Critic_ (ICML).
5. Schaul, T., Quan, J., Antonoglou, I., Silver, D. (2016). _Prioritized Experience Replay_ (ICLR).
6. Osband, I., Blundell, C., Pritzel, A., Van Roy, B. (2016). _Deep Exploration via Bootstrapped DQN_ (NeurIPS).
7. Chen, X., Zhou, Z., Wang, Z., et al. (2021). _Randomized Ensembled Double Q-Learning (REDQ)_ (NeurIPS).
8. Watkins, C. J. C. H., Dayan, P. (1992). _Q-learning_ (Machine Learning).
9. Fujimoto, S., Gu, S. (2021). _A Minimalist Approach to Offline Reinforcement Learning (TD3-BC)_ (NeurIPS).
