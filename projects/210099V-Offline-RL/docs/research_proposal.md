# Research Proposal: QR-DQN

**Student:** 210099V  
**Research Area:** Offline RL  
**Date:** 2025-09-01  

---

## Abstract
This proposal investigates a minimal yet impactful modification to **Quantile Regression Deep Q-Networks (QR-DQN)**: replacing the standard 1-step temporal-difference (TD) target with an **n-step bootstrap (n = 3)** while keeping all other design axes unchanged. Distributional RL methods such as QR-DQN are strong, practical baselines, but the isolated contribution of multi-step targets is typically reported only within “bundled” agents (e.g., Rainbow), confounding attribution. We propose a controlled ablation that fixes architecture, optimizer, replay, exploration, and target-update cadence, varying only the TD horizon. On the classic-control benchmark **CartPole-v1**, we hypothesize improved credit propagation, higher final returns at equal sample budgets, and reduced cross-seed variance. The study includes multi-seed training, large-sample greedy evaluations, and robustness checks (e.g., quantile-Huber loss). Deliverables include a clean, reproducible codebase and guidelines for practitioners on when short n-step returns benefit distributional value learning. Outcomes will clarify n-step’s stand-alone value, supporting evidence-based recommendations for small, low-effort upgrades to QR-DQN before adopting heavier additions like prioritized replay or dueling heads.

---

## 1. Introduction
Value-based deep RL traditionally estimates expected returns (DQN), whereas distributional RL models the full return distribution, often improving stability and sample efficiency. **QR-DQN** approximates quantiles via the pinball loss and is widely used due to its simplicity and effectiveness. While **Rainbow** demonstrated large gains from combining multiple tweaks—including n-step—there is limited controlled evidence isolating n-step’s impact within QR-DQN. This work targets that gap with a clean ablation.

---

## 2. Problem Statement
There is no systematic, controlled assessment of how short n-step returns (e.g., n = 3) affect learning dynamics and performance inside QR-DQN when all other factors are held constant. Practitioners lack clear guidance on whether adopting n-step alone is a low-effort, high-impact improvement on small, low-stochasticity tasks.

---

## 3. Literature Review Summary
Distributional RL (C51, QR-DQN, IQN, FQF) often outperforms expectation-based value learning. Multi-step TD targets accelerate credit propagation but can increase variance; **off-policy corrections** (Retrace, Tree-Backup, Q(λ)/Q(σ)) mitigate mismatch. **Rainbow** bundles these ideas, obscuring single-factor attribution. Reproducibility work underscores multi-seed, controlled comparisons. The literature suggests n-step should help on CartPole, but a focused QR-DQN ablation is underreported.

---

## 4. Research Objectives

### Primary Objective
Quantify the isolated effect of replacing 1-step TD with **n-step (n = 3)** in QR-DQN under a fixed training budget.

### Secondary Objectives
- Compare sample efficiency (AUC, time-to-threshold).  
- Assess cross-seed stability and variance.  
- Run robustness checks (quantile-Huber vs. pinball; brief n ∈ {1,3,5} sweep if time permits).  
- Release a minimal, reproducible codebase and practitioner guidelines.

---

## 5. Methodology

**Setting:** CartPole-v1; discrete actions; γ = 0.99.  

**Model:** MLP (128–128, ReLU); 51 quantiles; greedy action over mean of quantiles; **Double-Q** bootstrap; target copy every 1,000 steps.  

**Optimization:** Adam (lr 5e-4, ε = 1e-8); grad-norm clip 10; replay capacity 100k; batch 64; learn-start 1,000 steps.  

**Exploration:** ε-greedy 1.0 → 0.01 linearly over 25k steps, then fixed.  

**Ablation:** **1-step vs. n-step (n = 3)** targets; everything else identical.  

**Evaluation:** 50k env-steps per seed; interim greedy probes; final **1,000-episode** greedy evaluation per seed; multi-seed stats (mean/median/SD), AUC, time-to-threshold.

---

## 6. Expected Outcomes
- Higher final greedy returns and faster learning with n = 3 versus 1-step at equal budgets.  
- Lower cross-seed variance (improved stability).  
- Clear, reproducible evidence for adopting short n-step in QR-DQN on classic-control tasks.  
- Public code and concise guidelines for practitioners.

---

## 7. Timeline

| **Week** | **Task** |
|---------:|----------|
| 1–2 | Literature review; finalize protocol and metrics; seed & env pinning |
| 3–4 | Implement QR-DQN (1-step) reference; add n-step path; unit tests for targets |
| 5–8 | Training runs over ≥5 seeds; interim probes; logging & plotting utilities |
| 9–12 | Large-sample final evaluations; robustness run (quantile-Huber / brief n-sweep) |
| 13–15 | Analysis (AUC, time-to-threshold, variance); write figures and tables |
| 16 | Final report, code cleanup, artifact release |

---

## 8. Resources Required
- **Software:** Python 3.10+, PyTorch 2.x, Gym/Gymnasium, NumPy, Matplotlib, TQDM.  
- **Hardware:** CPU sufficient (GPU optional).  
- **Artifacts:** Version-locked environment (`requirements.txt`), seeds, plotting scripts.  
- **Repository:** GitHub project with scripts for train/eval/plot and README.

---

## References

1. V. Mnih et al., “Human-level control through deep reinforcement learning,” *Nature*, 2015.  
2. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.  
3. M. G. Bellemare, W. Dabney, and R. Munos, “A distributional perspective on reinforcement learning,” *ICML*, 2017.  
4. W. Dabney et al., “Distributional RL with quantile regression,” *AAAI*, 2018.  
5. W. Dabney et al., “Implicit quantile networks,” *ICML*, 2018.  
6. M. Hessel et al., “Rainbow: Combining improvements in deep RL,” *AAAI*, 2018.  
7. R. Munos et al., “Safe and efficient off-policy RL (Retrace),” *NeurIPS*, 2016.  
8. P. Henderson et al., “Deep RL that matters,” *AAAI*, 2018.  
