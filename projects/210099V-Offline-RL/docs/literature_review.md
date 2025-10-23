# Literature Review: QR-DQN

**Student:** 210099V  
**Research Area:** Offline RL  
**Date:** 2025-09-01  

---

## Abstract
This review surveys work most relevant to upgrading **Quantile Regression DQN (QR-DQN)** with multi-step targets. It covers:  
(i) distributional value learning (C51, QR-DQN, IQN, FQF),  
(ii) multi-step bootstrapping and off-policy corrections (Retrace, Q(λ), Q(σ), Tree-Backup),  
(iii) bundled improvements (Rainbow) versus isolated ablations and reproducibility practices, and  
(iv) risk-sensitive objectives (CVaR) that motivate accurate lower-tail quantiles.  
The key takeaway is that short multi-step horizons (e.g., n≈3–5) often improve credit propagation in distributional agents without destabilizing off-policy learning on low-stochasticity classic-control tasks, and that the isolated effect of n-step inside QR-DQN is under-documented relative to bundled agents.

---

## 1. Introduction
Value-based deep RL traditionally learns expected returns (DQN). Distributional RL instead models the return distribution, improving stability and sample efficiency. QR-DQN approximates quantiles with the pinball loss and is a practical, widely used baseline. While Rainbow popularized n-step returns inside a bundle of enhancements, fewer studies ablate n-step alone within QR-DQN under controlled conditions. This review situates your focused contribution—replacing 1-step TD with an n-step target (n=3) in QR-DQN—within the surrounding literature.

---

## 2. Search Methodology

### Search Terms Used
- “distributional reinforcement learning”, “C51”, “quantile regression DQN”, “QR-DQN”  
- “implicit quantile networks”, “IQN”; “fully parameterized quantile functions”, “FQF”  
- “multi-step returns”, “n-step TD”, “TD(λ)”, “eligibility traces”  
- “off-policy corrections”, “Retrace”, “Tree-Backup”, “Q(σ)”  
- “Rainbow DQN”, “prioritized replay”, “dueling networks”, “noisy exploration”  
- “risk-sensitive RL”, “CVaR RL”, “quantiles and risk”  
- “CartPole”, “classic control”, “reproducibility in deep RL”  

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  

### Time Period
2017–2024 focus (foundational earlier works from 1988–2016 included for context).

---

## 3. Key Areas of Research

### 3.1 Distributional Value Learning
Distributional RL models the full return distribution rather than its mean, often yielding better learning dynamics.

**Key Papers:**
- Bellemare et al., 2017 — Introduces the distributional perspective (C51 categorical support) and shows performance gains over expectation-based DQN.  
- Dabney et al., 2018 (QR-DQN) — Replaces categorical projection with quantile regression using the pinball/quantile-Huber loss; simpler objective, strong results.  
- Dabney et al., 2018 (IQN) — Learns a continuous quantile function via sampled τ, improving flexibility and performance.  
- Yang et al., 2019 (FQF) — Learns both quantile fractions and values end-to-end, adaptively allocating capacity over the distribution.  

**Relevance:** You retain the QR-DQN head and loss; your sole change is the target horizon (1-step → n-step), allowing clearer attribution.

---

### 3.2 Multi-Step Bootstrapping and Off-Policy Corrections
Multi-step TD accelerates credit propagation (bias–variance trade-off). Off-policy usage may need corrections.

**Key Papers:**
- Sutton & Barto, 2018 — Textbook treatment of TD(λ), n-step returns, traces, and the bias–variance trade-off.  
- Munos et al., 2016 (Retrace) — Safe/effective off-policy multi-step returns with truncated importance sampling; widely used with replay.  
- De Asis et al., 2018 (Q(σ)) — Unifies Expected-SARSA/Q-learning via a σ parameter; clarifies design space between expectation and sampling.  
- Harutyunyan et al., 2016 (Q(λ) off-policy) — Off-policy corrections for trace-based control.  
- Precup et al., 2000 (Tree-Backup) — Expectation-style backups that avoid explicit IS ratios.  

**Relevance:** On CartPole with ε-greedy behavior, a short horizon (n=3) typically helps without additional correction machinery.

---

### 3.3 Bundled Agents vs. Isolated Ablations
Rainbow bundles double Q, dueling, PER, noisy nets, distributional head, and n-step.

**Key Papers:**
- Hessel et al., 2018 (Rainbow) — Demonstrates large gains from combining improvements including n-step.  
- Henderson et al., 2018 — Recommends multi-seed, controlled comparisons; warns about confounding factors and fragile results.  

**Relevance:** Your study follows best practice by changing only n-step inside QR-DQN and reporting multi-seed outcomes.

---

### 3.4 Risk-Sensitive Objectives and Quantiles
Distributional outputs naturally expose tail behavior; CVaR focuses on lower-tail risk.

**Key Papers:**
- Chow et al., 2015 (CVaR RL) — Frames risk-sensitive control via CVaR optimization.  
- Bellemare et al., 2017 — Distributional view supports tail-aware summaries (e.g., quantiles/CVaR).  

**Relevance:** Although you evaluate risk-neutral (mean over quantiles), accurate lower quantiles from QR-DQN plus n-step can benefit future risk-aware extensions.

---

### 3.5 Classic Control Foundations
- Mnih et al., 2015 (DQN) — Baseline replay/target network formulation.  
- van Hasselt et al., 2016 (Double Q), Wang et al., 2016 (Dueling), Schaul et al., 2016 (PER), Fortunato et al., 2018 (NoisyNets) — Common architectural/exploration tweaks you keep fixed to isolate n-step.  
- Brockman et al., 2016 (Gym) — CartPole benchmark/task specs.  

---

## 4. Research Gaps and Opportunities

### Gap 1: Isolated effect of n-step inside QR-DQN on classic-control tasks
**Why it matters:** Rainbow shows gains with many simultaneous changes; attribution to n-step alone is unclear.  
**How your project addresses it:** You insert n-step (n=3) without altering replay, network, exploration, or target-update cadence and present multi-seed A/B evidence.

### Gap 2: Sensitivity of QR-DQN+n-step to γ, target-update cadence, and quantile count
**Why it matters:** Multi-step interacts with discounting and bootstrapping frequency; distributional heads add quantile-count effects.  
**How your project addresses it:** You propose follow-up ablations (n∈{1,2,3,5}, γ sweeps, target-update τ, and 51 vs. fewer quantiles) and provide a minimal, reproducible codebase to enable them.

### Gap 3: Off-policy corrections when behavior diverges from target in distributional settings
**Why it matters:** Larger behavior–target mismatch may require corrections (Retrace/V-trace/Tree-Backup).  
**How your project addresses it:** You justify why CartPole+ε-greedy needs no extra correction and outline how to plug Retrace into the same QR-DQN scaffold for future work.

---

## 5. Theoretical Framework
- **MDP & Return Distribution:** Standard MDP with discount γ; distributional RL models (Z^π(s,a)) rather than (E[Z^π]).  
- **Quantile Approximation:** QR-DQN represents Z via N quantiles at fixed fractions {τ_i} and trains with the pinball (or quantile-Huber) loss.  
- **Action Selection:** Greedy over the mean of learned quantiles (risk-neutral).  
- **n-Step Bootstrapping:** Target R_t^{(n)} = Σ γ^k r_{t+k} + γ^n (1-d) Z(s_{t+n}, a^*). Short n reduces bootstrapping bias and speeds credit propagation.

---

## 6. Methodology Insights
**Common Methods:** Experience replay, target networks, ε-greedy, Adam/AdamW, multi-seed reporting, classic-control and Atari/MinAtar benchmarks.  

**For Your Study:**  
- Keep everything fixed except the TD horizon (1-step vs. n-step).  
- Use uniform replay (capacity ~1e5), batch 64, double Q bootstrapping, hard target updates, 51 quantiles.  
- Evaluate greedy returns with multiple seeds; report mean/median/SD and per-seed deltas.  
- Plot learning curves and seed-wise bars; include 1k-episode post-training evaluations for robustness.

---

## 7. Conclusion
The literature supports distributional value learning and the utility of n-step targets, but bundled agents obscure n-step’s isolated contribution. A minimalist n-step upgrade to QR-DQN is well-motivated, easy to implement, and—on low-stochasticity tasks like CartPole—expected to deliver consistent gains. This review frames your study within that context and highlights targeted follow-ups (n/γ/τ/quantiles, off-policy corrections, risk-aware evaluation).

---

## References

1. V. Mnih et al., “Human-level control through deep reinforcement learning,” *Nature*, 2015.  
2. R. Sutton and A. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.  
3. M. Bellemare, W. Dabney, and R. Munos, “A distributional perspective on reinforcement learning,” *ICML*, 2017.  
4. W. Dabney et al., “Distributional RL with quantile regression,” *AAAI*, 2018.  
5. W. Dabney et al., “Implicit quantile networks,” *ICML*, 2018.  
6. Z. Yang et al., “Fully parameterized quantile function,” *NeurIPS*, 2019.  
7. M. Hessel et al., “Rainbow: Combining improvements in deep RL,” *AAAI*, 2018.  
8. R. Munos et al., “Safe and efficient off-policy RL (Retrace),” *NeurIPS*, 2016.  
9. K. De Asis et al., “Multi-step RL: A unifying algorithm (Q(σ)),” *AAAI*, 2018.  
10. A. Harutyunyan et al., “Q(λ) with off-policy corrections,” *NeurIPS*, 2016.  
11. D. Precup, R. Sutton, and S. Singh, “Eligibility traces for off-policy policy evaluation,” *ICML*, 2000.  
12. Y. Chow et al., “Risk-sensitive and robust decision-making: a CVaR approach,” *NeurIPS*, 2015.  
13. H. van Hasselt et al., “Deep RL with double Q-learning,” *AAAI*, 2016.  
14. Z. Wang et al., “Dueling network architectures for deep RL,” *ICML*, 2016.  
15. T. Schaul et al., “Prioritized experience replay,” *ICLR*, 2016.  
16. M. Fortunato et al., “Noisy networks for exploration,” *ICLR*, 2018.  
17. G. Brockman et al., “OpenAI Gym,” *arXiv:1606.01540*, 2016.  
18. P. Henderson et al., “Deep RL that matters,” *AAAI*, 2018.  
