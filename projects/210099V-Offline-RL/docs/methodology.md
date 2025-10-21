# Methodology: QR-DQN

**Student:** 210099V  
**Research Area:** Offline RL  
**Date:** 2025-09-01  

---

## 1. Overview
This methodology specifies how we will study offline value-based reinforcement learning using distributional Q-learning.  
The core idea is to train agents **without environment interaction** from fixed datasets (logged trajectories), focusing on:

- A distributional critic (QR-DQN) adapted for offline training  
- A simple, effective conservatism mechanism (CQL-style penalty + BC anchor) to counter extrapolation error  
- An ablation on n-step targets (1-step vs. n=3/5) in the offline regime  
- Robust policy evaluation via simulator rollouts (no data collection) and off-policy estimators (WIS/DR/FQE)

---

## 2. Research Design

**Approach:** Experimental, with controlled ablations.  
We compare an **Offline QR-DQN (distributional critic)** against standard offline baselines (BC, CQL-DQN, BCQ/IQL variants) on discrete-action benchmarks (MinAtar Breakout/Pong; selected RLU Atari shards).  

**Hypotheses:**  
- Short n-step returns (n≈3–5) improve offline QR-DQN stability/learning vs. 1-step when paired with mild conservatism.  
- A lightweight conservatism term + BC anchor suffices on low-stochasticity tasks to close most of the return gap to heavier baselines.

---

## 3. Data Collection

### 3.1 Data Sources
- **MinAtar Offline (Breakout, Pong)** – logged transitions from pre-collected policies (random → medium → expert mixtures).  
- **RL Unplugged (Atari) shards (optional)** – e.g., Pong/Breakout runs for scalability checks.  
- **(If needed)** Custom CartPole offline datasets collected once from scripted ε-greedy policies (no training-time interaction).

### 3.2 Data Description
- **Observation spaces:**  
  - MinAtar: low-dim grid (e.g., 10×10×C)  
  - Atari: 84×84 grayscale with frame-stacking (×4) if used  
- **Action spaces:** discrete (env-dependent)  
- **Dataset composition:** multiple trajectories with episode boundaries, rewards, and terminals.  
  Datasets stratified into train/valid/test by trajectory (no leakage).  
- **Scale:** ~10⁵–10⁶ transitions (MinAtar) and 10⁶–10⁸ (Atari shards); sub-sampling used to meet compute budgets.

### 3.3 Data Preprocessing
- Validate episode boundaries; ensure terminal flags and time-limits are consistent.  
- Reward normalization/clipping (env-specific; keep an ablation).  
- Observation transforms: MinAtar (identity); Atari (resize, grayscale, frame-stack if used).  
- Action/obs checks: encode discrete actions; verify one-hot vs. index consistency.  
- **n-step targets precompute:** for n∈{1,3,5}, compute R_t^{(n)}, terminal masks, and (s_{t+n}) within each trajectory (stop at terminals).  
- Splits: 80/10/10 by full episodes.  
- Store as sharded files (e.g., NPZ/Parquet) with indices for fast sampling.

---

## 4. Model Architecture

- **Backbone:** MLP (MinAtar) or shallow CNN (Atari), outputting |A|×N quantile values; N=51 mid-point quantiles.  
- **Policy:** greedy over the mean of quantiles (risk-neutral eval).  
- **Targeting:** Double Q bootstrapping with a target network (hard update every τ steps).  

**Loss (offline):**
- Quantile pinball/Huber loss for distributional TD.  
- **Conservatism (CQL-style):** encourage lower Q on unseen actions via log-sum-exp over actions vs. behavior-action Q; weight α (tune).  
- **BC anchor (discrete):** cross-entropy between policy logits and behavior action (weight β) to keep policy near data support.  

**Total loss:**
\[ \mathcal{L} = \mathcal{L}_{\text{QR}} + \alpha\mathcal{L}_{\text{CQL}} + \beta\mathcal{L}_{\text{BC}} \]

**Ablations:** n∈{1,3,5}; α∈{0.0,0.1,0.2}; β∈{0.0,0.1}.  
**Optimizer:** Adam (lr=5e-4), grad-norm clip 10.  
**Replay:** uniform over logged data (no new env steps).

---

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- Average return over 100 deterministic episodes (MinAtar) and 50–100 (Atari shards).  
- Normalized score (per-env, if available).  
- Stability metrics: inter-seed mean ± SD (≥5 seeds).  
- Offline estimators: FQE on a held-out set; WIS (weighted importance sampling) and Doubly-Robust where behavior policies are available.  
- Efficiency: wall-clock time, updates/sec.

### 5.2 Baseline Models
- **QR-DQN**.  
- **Our model:** QR-DQN + n-step.

### 5.3 Hardware/Software Requirements
- **Hardware:** 1× NVIDIA GPU (≥8 GB VRAM) for MinAtar; ≥16 GB VRAM for Atari shards; CPU RAM ≥32 GB; SSD ≥50 GB (Atari).  
- **Software:** Python 3.10+, PyTorch 2.x (+CUDA 12.x), NumPy, Gymnasium/MinAtar, (optional) TorchRL, Matplotlib.  
  Reproducibility ensured via fixed seeds and deterministic flags.

---

## 6. Implementation Plan

| **Phase** | **Tasks** | **Duration** | **Deliverables** |
|------------|------------|--------------|------------------|
| **Phase 1** | Data ingestion, integrity checks; episode-wise splits; preprocessing pipelines; n-step cache builder | 2 weeks | Clean sharded datasets + preprocessing scripts |
| **Phase 2** | Implement Offline QR-DQN (distributional head, double Q, target net); add CQL loss + BC anchor; config system | 3 weeks | Working training loop + configs (n, α, β) |
| **Phase 3** | Run baselines (BC, naïve DQN/QR-DQN, CQL-DQN); run our model across seeds and n∈{1,3,5}; log metrics/curves | 2 weeks | Results (CSV/plots), model checkpoints |
| **Phase 4** | Analysis: aggregate seeds; significance checks; ablation tables; write-up and figure polish | 1 week | Final report section + artifact bundle |

---

## 7. Risk Analysis

| **Risk** | **Impact** | **Mitigation** |
|-----------|-------------|----------------|
| Extrapolation error (Q overestimation on OOD actions) | Poor returns / instability | Use CQL penalty (tune α); BC anchor (β); monitor action-gap histograms |
| Distribution shift (train/valid mismatch) | Overfitting | Episode-level splits; early-stop on valid; weight decay; data augmentation where safe |
| Dataset quality (low coverage) | Upper-bounded performance | Choose “medium/medium-expert” splits; report coverage metrics (unique (s,a), action entropy) |
| Compute limits | Incomplete ablations | Prioritize MinAtar; sub-sample Atari; log learning curves early to prune configs |
| Metric bias (optimistic rollouts) | Misleading conclusions | Add FQE/WIS/DR checks on held-out data; report inter-seed SD and confidence intervals |
| Reproducibility drift | Non-replicable results | Seed everything; save configs & checkpoints; publish scripts + exact versions |

---

## 8. Expected Outcomes
- **Primary:** Demonstrate that n-step (n≈3) within Offline QR-DQN plus light conservatism yields consistent gains over 1-step and approaches CQL-DQN on MinAtar.  
- **Secondary:** Sensitivity analysis clarifying interactions between n, α (CQL weight), β (BC anchor), and target-update cadence.  
- **Artifacts:** Clean offline datasets/splits, training/eval scripts, configuration files, and plots suitable for paper inclusion.  
- **Broader impact:** A minimal, reproducible recipe for offline distributional Q-learning in discrete-action domains that can extend to risk-aware evaluation (e.g., CVaR over learned quantiles).
