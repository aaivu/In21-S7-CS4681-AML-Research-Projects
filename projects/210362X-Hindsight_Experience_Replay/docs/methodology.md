# Methodology: Hindsight Experience Replay with PER

**Student:** 210362X
**Research Area:** Hindsight Experience Replay
**Date:** 2025-09-03

## 1. Overview

The proposed methodology introduces a unified reinforcement learning (RL) framework built upon the **Twin Delayed Deep Deterministic Policy Gradient (TD3)** architecture [1, 2]. This framework integrates **Hindsight Experience Replay (HER)**, **Prioritized Experience Replay (PER)**, **Reward Ensembles**, and **Random Network Distillation (RND)** to jointly address challenges related to exploration, reward robustness, and sample efficiency in sparse-reward continuous control tasks [1, 3, 4]. The core goal is to leverage the synergistic benefits of these components, where HER transforms sparse rewards, PER focuses learning on high-value transitions, RND enhances exploration diversity, and the Reward Ensemble stabilises training by mitigating reward noise [4, 5].

## 2. Research Design

The overall research approach involves an **integrated pipeline** design—TD3 + HER + PER + RND + Reward Ensemble [4]. This design is evaluated against various baselines to quantify the contribution and synergy of each added component [6]. Experiments are conducted in a **goal-conditioned, sparse-reward continuous control environment** (LunarLanderContinuous-v3 augmented for goals) [7]. The primary objective is to demonstrate **faster convergence** and **improved stability** compared to TD3 + HER alone [1]. The methodology involves detailed quantitative tracking of learning curves, success rates, and internal metrics such as Temporal-Difference (TD) errors and ensemble variance [8-10].

## 3. Data Collection

### 3.1 Data Sources
The primary data source is the **LunarLanderContinuous-v3** environment from the OpenAI Gym suite, augmented to function as a sparse-reward, goal-conditioned task [7, 11].

### 3.2 Data Description
The environment features an **8-dimensional observation space** (including position, velocity, angle, angular velocity, and leg contact information) and a **2-dimensional continuous action space** (controlling the main engine and side thrusters) [11]. The observation is augmented with a goal vector, $o = [s, g]$ [12].

The training budget is **500,000 timesteps** (approximately 5000 episodes) [13]. Transitions are stored in a replay buffer with a capacity of **1,000,000 transitions** [14].

### 3.3 Data Preprocessing
1.  **Goal Augmentation:** The raw environment state ($s$) is augmented with the desired goal ($g$) to form the goal-augmented observation ($o = [s, g]$) used by the policy [12].
2.  **HER Transition Generation:** At the end of each episode, hindsight transitions are generated using the **"future" HER strategy**, which selects $k=4$ future states from the same episode as alternative goals ($g'$) [15-17].
3.  **Reward Recomputation:** For the hindsight transitions, rewards ($r'_{her}$) are **recomputed** using a distance-based function: $r_{her} = -\|s'_p - g\|^2 + \mathbb{I}_{\|s'_p-g\|<\delta} \times R_{success}$ [15]. The success threshold $\delta$ is set to **0.05**, and the success reward $R_{success}$ is **10.0** [14].
4.  **Priority Calculation:** Both original and hindsight transitions are stored with a priority $p_i$ calculated based on the **mean absolute TD-error** across the twin critics [16, 18]. Initial priority $p_{max}$ is assigned to newly stored transitions [17].

## 4. Model Architecture

The core algorithm is **TD3**, which improves stability over DDPG using **twin critics** (taking the minimum Q-value for targets), **target policy smoothing** (adding clipped Gaussian noise $\epsilon$) [19, 20], and **delayed policy updates** (updating the actor every $d=2$ critic updates) [14, 20].

The integrated model architecture includes:

*   **TD3 Networks:** The actor network and twin critic networks each have **two hidden layers with 256 units** [7].
*   **Prioritized Experience Replay (PER):** Priority $p_i$ is calculated using the TD-error ($\delta_i$) raised to the power $\alpha=0.6$ [16, 21]. Importance-sampling weights $w_i$ are applied to gradient updates to correct for bias, with the correction parameter $\beta$ annealed linearly from **0.4 to 1.0** over training [21]. The mean absolute TD-error across both critics, $|\delta| = (|\delta_1| + |\delta_2|) / 2$, is used for priority updates [18].
*   **Reward Ensemble:** Consists of **$N=3$ independent neural networks** [14, 22]. Each network has two hidden layers of 256 units [22]. These networks predict the true environment reward ($r_{env}$) from unaugmented states [22]. The reward used for critic training ($r_{used}$) is a blend of the environment reward and the ensemble mean ($\bar{r}$), using a blending factor $\lambda=0.5$ [14, 23].
*   **Random Network Distillation (RND):** Includes a fixed target network ($f_{target}$) and a trainable predictor network ($f_{pred}$). Both RND networks have **two hidden layers of 128 units** and output a **128-dimensional embedding** [14, 24]. The prediction error $r_{int} = \|f_{pred}(s) - f_{target}(s)\|^2$ is normalized [24, 25]. In the current implementation, RND influences exploration indirectly by increasing trajectory diversity, rather than directly mixing $r_{int}$ with extrinsic rewards [25].

## 5. Experimental Setup

### 5.1 Evaluation Metrics
Evaluation metrics are logged throughout training, including [6, 9, 13]:

*   **Success Rate:** Measured as the percentage of successful episodes in evaluation runs, typically calculated over a moving average [13, 26].
*   **Convergence Speed:** Timesteps required to reach a specific success threshold (e.g., 80% success rate) [13, 27].
*   **Mean Absolute TD-Error:** Used to assess the stability and accuracy of value function learning [9].
*   **Initial Discovery Time:** Average timesteps required to achieve the first successful episode [9].
*   **Trajectory Diversity:** Measured by the number of unique state-action pairs in the replay buffer, used to confirm RND's impact [6].

### 5.2 Baseline Models
The proposed full model (TD3 + HER + PER + RND + Reward Ensemble) will be compared against [26, 27]:

*   **TD3 + HER:** The standard baseline implementation leveraging TD3 stability and HER sample efficiency.
*   **Baseline TD3:** The foundational algorithm without HER or other enhancements (which typically fails to achieve consistent success in sparse-reward settings) [13].
*   **Ablation Studies:** Comparisons against models where individual components (PER, Reward Ensemble, RND) are removed from the full framework [6].

### 5.3 Hardware/Software Requirements
*   **Software/Tools:** Reinforcement learning libraries compatible with the **OpenAI Gym suite** for the LunarLanderContinuous-v3 environment [11]. Code is implemented and available on GitHub [13, 28].
*   **Computational Resources:** Access to sufficient CPU/GPU resources to run training for **500,000 timesteps** [14].

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Literature Review & Methodology Development | 2 weeks | Finalized integration plan and hyperparameters [14] |
| Phase 2 | Model implementation (TD3+HER baseline, integrating PER, RND, and Ensemble) | 3 weeks | Working model and integrated RL framework [8] |
| Phase 3 | Experiments (Ablation studies and full model testing) | 2 weeks | Raw results (Learning curves, TD-errors, Success rates) [9, 13] |
| Phase 4 | Analysis and Writing (Interpreting results, confirming synergy) | 1 week | Final methodology and results report draft |

## 7. Risk Analysis

| Potential Risk | Mitigation Strategy |
| :--- | :--- |
| **Biased Learned Rewards:** The Reward Ensemble overfits or learns incorrect reward patterns early in training, misleading the critic [29]. | Mitigated by **blending** the ensemble mean output with the true environment reward ($\lambda=0.5$) and using ensemble diversity [23, 29]. |
| **Intrinsic Reward Scaling:** $r_{int}$ dominates extrinsic rewards, causing the agent to prioritise novelty over task completion [30]. | Mitigated by **normalizing** $r_{int}$ using running statistics [25]. The current implementation avoids direct mixing, reducing this risk, though future work might need careful adaptive tuning of $\lambda_{int}$ [25, 30, 31]. |
| **Suboptimal HER Goal Selection:** Uniform goal selection in the "future" strategy may not prioritize the most informative hindsight experiences [30]. | Addressed by combining HER with **PER**, which dynamically prioritizes high TD-error hindsight transitions, accelerating learning in informative regions [18, 32]. Future work may explore priority-based goal selection [30, 33]. |
| **Generalization Limits:** Results validated primarily on navigation-like tasks [32]. | Future work is required to conduct comprehensive ablation studies across diverse environments (e.g., FetchReach, AntMaze) to validate generalization [33]. |

## 8. Expected Outcomes

The integrated framework is expected to deliver **superior performance** and **increased stability** [4].

*   **Faster Convergence:** The full model is expected to achieve approximately **25-30% faster convergence** to the success threshold compared to TD3 + HER implementations [1, 27]. The full model should achieve a success rate above 80% after roughly **300,000 timesteps** [13].
*   **High Final Success Rate:** The model is anticipated to achieve a higher final success rate, averaging **87.3%± 4.2%** over evaluation periods [26].
*   **Stability:** The Reward Ensemble is expected to stabilize critic training, evidenced by **consistently lower TD-errors** and reduced variance in later training stages [9].
*   **Efficient Exploration:** RND is expected to increase trajectory diversity by **15-20%** compared to TD3 + HER, contributing to the faster initial discovery of successful strategies (e.g., first successful episode around 45,000 timesteps) [6, 9].
*   **Synergistic Contribution:** The results will confirm that the combined performance **exceeds the sum of individual components**, demonstrating synergistic effects, particularly between PER, HER, and the Reward Ensemble [27, 32].

---

**Note:** Update this document as your methodology evolves during implementation.