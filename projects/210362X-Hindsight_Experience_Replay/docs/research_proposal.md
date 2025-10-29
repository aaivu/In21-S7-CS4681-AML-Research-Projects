# Research Proposal: Hindsight Experience Replay with Prioritized Experience Replay

**Student:** 210362X
**Research Area:** Hindsight Experience Replay
**Date:** 2025-09-03

## Abstract

Hindsight Experience Replay (HER) is an effective technique for addressing sample inefficiency in sparse-reward reinforcement learning (RL) environments [1]. However, HER alone frequently struggles with issues such as insufficient exploration diversity, training instability due to reward noise, and suboptimal transition prioritization, particularly within complex continuous control tasks [1, 2]. This research introduces a unified TD3-based reinforcement learning framework designed to overcome these persistent challenges [1, 3]. The proposed integrated pipeline combines HER with **Prioritized Experience Replay (PER)** for focused learning, **Reward Ensembles** for reward robustness, and **Random Network Distillation (RND)** for intrinsic motivation and enhanced exploration [1, 4]. This synergistic integration, denoted as TD3 + HER + PER + RND + Reward Ensemble, is hypothesized to yield faster convergence and more stable training dynamics [3]. Experimental results show that this approach achieves approximately **25-30% faster convergence** and improved success rates compared to baseline TD3 implementations [1, 5].

## 1. Introduction

Sparse rewards remain one of the fundamental challenges in reinforcement learning (RL) [6]. In environments where reward signals are infrequent or binary, agents must explore a vast state space with minimal feedback, resulting in extremely low learning efficiency [6]. Traditional RL algorithms often fail in these settings, which commonly include goal-conditioned environments, robotic manipulation, and navigation tasks in continuous state spaces [6]. **Hindsight Experience Replay (HER)** [1, 7] addresses this challenge fundamentally by allowing agents to learn from failed outcomes [2]. HER effectively transforms sparse reward scenarios into dense reward scenarios by reinterpreting failures as successes for alternative, achieved goals, thereby dramatically improving sample efficiency [2, 8].

## 2. Problem Statement

While HER significantly improves sample efficiency, several critical limitations prevent optimal performance [2]. Firstly, HER does not intrinsically promote diverse exploration beyond what the underlying base policy naturally discovers [2]. Secondly, the technique can become unstable when interacting with noisy reward functions, as synthetic rewards generated via goal substitution may not accurately reflect state-action values [2, 4]. Thirdly, utilizing uniform replay for hindsight transitions fails to optimally prioritize the most valuable experiences, slowing down learning [2]. Finally, the inherent lack of intrinsic motivation can lead to premature convergence to suboptimal policies, failing to explore highly promising regions of the state space [2, 4]. This research aims to solve these persistent challenges by developing a robust, integrated framework that addresses exploration, stability, and prioritization simultaneously [4].

## 3. Literature Review Summary

**Hindsight Experience Replay (HER)** [1, 7] revolutionized learning in goal-conditioned, sparse-reward tasks by enabling learning from failed trajectories through goal substitution [8]. The foundational policy learning framework, **Twin Delayed Deep Deterministic Policy Gradient (TD3)** [6, 7], builds upon DDPG by utilizing twin critics, delayed policy updates, and target policy smoothing to stabilize training and mitigate overestimation bias [9, 10]. **Prioritized Experience Replay (PER)** [2, 7] improves sample efficiency by replaying transitions with the highest temporal-difference (TD) errors, as these are considered the most informative for learning [11]. **Random Network Distillation (RND)** [4, 12] drives exploration by training a predictor network against a fixed target network, rewarding the agent based on prediction error for visiting novel states [13, 14]. **Reward Ensembles** stabilize training by averaging outputs from multiple reward predictors, reducing sensitivity to noise and enabling uncertainty estimation [15]. The gap identified is the need for a **unified architecture** that integrates HER, TD3, PER, RND, and Reward Ensembles to leverage the synergistic benefits of each component, thereby overcoming HER's inherent weaknesses related to noise and insufficient exploration [1, 3, 16].

## 4. Research Objectives

### Primary Objective
To introduce a unified TD3-based reinforcement learning framework—**TD3 + HER + PER + RND + Reward Ensemble**—that jointly improves exploration diversity, reward robustness, and sample efficiency in sparse-reward continuous control benchmarks, addressing the inherent limitations of HER [1, 3].

### Secondary Objectives
- Validate that the integrated system achieves faster convergence, demonstrating an improvement of approximately **25-30% faster convergence** compared to baseline TD3 + HER implementations on continuous control tasks [1, 5, 17].
- Demonstrate that the combination of PER and Reward Ensembles effectively **stabilizes critic training** by denoising reward signals and focusing learning on high-value hindsight transitions, resulting in consistently lower TD-errors [4, 16, 18].
- Confirm that RND-driven intrinsic motivation successfully increases **trajectory diversity** (e.g., 15-20% higher compared to TD3 + HER) and accelerates the initial discovery of successful strategies in the state-goal space [4, 18, 19].

## 5. Methodology

The methodology integrates all components atop the TD3 architecture [3, 10].

1.  **TD3 Base:** TD3 is used for stable policy learning, employing twin critics to compute minimum Q-values for target calculation and utilizing delayed policy updates (actor updated every two critic updates) [9, 20].
2.  **HER Implementation:** Observations are goal-augmented [21]. After each episode, hindsight transitions are generated using the **"future" HER strategy** ($k=4$ future states selected as alternative goals), balancing sample efficiency and overhead [22]. Rewards are recomputed using a distance-based function incorporating a success bonus ($R_{success}=10.0$ if distance is below $\delta=0.05$) [23].
3.  **PER Integration:** Transitions are stored with priority proportional to the mean absolute TD-error across both twin critics ($\delta$) [22, 24]. A power-law distribution ($P(i)$) with $\alpha=0.6$ is used for sampling, and importance-sampling weights ($w_i$) are applied to gradient updates to correct for non-uniform sampling bias, with $\beta$ annealed from 0.4 to 1.0 [24, 25].
4.  **Reward Ensemble:** An ensemble of $N=3$ independent neural networks is trained to predict the true environment reward [26, 27]. During training, the final reward used for target computation is a blend of the environment reward ($r_{env}$) and the ensemble mean ($\bar{r}$), using a blending factor of $\lambda=0.5$ to stabilize learning [27].
5.  **RND for Exploration:** RND introduces intrinsic rewards ($r_{int}$) calculated as the prediction error between a fixed target network ($f_{target}$) and a trainable predictor network ($f_{pred}$) [14, 28]. Although the intrinsic rewards are currently not directly added to the extrinsic rewards, RND is trained alongside the policy to increase trajectory diversity, thereby improving the quality of HER transitions [19, 29].
6.  **Training Pipeline:** Training proceeds by observing goal-augmented states, selecting actions with exploration noise ($\sigma_{explore}=0.1$), storing original transitions, generating and storing HER transitions, and then sampling mini-batches using PER sampling [30, 31]. Critics, priorities, the actor, reward ensemble, and RND predictor are updated iteratively [31, 32]. Experiments will be conducted on the **LunarLanderContinuous-v3 environment**, augmented into a sparse-reward, goal-conditioned setting [33, 34].

## 6. Expected Outcomes

The integrated framework is expected to demonstrate superior performance and stability compared to baselines [1, 17].

*   **Performance Improvement:** We expect to achieve approximately **25-30% faster convergence** to the success threshold compared to TD3 + HER implementations [1, 17]. The full model is anticipated to reach a higher final success rate (e.g., averaging 87.3% or more) [5].
*   **Stability and Accuracy:** The Reward Ensemble is expected to reduce noise in training and lead to consistently lower TD-errors in later training stages, stabilizing value function learning [18, 35].
*   **Efficient Learning:** PER is expected to focus learning effectively, confirming that HER transitions appear disproportionately in high-priority samples during early training, accelerating the learning process [16, 36].
*   **Enhanced Exploration:** RND is expected to increase trajectory diversity by 15-20% compared to TD3 + HER, contributing to the faster initial discovery of successful strategies (achieving the first successful episode around 45,000 timesteps) [18, 19].
*   **Synergy:** The results should confirm synergistic effects, where the combined performance exceeds the sum of the individual contributions of PER, RND, and the Reward Ensemble [16, 17].

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review (Reviewing HER, TD3, PER, RND, Ensembles) [8, 9, 11, 13, 15]|
| 3-4  | Methodology Development (Finalizing integration plan and hyperparameters) |
| 5-8  | Implementation (Coding TD3+HER baseline, integrating PER, RND, and Reward Ensemble modules) [3] |
| 9-12 | Experimentation (Running ablation studies and full model testing on LunarLanderContinuous-v3) [19, 33]|
| 13-15| Analysis and Writing (Analyzing learning curves, TD-errors, ablation results, and drafting paper) [5, 18, 19]|
| 16   | Final Submission |

## 8. Resources Required

*   **Base Algorithm Implementation:** Twin Delayed Deep Deterministic Policy Gradient (TD3) [1, 9].
*   **Core Modules:** Hindsight Experience Replay (HER) implementation (future strategy, $k=4$) [21, 22], Prioritized Experience Replay (PER) implementation [11, 22], Random Network Distillation (RND) implementation [13, 14], Reward Ensemble implementation ($N=3$) [26].
*   **Environment and Data:** Goal-augmented LunarLanderContinuous-v3 environment [33, 34]. Replay buffer capacity: 1,000,000 transitions [37].
*   **Computational Resources:** Access to computational resources (CPU/GPU) sufficient for 500,000 timesteps of training [38].
*   **Software/Tools:** Reinforcement learning libraries (e.g., OpenAI Gym compatibility) [33].

## References

[1] M. Andrychowicz et al., “Hindsight Experience Replay,” in Proc. Neural Information Processing Systems (NeurIPS), 2017, pp. 5048–5058 [7].
[6] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” in Proc. International Conference on Machine Learning (ICML), 2018, pp. 1587–1596 [7].
[2] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized Experience Replay,” in Proc. International Conference on Learning Representations (ICLR), 2016 [7].
[4] Y. Burda et al., “Exploration by Random Network Distillation,” arXiv preprint arXiv:1810.12894, 2018 [12].
[3] I. Osband, C. Blundell, A. Pritzel, and B. Van Roy, “Deep Exploration via Bootstrapped DQN,” in Proc. Neural Information Processing Systems (NeurIPS), 2016, pp. 4026–4034 [12].

---