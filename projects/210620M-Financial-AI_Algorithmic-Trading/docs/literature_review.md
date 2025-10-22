# Literature Review: Financial AI:Algorithmic Trading

[cite_start]**Student:** 210620M 
[cite_start]**Research Area:** Financial AI:Algorithmic Trading (Enhancing FinRL DRL Framework) [cite: 2, 6]
**Date:** 2025-09-01

## Abstract

This literature review explores recent developments in applying Artificial Intelligence, particularly Deep Reinforcement Learning (DRL), to algorithmic trading. [cite_start]The review covers key areas including DRL-based trading strategies [cite: 32][cite_start], portfolio management [cite: 16][cite_start], risk-aware reward functions [cite: 30][cite_start], hyperparameter tuning [cite: 9, 101][cite_start], and ensemble techniques[cite: 9, 108]. [cite_start]It highlights the challenges of overfitting, sensitivity to market volatility, and limited real-world robustness[cite: 20, 59, 65]. [cite_start]Emerging trends, such as multi-objective optimization [cite: 9, 104][cite_start], hybrid agents, and risk-adjusted trading strategies[cite: 10], are identified as promising directions. [cite_start]Overall, the review provides a foundation for designing more adaptive and resilient trading agents[cite: 111].


## 1. Introduction

[cite_start]Algorithmic trading has increasingly leveraged AI to automate financial decision-making, moving beyond traditional rule-based strategies[cite: 6]. [cite_start]DRL enables agents to learn optimal trading policies directly from market interactions, adapting to dynamic environments[cite: 32, 83]. [cite_start]This review examines the literature on DRL in trading, portfolio optimization [cite: 16][cite_start], risk-sensitive reward design [cite: 30, 35][cite_start], and techniques to improve agent robustness[cite: 10, 111]. [cite_start]The scope includes studies from 2018 onward, focusing on both methodological developments and practical implementation challenges in real-world financial markets[cite: 27, 74].


## 2. Search Methodology

### Search Terms Used
- [cite_start]Deep Reinforcement Learning in Trading [cite: 6, 31]
- [cite_start]FinRL Framework [cite: 6, 33]
- [cite_start]Portfolio Management DRL [cite: 16]
- [cite_start]Risk Metrics DRL [cite: 29, 30]
- [cite_start]DRL Algorithms (DQN, PPO, DDPG, TD3, SAC) 

### Databases Searched
- [ ] IEEE Xplore
- [ ] ACM Digital Library
- [ ] Google Scholar
- [ ] ArXiv
- [x] Other: **Proc. 34th Conf. [cite_start]Neural Information Processing Systems (NeurIPS) Deep Reinforcement Learning Workshop** [cite: 144, 145]

### Time Period
[cite_start]2015 - 2020 (Seminal Papers) [cite: 147, 151] [cite_start]and 2017 - 2018 (Key DRL Developments) [cite: 148, 153, 155]

## 3. Key Areas of Research

### [cite_start]3.1 Deep Reinforcement Learning Algorithms in Trading [cite: 37]
[cite_start]Research focuses on classifying DRL models for finance into value-based (DQN), policy-based (PPO), and actor-critic methods (DDPG, TD3, SAC, A2C)[cite: 38]. [cite_start]These algorithms are applied to a wide range of trading tasks, from discrete buy/sell decisions to continuous portfolio allocation[cite: 39]. [cite_start]PPO is widely used for its balance of stability and adaptability, while actor-critic methods like TD3 and SAC are favored for continuous control tasks like multi-asset portfolio management, due to their robustness enhancements over DDPG[cite: 49, 62, 65].

**Key Papers:**
- [cite_start]**Mnih et al., 2015 [cite: 146, 147][cite_start]** - Introduced DQN, which uses deep neural networks to approximate the action-value function, making it effective for discrete actions[cite: 42, 43].
- [cite_start]**Schulman et al., 2017 [cite: 148][cite_start]** - Presented PPO, a policy-gradient method that stabilizes training by constraining updates, a crucial feature for volatile financial markets[cite: 47, 48].
- [cite_start]**Fujimoto et al., 2018 [cite: 152, 153][cite_start]** - Developed TD3 to mitigate overestimation errors in DDPG, making the actor-critic approach more robust for multi-asset trading and portfolio management[cite: 60, 62].

### [cite_start]3.2 Domain-Specific Concepts and Evaluation [cite: 12]
This area covers the foundational concepts necessary for designing and evaluating DRL trading agents. [cite_start]Core concepts include the portfolio (the collection of assets managed by the agent) [cite: 14, 15] [cite_start]and market friction (costs like transaction costs and slippage) which significantly impact strategy viability[cite: 26, 27]. [cite_start]Evaluation relies on backtesting with historical data to prevent overfitting [cite: 18, 20] [cite_start]and performance assessment against a benchmark (e.g., S\&P 500)[cite: 22, 23]. [cite_start]Risk metrics like Maximum Drawdown and Sharpe Ratio are essential for ensuring strategies are risk-aware[cite: 29, 30].

**Key Papers:**
- [cite_start]**Liu et al., 2020 (FinRL) [cite: 144, 145][cite_start]** - Frames a practical stack for DRL in finance, summarizing common metrics, baselines, and frictions for realistic evaluation, and standardizing definitions via the FinRL library[cite: 33, 36].
- [Author, Year] - [Brief summary of contribution]

## 4. Research Gaps and Opportunities

The project's proposed enhancements address the following gaps in baseline DRL trading models:

### [cite_start]Gap 1: Sub-optimal Performance Due to Fixed Hyperparameters [cite: 101, 103]
[cite_start]**Why it matters:** DRL agent performance is highly sensitive to hyperparameters (e.g., learning rate, discount factor), and relying on defaults often leads to sub-optimal or unstable performance, particularly in high-dimensional and non-stationary financial environments[cite: 50, 101].
[cite_start]**How your project addresses it:** **Systematic Hyperparameter Tuning** will be implemented using automated search tools to find the best configuration for the DRL agents, ensuring they operate at their optimal performance levels[cite: 101, 102].

### [cite_start]Gap 2: Maximizing Return without Sufficient Risk Awareness [cite: 105, 107]
[cite_start]**Why it matters:** Standard DRL reward functions primarily aim to maximize cumulative return, which can result in overly aggressive, unstable, and high-risk strategies, especially in volatile markets, as risk metrics are often only used for post-evaluation[cite: 105, 30].
[cite_start]**How your project addresses it:** **Loss Function Enhancement** will involve designing a custom multi-objective loss function that balances maximizing return and **minimizing risk**, guiding the agent to learn more risk-preventive and stable trading strategies[cite: 104, 106, 107].

### [cite_start]Gap 3: Lack of Robustness and Resilience in Single-Model Strategies [cite: 111]
[cite_start]**Why it matters:** Single DRL models are often prone to instability, sensitive to market shifts, and may struggle with overfitting, making them brittle when deployed to unseen data or different market regimes[cite: 20, 59, 111].
**How your project addresses it:** **Ensemble Methods** will be developed by training multiple diverse DRL agents (different algorithms/settings) and combining their signals using a meta-agent or voting mechanism. [cite_start]This is expected to produce a more robust and resilient strategy than any single model[cite: 108, 109, 110].

## 5. Theoretical Framework

[cite_start]The theoretical foundation for DRL in financial trading is the casting of the task as a **time-driven Markov Decision Process (MDP)**[cite: 34].
* **State:** Includes cash, positions, prices, volumes, and technical factors[cite: 34].
* [cite_start]**Actions:** Are discrete (e.g., buy/sell/hold) or continuous (e.g., trade sizes/portfolio weights)[cite: 35].
* [cite_start]**Rewards:** Range from portfolio value change to risk-adjusted objectives[cite: 35].

[cite_start]The practical implementation is built upon the **FinRL Three-Layer Architecture**[cite: 67]:
1.  [cite_start]**Market Environments (FinRL-Meta):** Based on the OpenAI Gym interface, providing realistic, customizable, and friction-inclusive simulation[cite: 70, 72, 74, 75].
2.  [cite_start]**DRL Agents:** Houses the core intelligence, supporting a diversity of modular DRL algorithms (DQN, PPO, SAC, etc.)[cite: 78, 80, 81].
3.  [cite_start]**Applications:** Focuses on real-world utility, covering specific trading strategies, backtesting, and performance metrics[cite: 85, 87, 88, 90].

## 6. Methodology Insights

[cite_start]The literature review shows that DRL in finance commonly utilizes model-free algorithms for sequential decision-making[cite: 32].

**Commonly Used Methodologies:**
* [cite_start]**DRL Algorithms:** PPO is frequently employed for its stability, while TD3 and SAC are preferred for complex, continuous portfolio allocation due to their enhanced robustness over DDPG[cite: 49, 62, 65].
* **Evaluation:** A strict **train-validation-test split** with an out-of-sample test set is critical for preventing data leakage and ensuring realistic performance assessment[cite: 36, 113, 115].
* [cite_start]**Performance Metrics:** Essential metrics are **Cumulative Return** (profitability), **Sharpe Ratio** (risk-adjusted return), and **Maximum Drawdown** (maximum potential loss)[cite: 117, 118, 119, 120].

**Most Promising for this Work:**
The project will focus on advanced methodologies to enhance the baseline:
* [cite_start]**Systematic Hyperparameter Tuning**.
* **Multi-Objective Loss Function Design** for explicit risk control[cite: 104, 106].
* [cite_start]**Ensemble Strategy** development to combine the strengths of multiple DRL agents (e.g., PPO, TD3, SAC) for superior robustness[cite: 108, 109].

## 7. Conclusion

[cite_start]The review confirms that DRL, particularly through the FinRL framework, offers a powerful approach for automated algorithmic trading[cite: 6]. [cite_start]Key DRL algorithms like PPO, TD3, and SAC are foundational for modern trading agents[cite: 80]. [cite_start]However, there is a clear opportunity to enhance the robustness and risk-awareness of these models[cite: 9]. [cite_start]The proposed methodology—combining systematic hyperparameter tuning, a risk-aware multi-objective loss function, and an ensemble strategy—directly addresses the limitations of single-model, return-maximizing agents[cite: 100]. [cite_start]This integrated approach lays the groundwork for designing adaptive, stable, and profitable trading strategies for real-world deployment[cite: 10, 111].

## References

1. X.-Y. Liu, H. Yang, Q. Chen, R. Zhang, L. Yang, B. Xiao, and C. D. Wang, "FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance," Proc. 34th Conf. [cite_start]Neural Information Processing Systems (NeurIPS) Deep Reinforcement Learning Workshop, Vancouver, Canada, Dec. 2020. [cite: 144, 145]
2. V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., "Human-level control through deep reinforcement learning," Nature, vol. 518, no. [cite_start]7540, pp. 529-533, 2015. [cite: 146, 147]
3. [cite_start]J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017. [cite: 148]
4. V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, "Asynchronous methods for deep reinforcement learning," in Proc. 33rd Int. [cite_start]Conf. on Machine Learning (ICML), 2016, pp. 1928-1937. [cite: 149, 150]
5. [cite_start]T. P. Lillicrap et al., "Continuous control with deep reinforcement learning," arXiv preprint arXiv:1509.02971, 2015. [cite: 151]
6. S. Fujimoto, H. van Hoof, and D. Meger, "Addressing function approximation error in actor-critic methods," Proceedings of the 35th International Conference on Machine Learning (ICML), vol. [cite_start]80, pp. 1587-1596, 2018. [cite: 152, 153]
7. T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," Proceedings of the 35th International Conference on Machine Learning (ICML), vol. [cite_start]80, pp. 1861-1870, 2018. [cite: 154, 155]