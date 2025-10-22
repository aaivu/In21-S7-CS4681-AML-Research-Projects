# Research Proposal: Financial AI:Algorithmic Trading

**Student:** 210620M
**Research Area:** Financial AI:Algorithmic Trading
**Date:** 2025-09-01

## Abstract

This report presents the ongoing progress in enhancing the FinRL deep reinforcement
learning framework for automated stock trading. The work focuses on understanding
domain-specific financial concepts, reproducing baseline models, and designing methodologies
to improve trading performance. Key activities include studying portfolio management,
backtesting, market frictions, and risk metrics, as well as implementing DRL algorithms such as
DQN, PPO, DDPG, TD3, and SAC. Proposed enhancements include systematic
hyperparameter tuning, multi-objective loss function design, and ensemble strategies to improve
robustness and risk-adjusted returns. Preliminary findings demonstrate that these improvements
can guide agents toward more adaptive, stable, and profitable trading strategies, laying the
groundwork for further research and real-world deployment.

## 1. Introduction

The rise of artificial intelligence in finance has transformed traditional trading approaches, enabling automated strategies that can process vast market data and execute trades with minimal human intervention. Algorithmic trading leverages computational models to identify profitable opportunities, manage risk, and optimize portfolio performance in real time. Deep Reinforcement Learning (DRL) has emerged as a promising approach for adaptive trading, allowing agents to learn optimal policies directly from market interactions. Despite its potential, applying DRL in finance remains challenging due to market volatility, noisy data, and the need for risk-sensitive decision-making. This research explores the integration of DRL algorithms with robust portfolio management techniques to develop trading agents capable of generating stable, risk-adjusted returns. The significance lies in advancing algorithmic trading strategies that are more adaptive, reliable, and aligned with real-world financial constraints.

## 2. Problem Statement

Current DRL-based trading frameworks often struggle with market volatility, overfitting, and inconsistent performance across different financial instruments. Baseline models in frameworks like FinRL provide a foundation but lack systematic hyperparameter optimization, ensemble strategies, and risk-sensitive objective functions. The research problem is to enhance DRL trading agents to improve their adaptability, robustness, and profitability under real-world market conditions, while ensuring effective risk management. Specifically, this study aims to address the gap between theoretical DRL performance and practical applicability in dynamic financial markets.

## 3. Literature Review Summary

Recent studies have demonstrated the application of DRL algorithms DQN, PPO, DDPG, TD3, and SAC in portfolio optimization and trading simulations. These works highlight the benefits of reinforcement learning in adaptive decision-making but also reveal limitations such as high variance in returns, sensitivity to hyperparameters, and poor generalization in volatile markets. Existing literature emphasizes single-agent learning and simplified environments, leaving gaps in ensemble strategies, multi-objective optimization, and integration of market frictions. This research seeks to address these gaps by designing enhanced DRL frameworks that consider risk-adjusted returns, transaction costs, and adaptive learning techniques.

## 4. Research Objectives

### Primary Objective
Develop a robust, adaptive DRL-based trading framework capable of generating consistent, risk-adjusted returns across different market conditions.

### Secondary Objectives
-Implement systematic hyperparameter tuning for multiple DRL algorithms to optimize performance.

-Design multi-objective loss functions that incorporate profitability, volatility, and risk metrics.

-Explore ensemble strategies to improve agent robustness and reduce strategy overfitting.

-Evaluate the enhanced framework using real-world market data and backtesting environments.

## 5. Methodology

-Environment Setup: Utilize FinRL as the base framework for training trading agents with historical stock data. Include realistic market constraints such as transaction costs and slippage.

-Algorithm Implementation: Implement DRL algorithms (A2C, PPO, DDPG) for asset management.

-Hyperparameter Optimization: Apply Optuna or similar frameworks for automated hyperparameter tuning to improve convergence and stability.

-Multi-Objective Loss Design: Incorporate objectives such as risk-adjusted return, drawdown minimization, and Sharpe ratio into reward functions.

-Ensemble Learning: Combine predictions from multiple DRL agents to enhance robustness and reduce variance in trading performance.


## 6. Expected Outcomes

-A DRL-based trading framework with optimized hyperparameters and ensemble strategies.

-Trading agents capable of adaptive decision-making, outperforming baseline models in backtesting.

-Insights into the effect of multi-objective optimization on risk-adjusted returns.

-A validated methodology for applying DRL in realistic financial environments, bridging the gap between theoretical models and practical trading.

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development |
| 5-8  | Implementation |
| 9-12 | Experimentation |
| 13-15| Analysis and Writing |
| 16   | Final Submission |

## 8. Resources Required

-Datasets: Historical stock prices, ETFs, market indices, and trading volumes.

-Tools & Libraries: Python, FinRL framework, TensorFlow/PyTorch, Optuna, Pandas, NumPy, Matplotlib.

-Computing Resources: GPU-enabled machines or cloud computing instances for training DRL models.

-Software: Git for version control, Jupyter Notebook for prototyping, and visualization tools for analysis.

## References

[Add references in academic format]

---

**Submission Instructions:**
1. Complete all sections above
2. Commit your changes to the repository
3. Create an issue with the label "milestone" and "research-proposal"
4. Tag your supervisors in the issue for review