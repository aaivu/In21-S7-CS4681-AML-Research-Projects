# Literature Review: Reasoning AI:Chain-of-Thought

**Student:** 210386A
**Research Area:** Reasoning AI:Chain-of-Thought
**Date:** 2025-09-01

## Abstract

This literature review explores recent advancements in Reasoning AI, focusing on Chain-of-Thought (CoT) reasoning and self-correction mechanisms in Large Language Models (LLMs). While LLMs exhibit strong capabilities in natural language understanding and generation, their ability to reason coherently, self-correct, and refine outputs autonomously remains a challenge. The review synthesizes developments across four major research directions: prompting-based intrinsic self-correction, reinforcement learning (RL) fine-tuning, generation-time correction, and multi-agent frameworks. Collectively, these studies reveal how fine-tuning and real-time correction strategies improve reasoning reliability and robustness, highlighting key gaps in autonomous error identification, interpretability, and multi-step reasoning stability.

## 1. Introduction

Reasoning is a core component of artificial intelligence, enabling models to perform complex, multi-step cognitive tasks such as problem-solving, mathematical derivation, and logical inference. Chain-of-Thought (CoT) prompting—explicitly guiding models through intermediate reasoning steps—has significantly enhanced LLMs’ interpretability and problem-solving accuracy. However, despite these advances, models still exhibit inconsistencies, hallucinations, and reasoning collapse over long sequences.

Recent research emphasizes enabling self-correction—allowing models to detect and fix their own reasoning errors through feedback, reflection, or multi-agent collaboration. This literature review examines key directions for improving reasoning in LLMs through prompting, RL-based fine-tuning, generation-time correction, and multi-agent reasoning systems.
## 2. Search Methodology

### Search Terms Used
""Chain-of-Thought Reasoning,” “LLM Self-Correction,” “Reinforcement Learning in LLMs,” “Reasoning with LLMs,” “Multi-Agent Reasoning,” “Feedback-Guided Decoding,” “Self-Verification in AI.”

Variants: “Self-Reflective LLMs,” “RLHF Fine-tuning,” “Iterative Reasoning,” “Self-Debugging Models.

### Databases Searched
-  IEEE Xplore
-  ACM Digital Library
-  Google Scholar
-  ArXiv


### Time Period
2018–2025, with a focus on post-2022 developments in reasoning and self-correction for LLMs.

## 3. Key Areas of Research

### 3.1 [Prompting for Intrinsic Self-Correction]
Early work proposed that prompting LLMs to critique or revise their own answers could enhance reasoning. However, recent evidence shows that naïve self-correction often leads to degraded performance (Huang et al., 2023; Qu et al., 2024; Tyen et al., 2024; Zheng et al., 2024). Failures arise from incorrect assumptions about available feedback and overreliance on initial weak prompts.

In tasks like code repair, even high-performing models struggle when feedback is partial or ambiguous (Olausson et al., 2023). These studies demonstrate that prompting alone is insufficient to achieve robust self-correction and must be complemented by structured feedback or external evaluation mechanisms.

**Key Papers:**

Kim et al. (2023) — Demonstrated preliminary success in prompt-based self-reflection.

Huang et al. (2023) — Showed performance degradation in naive correction prompting.

Olausson et al. (2023) — Evaluated LLM code self-repair limitations with partial feedback.

### 3.2 [Reinforcement Learning for LLM Fine-Tuning]
Reinforcement Learning from Human Feedback (RLHF) has emerged as a dominant paradigm for aligning LLMs with human intent (Christiano et al., 2017; Ouyang et al., 2022). Fine-tuning through reward-based optimization enhances reasoning consistency and adherence to task objectives.

Recent variants—such as ReST (Gulcehre et al., 2023), Reward-Ranked Fine-tuning (Dong et al., 2023), and AlpacaFarm (Dubois et al., 2023)—show that optimizing for high-reward responses via cross-entropy loss can achieve results comparable to full PPO-based RLHF.

Newer works, including SCoRe (Kumar et al., 2024) and GLoRe (Havrilla et al., 2024b), emphasize multi-turn reinforcement learning, where models iteratively refine outputs through feedback loops. These frameworks improve long-horizon reasoning and reduce hallucination by incorporating self-verification and reward shaping.

**Key Papers:**

Ouyang et al. (2022) — Introduced InstructGPT with RLHF alignment.

Gulcehre et al. (2023) — Proposed Reinforced Self-Training (ReST).

Kumar et al. (2024) — Introduced SCoRe: multi-turn RL for iterative self-correction.

Havrilla et al. (2024b) — Developed GLoRe for global/local reasoning refinement.

### 3.3 [Multi-Agent and Iterative Self-Reflection]
Recent trends involve multi-agent frameworks where multiple LLM instances collaborate to enhance reasoning robustness.

Mixture-of-Agents (MoA) (Wang et al., 2024) — Employs multiple agents generating and refining diverse responses through layered reasoning.

Agent-R (Yuan et al., 2025) — Uses Monte Carlo Tree Search for self-correction trajectories.

CORY (Ma et al., 2024) — Implements cooperative multi-agent RL where agents exchange roles (pioneer and observer) to promote mutual learning and prevent policy collapse.

These systems emulate human group reasoning, fostering diversity and iterative refinement—crucial for complex reasoning under uncertainty.

**Key Papers:**

Wang et al. (2024) — Mixture-of-Agents (MoA) for collaborative reasoning.

Ma et al. (2024) — Sequential cooperative multi-agent reinforcement learning.

Yuan et al. (2025) — Agent-R with MCTS for iterative correction.
## 4. Research Gaps and Opportunities

[Identify gaps in current research that your project could address]

Gap 1: Limited Autonomy in Self-Correction

Why it matters: Current models rely heavily on external supervision or feedback. True autonomous self-correction would allow continuous learning from errors without retraining.
How your project addresses it: Investigate unsupervised or semi-supervised self-reflection loops for intrinsic correction during reasoning.

Gap 2: Inadequate Handling of Long-Horizon Reasoning

Why it matters: Many LLMs lose coherence in multi-step reasoning or long-context tasks.
How your project addresses it: Explore hierarchical reasoning architectures integrating CoT with memory-augmented agents.

Gap 3: Evaluation Challenges

Why it matters: Quantifying reasoning accuracy and self-correction reliability lacks standardized benchmarks.
How your project addresses it: Develop evaluation metrics emphasizing interpretability, error localization, and reasoning trace fidelity.

## 5. Theoretical Framework

This study is grounded in Cognitive Reinforcement Theory and Meta-Reasoning Frameworks, positing that reasoning AI can emulate human-like self-awareness by continuously evaluating and refining its own thought chains. The Chain-of-Thought paradigm functions as a cognitive trace, while RL-based refinement parallels metacognitive regulation in human learning.

## 6. Methodology Insights

Commonly Used Methods: RLHF (Ouyang et al., 2022), PPO optimization, supervised fine-tuning with human or model feedback, and multi-agent coordination strategies.

Promising Directions: Multi-turn reinforcement learning (SCoRe, GLoRe), feedback-guided decoding (Tree-of-Thought, GRACE), and hybrid architectures combining prompting with self-corrective feedback.

Evaluation: Use reasoning accuracy, reflection depth, and consistency across multi-step reasoning as metrics.

## 7. Conclusion

The literature indicates that enhancing reasoning in LLMs requires moving beyond single-step prompting toward multi-stage, feedback-driven reasoning pipelines. RL-based fine-tuning, generation-time correction, and multi-agent collaboration form the most promising paths forward. Future research should target fully autonomous reasoning systems capable of detecting, explaining, and correcting their own reasoning processes without external aid.

## References
Havrilla, A. et al. (2024). Teaching Large Language Models to Reason with Reinforcement Learning. arXiv:2408.13296.

Kumar, A. et al. (2024). Training Language Models to Self-Correct via Reinforcement Learning. arXiv.

Ma, H. et al. (2024). Coevolving with the Other You: Cooperative Multi-Agent Reinforcement Learning. NeurIPS 2024.

Wang, J. et al. (2024). Mixture-of-Agents Enhances Large Language Model Capabilities. arXiv.

Yao, S. et al. (2023). Tree-of-Thought: Deliberate Problem Solving with Large Language Models. arXiv.

Chen, X. et al. (2023). Teaching Large Language Models to Self-Debug. arXiv.

Gulcehre, C. et al. (2023). Reinforced Self-Training (ReST) for Language Modeling. arXiv.

Pan, L. et al. (2023). Automatically Correcting Large Language Models: A Survey. arXiv.

Ouyang, L. et al. (2022). Training Language Models to Follow Instructions with Human Feedback. arXiv.

Yuan, Y. et al. (2025). Agent-R: Self-Corrective Reasoning with Monte Carlo Tree Search. arXiv.

Li, J. et al. (2023). DIVERSE: Diverse Decoding for Reasoning Tasks. arXiv.

Khalifa, M. et al. (2023). GRACE: Feedback-Guided Decoding for LLM Reasoning. arXiv.

Bai, Y. et al. (2022). Training a Helpful and Harmless Assistant with RLHF. Anthropic Technical Report.

Dubois, Y. et al. (2023). AlpacaFarm: Benchmarking RLHF Fine-Tuning Methods. arXiv.

Kim, Y. et al. (2023). Self-Refinement via Prompting: Exploring LLM Intrinsic Correction. arXiv.

