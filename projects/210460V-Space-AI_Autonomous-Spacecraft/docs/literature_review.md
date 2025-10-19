# Literature Review: Space AI:Autonomous Spacecraft

**Student:** 210460V\
**Research Area:** Space AI:Autonomous Spacecraft\
**Date:** 2025-09-01

## Abstract
The growing complexity of autonomous spacecraft operations demands trajectory generation models that combine **data-driven intelligence** with **physical consistency**. Traditional deep learning architectures, while powerful in pattern recognition, often struggle to maintain adherence to the physical laws governing orbital motion when extrapolated beyond training data. The **Autonomous Rendezvous Transformer (ART)** represents a major leap forward in this domain by formulating spacecraft rendezvous as a sequence prediction problem, leveraging the Transformer’s long-range temporal modeling capability to imitate optimal trajectories learned from convex optimization datasets. However, like most data-driven models, ART remains fundamentally unconstrained by dynamics, leading to potential deviations from physically feasible orbital behavior.  

In parallel, the field of **Physics-Aware Machine Learning (PIML)**—notably through *Physics-Informed Neural Networks (PINNs)* and *Transformer-based physics models*—has demonstrated how embedding physical laws within learning objectives can produce models that are both interpretable and dynamically stable. By integrating these two paradigms, the **Physics-Aware Autonomous Rendezvous Transformer (PA-ART)** introduces a novel framework that couples imitation learning with orbital dynamics regularization.

This literature review synthesizes the research foundations of both ART and PIML, highlighting how their convergence enables the development of physics-consistent, data-efficient, and robust learning systems for spacecraft trajectory generation.

## 1. Introduction
Autonomous spacecraft rendezvous and docking are among the most complex challenges in modern astrodynamics. They require precise control of multi-body orbital motion under constraints of limited fuel, communication delay, and uncertain environmental disturbances. Recent advances in machine learning have opened new possibilities for **data-driven trajectory optimization**, where neural networks learn control policies or state transitions directly from optimal control datasets. Among these, the **Autonomous Rendezvous Transformer (ART)** framework has shown particular promise, recasting trajectory generation as a **sequence modeling task**. By leveraging the self-attention mechanism of Transformers, ART effectively captures long-range temporal dependencies between orbital states and control inputs, outperforming recurrent and convolutional architectures in learning complex rendezvous maneuvers.

Yet, while ART can mimic optimal control behavior with high fidelity, its **lack of explicit physics awareness** limits its generalization in unseen or perturbed orbital regimes. This shortcoming mirrors a broader issue in machine learning: purely data-driven models can achieve low training error but violate fundamental conservation or kinematic laws when extrapolated. Such deviations can be catastrophic in safety-critical domains like spaceflight, where even small physical inconsistencies can render a trajectory infeasible.

The emerging discipline of **Physics-Aware Machine Learning (PIML)** - also known as *Physics-Informed Machine Learning* - addresses this issue by integrating governing physical principles directly into the learning process. Approaches such as **Physics-Informed Neural Networks (PINNs)** embed differential equation residuals into the loss function, while newer **Transformer-based physics models** incorporate physical constraints through attention mechanisms and parameter conditioning. These methods have proven effective in improving data efficiency, stability, and long-horizon prediction accuracy.

Bridging these two areas, **Physics-Aware ART (PA-ART)** represents a new generation of learning frameworks that fuse the **representational power of Transformers** with the **physical rigor of orbital dynamics**. By augmenting ART’s imitation-based objective with physics-consistency and multi-step rollout losses, PA-ART aims to generate trajectories that are not only behaviorally realistic but also dynamically valid across extended horizons.  

The following sections of this literature review examine the foundational work behind both ART and Physics-Aware Machine Learning, identify research gaps in each domain, and discuss how their integration forms the conceptual backbone for PA-ART. This synthesis establishes the motivation for developing hybrid learning systems capable of **physically grounded autonomy** in future space mission scenarios.

## 2. Search Methodology

### Search Terms Used
- Autonomous Rendezvous Transformer
- Physics-Informed Neural Networks

### Databases Searched
- [ ] Google Scholar
- [ ] ArXiv
- [ ] ScienceDirect

### Time Period
[e.g., 2018-2024, focusing on recent developments]

## 3. Key Areas of Research

### 3.1 Transformers (background and mechanics)
Transformers are sequence models built around self-attention, which weighs each token’s representation against all others via query–key dot products and a softmax normalization. This allows modeling long-range dependencies while processing sequences in parallel (unlike RNNs). Causal (GPT-style) Transformers enforce temporal ordering by masking out future positions so that token i cannot attend to tokens j>i.

### 3.2 Autonomous Rendezvous Transformer
Spacecraft rendezvous, proximity operations, and docking (RPOD) require trajectories that are dynamically feasible, safe (e.g., respect keep-out zones), and fuel-efficient. Classical optimal control reliably enforces constraints but can be slow or brittle without good initial guesses. Pure learning methods are fast but struggle to provide hard guarantees. ART addresses this gap by learning to *generate high-quality, time-parameterized warm-starts* for a sequential convex program (SCP), combining learning’s speed with optimization’s guarantees.

#### Core Contributions
ART contributes: (i) a Transformer-based framework that casts warm-start generation for non-convex OCPs as conditional sequence modeling; (ii) a systematic study of architectural and training choices (representation, conditioning, inference modes); and (iii) evidence that ART produces accurate, fuel-efficient trajectories and reduces iterations and runtime when seeding an SCP for safety-critical constraints.

#### Architecture
**Representation:** Each time step is encoded with four interleaved modalities: reward-to-go $R(t_i)$, constraint-to-go $C(t_i)$, state $x(t_i)$, and control $u(t_i)$.\
**Encoders and time:** Modality-specific linear encoders plus learned positional/time embeddings.\
**Backbone:** A causal GPT-style Transformer autoregressively predicts future controls and (optionally) states.\
**Training:** Teacher-forced supervised learning on offline optimal/near-optimal trajectories with an L2​ loss on one-step state/control predictions.\
**Conditioning knob:** At inference, the user specifies initial $R(t_1)$ (cost ambition) and $C(t_1)$ (feasibility tendency), steering the generated plan before SCP refinement.

#### Dynamics and Inference
**MDP framing:** State x is the relative motion (e.g., RTN or ROE), action $u$ is impulsive $\Delta$ $v$, dynamics propagate via a known model; reward penalizes $\|\Delta v \|$; constraints capture keep-out and approach logic.

**Two modes:**
1.  _Transformer-only_: model predicts both u and x.
    
2.  _Dynamics-in-the-loop_: model predicts u while x is propagated by dynamics - improving feasibility and used for warm-starts.**Warm-start + SCP.** The generated trajectory initializes an SCP that enforces non-convex constraints (e.g., keep-out ellipsoids) through linearization with trust-region updates.
    

#### Applications and Datasets
The paper instantiates ART on three rendezvous OCPs in ROE space: (1) a convex two-point boundary value problem, (2) a convex rendezvous with a predocking waypoint and approach cone, and (3) a non-convex rendezvous with an ellipsoidal keep-out zone solved via SCP. Large offline datasets (ISS-like scenarios) are synthesized by sampling initial conditions and solving these OCPs to provide supervision.

### 3.2 Physics-Aware Machine Learning
hysics-Aware Machine Learning (often termed *Physics-Informed Machine Learning*, PIML) bridges the gap between data-driven learning and physical modeling. Instead of learning purely statistical correlations, PIML embeds known physical principles - such as conservation laws, kinematic equations, or boundary constraints - into the model’s structure or loss function. This fusion improves **data efficiency**, **generalization**, and **physical interpretability**, making it a cornerstone for reliable modeling of real-world dynamical systems.  

Two major approaches have defined the evolution of physics-aware learning:  
(1) *Physics-Informed Neural Networks (PINNs)* and  
(2) *Transformer-based models for physical dynamics*.  
Both have shaped the conceptual and methodological foundation for the Physics-Aware ART framework.

#### Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) directly integrate governing equations into their training objectives. A PINN represents the solution of a physical system through a neural network, where the total loss combines data-fitting terms and physics residuals derived from partial differential equations (PDEs). This ensures that learned solutions respect known physical laws even when data are sparse or noisy.

Early PINN studies (Raissi et al., 2019) manually balanced data and physics terms, often resulting in unstable convergence. Later works introduced **adaptive weighting strategies** that treat loss balancing as a multi-task optimization problem. For instance, Xiang et al. (2021) used likelihood maximization to dynamically adjust each term’s weight, while Li & Feng (2022) applied a minimax formulation that trained the weighting factors themselves. Such dynamic weighting has proven essential for handling stiff or multi-scale PDE problems, where a single loss component might otherwise dominate the learning process.

To address **long-term temporal instability**, researchers developed *time-marching PINNs*. Instead of solving an entire trajectory globally, these methods divide the temporal domain into smaller segments and train local sub-networks sequentially. This approach - pioneered by Wight & Zhao (2020) and later extended by Chen et al. (2024) - ensures continuity between segments and reduces error accumulation over long horizons. As a result, PINNs have become highly effective for long-duration dynamical predictions in applications such as fluid dynamics, turbulence, material mechanics, and biomedical modeling.

#### Transformer-Based Models for Physical Dynamics

Transformers, originally designed for sequence modeling, have recently emerged as powerful tools for physics-informed learning. Their self-attention mechanism captures long-range spatial and temporal dependencies, making them well-suited for multi-scale physical systems.

The **PDE-Transformer** (Holzschuh et al., 2025) exemplifies this trend by treating PDE solutions as sequence data. It encodes spatial-temporal patches as tokens and conditions the attention layers on physical parameters such as boundary conditions and PDE coefficients. This conditioning acts as a *soft physics constraint*, guiding the model toward physically consistent behavior without explicitly encoding PDEs.

Hybrid architectures that combine **Transformers with U-Nets or CNNs** have further enhanced the representation of local and global physical effects. For example, Liu et al. (2025) introduced a U-Net Transformer constrained by PDE residuals at multiple layers, enabling extrapolation beyond the training horizon. Similarly, Liang et al. (2023) demonstrated that embedding physical priors allowed a model to predict arbitrarily long trajectories given only a short input sequence—indicating that it learned the underlying governing rules rather than memorizing patterns.


## 4. Research Gaps and Opportunities
### Gap 1: Balancing Physics and Data in Sequential Models
**Why it matters:**  
Existing physics-aware models often struggle to balance the influence of physical constraints and data fidelity, leading to either over-constrained dynamics or loss of learning flexibility.  
**How PA-ART project addresses it:**  
PA-ART introduces a *progressive weighting scheme* for the physics loss ($\lambda_{\text{dyn}}$\) and rollout loss ($\alpha_{\text{roll}}$\), enabling a gradual shift from data imitation to physics compliance during training.

### Gap 2: Long-Horizon Instability and Error Accumulation
**Why it matters:**  
Many sequence models—including PINNs and Transformers—suffer from compounding errors when predicting over long horizons, resulting in numerical divergence.  
**How PA-ART addresses it:**  
PA-ART employs a *multi-step rollout loss* that enforces consistency across several future steps, reducing accumulated error and improving long-horizon stability.

### Gap 3: Lack of Physics Integration in Transformer-Based Space Trajectory Models
**Why it matters:**  
Transformers have shown promise for sequence prediction but are rarely applied to space trajectory generation under explicit orbital dynamics constraints.  
**How PA-ART addresses it:**  
By embedding orbital physics into ART’s training loss, PA-ART bridges data-driven imitation learning and analytical spacecraft dynamics, producing physically valid trajectories even in extrapolated orbital regimes.

## 5. Theoretical Framework
The theoretical foundation of PA-ART lies in *physics-informed learning theory*, where physical laws act as inductive biases that restrict the hypothesis space of the neural network. By penalizing physically inconsistent outputs, the model’s optimization trajectory is regularized toward solutions that are both data-consistent and dynamically feasible. This aligns with the variational principles in physics-informed deep learning and with optimal control formulations used in orbital mechanics.

Mathematically, PA-ART minimizes a composite loss:

$L$ = $L_{\text{imit}}\$ + $\lambda_{\text{dyn}}$ $L_{\text{dyn}}$ + $\alpha_{\text{roll}}$ $\mathcal{L}_{\text{roll}}$

where $\mathcal{L}_{\text{imit}}$ enforces imitation of optimal trajectories, and the physics-aware terms impose consistency with discrete orbital propagation laws.

## 6. Methodology Insights
Current methodologies in physics-aware machine learning generally fall into two categories:
1. **Loss-based integration** - embedding physical residuals directly in the training objective (e.g., PINNs, PA-ART).  
2. **Architecture-based integration** - incorporating physical priors through structured architectures or conditioning (e.g., PDE-Transformers).

Among these, **loss-based integration** offers the flexibility to retrofit existing architectures (such as Transformers) with physical awareness without altering their core design.  
This adaptability made it the preferred approach for extending ART into PA-ART, where the physical knowledge is encoded through differentiable orbital propagators inside the loss function.

## 7. Conclusion
The literature on Physics-Aware Machine Learning emphasizes the power of combining deep learning with physical reasoning. From PINNs that encode governing equations to Transformers that learn physical consistency through attention and conditioning, these models demonstrate that physics-guided learning enhances both interpretability and reliability.  

Building on these ideas, **Physics-Aware ART (PA-ART)** extends the Transformer paradigm for autonomous spacecraft rendezvous by embedding orbital dynamics into its loss function.  
This hybridization of imitation learning and physics constraints directly addresses the long-horizon instability and dynamic infeasibility seen in purely data-driven approaches—paving the way toward robust, physically consistent trajectory generation for future space autonomy systems.

## 8. References

1. **Guffanti, T.**, **Gammelli, D.**, **D’Amico, S.**, & **Pavone, M.** (2024). *Transformers for trajectory optimization with application to spacecraft rendezvous.* In **Proc. IEEE Aerospace Conference**, pp. 1–9.

2. **Celestini, D.**, **Gammelli, D.**, **Guffanti, T.**, **D’Amico, S.**, **Capello, E.**, & **Pavone, M.** (2024). *Transformer-Based Model Predictive Control: Trajectory Optimization via Sequence Modeling.* **IEEE Robotics and Automation Letters**, 9(11), 9820–9827. DOI: [10.1109/LRA.2024.3466069](https://doi.org/10.1109/LRA.2024.3466069)

3. **Vaswani, A.**, **Shazeer, N.**, **Parmar, N.**, **Uszkoreit, J.**, **Jones, L.**, **Gomez, A.N.**, **Kaiser, Ł.**, & **Polosukhin, I.** (2017). *Attention is all you need.* In **Advances in Neural Information Processing Systems**, 30.

4. **Radford, A.**, **Narasimhan, K.**, **Salimans, T.**, & **Sutskever, I.** (2018). *Improving language understanding by generative pre-training.* [Online]. Available: [OpenAI Research](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

5. **Dosovitskiy, A.**, **Beyer, L.**, et al. (2021). *An image is worth 16x16 words: Transformers for image recognition at scale.* In **International Conference on Learning Representations**.

6. **Radosavovic, I.**, **Xiao, T.**, **Zhang, B.**, **Darrell, T.**, **Malik, J.**, & **Sreenath, K.** (2023). *Learning humanoid locomotion with transformers.* arXiv preprint: [arXiv:2303.03381](https://arxiv.org/abs/2303.03381)

7. **Radford, A.**, **Wu, J.**, **Child, R.**, **Luan, D.**, **Amodei, D.**, & **Sutskever, I.** (2019). *Language models are unsupervised multitask learners.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

8. **Raissi, M.**, **Perdikaris, P.**, & **Karniadakis, G.E.** (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* **Journal of Computational Physics**, 378, 686–707. DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

9. **Raissi, M.**, **Perdikaris, P.**, **Ahmadi, N.**, & **Karniadakis, G.E.** (2024). *Physics-Informed Neural Networks and Extensions.* *Frontiers of Science Awards 2024.* [arXiv:2408.16806](https://arxiv.org/pdf/2408.16806v1)

10. **Xiang, Z.**, **Peng, W.**, **Zheng, X.**, **Zhao, X.**, & **Yao, W.** (2021). *Self-adaptive loss balanced Physics-informed neural networks for the incompressible Navier–Stokes equations.* [arXiv:2104.06217](https://arxiv.org/pdf/2104.06217)

11. **Li, S.**, & **Feng, X.** (2022). *Dynamic Weight Strategy of Physics-Informed Neural Networks for the 2D Navier–Stokes Equations.* **Entropy**, 24(9), 1254. DOI: [10.3390/e24091254](https://doi.org/10.3390/e24091254)

12. **Wight, C.L.**, & **Zhao, J.** (2020). *Solving Allen–Cahn and Cahn–Hilliard Equations using Adaptive Physics-Informed Neural Networks.* [arXiv:2007.04542](https://arxiv.org/abs/2007.04542)

13. **Chen, Z.**, **Lai, S.K.**, & **Yang, Z.** (2024). *AT-PINN: Advanced time-marching physics-informed neural network for structural vibration analysis.* **Thin-Walled Structures**, 196, 111423. DOI: [10.1016/j.tws.2023.111423](https://doi.org/10.1016/j.tws.2023.111423)

14. **Steinfurth, B.**, & **Weiss, J.** (2024). *Assimilating experimental data of a mean three-dimensional separated flow using physics-informed neural networks.* **Physics of Fluids**, 36(1). DOI: [10.1063/5.0183463](https://doi.org/10.1063/5.0183463)

15. **Cai, S.**, **Wang, Z.**, **Wang, S.**, **Perdikaris, P.**, & **Karniadakis, G.** (2021). *Physics-Informed Neural Networks (PINNs) for Heat Transfer Problems.* **Journal of Heat Transfer**, March 2021. DOI: [10.1115/1.4050542](https://doi.org/10.1115/1.4050542)

16. **Zhang, E.**, **Dao, M.**, **Karniadakis, G.E.**, & **Suresh, S.** (2022). *Analyses of internal structures and defects in materials using physics-informed neural networks.* **Science Advances**, 8(7). DOI: [10.1126/sciadv.abk0644](https://doi.org/10.1126/sciadv.abk0644)

17. **Rodrigues, J.A.** (2024). *Using Physics-Informed Neural Networks (PINNs) for Tumor Cell Growth Modeling.* **Mathematics**, 12(8), 1195. DOI: [10.3390/math12081195](https://doi.org/10.3390/math12081195)

18. **Holzschuh, B.**, **Liu, Q.**, **Kohl, G.**, & **Thuerey, N.** (2025). *PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations.* [arXiv:2505.24717](https://arxiv.org/abs/2505.24717)

19. **Liu, Y.**, **Wang, J.**, **Zhang, N.**, **Qian, X.**, & **Chai, S.** (2025). *Beyond training horizons: A physics-informed U-Net transformer for time-dependent CO₂ storage simulation in environmental applications.* **Applied Soft Computing**, 185, 113987. DOI: [10.1016/j.asoc.2025.113987](https://doi.org/10.1016/j.asoc.2025.113987)

20. **Liang, Y.**, **Niu, R.**, **Yue, J.**, & **Lei, M.** (2023). *A Physics-Informed Recurrent Neural Network for Solving Time-Dependent Partial Differential Equations.* **International Journal of Computational Methods**, 21(10). DOI: [10.1142/s0219876223410037](https://doi.org/10.1142/s0219876223410037)

21. **Karniadakis, G.E.**, **Kevrekidis, I.G.**, **Lu, L.**, **Perdikaris, P.**, **Wang, S.**, & **Yang, L.** (2021). *Physics-informed machine learning.* **Nature Reviews Physics**, 3(6), 422–440. DOI: [10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)

22. **Parisotto, E.**, et al. (2020). *Stabilizing Transformers for Reinforcement Learning.* **PMLR**, pp. 7487–7498. Available: [PMLR Proceedings](https://proceedings.mlr.press/v119/parisotto20a)

23. **Janner, M.**, **Fu, J.**, **Zhang, M.**, & **Levine, S.** (2019). *When to Trust Your Model: Model-Based Policy Optimization.* [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2019/file/5faf461eff3099671ad63c6f3f094f7f-Paper.pdf)

24. **Tarvainen, A.**, & **Valpola, H.** (2017). *Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.* **Neural Information Processing Systems**, 2017. [Proceedings Link](https://proceedings.neurips.cc/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)


---