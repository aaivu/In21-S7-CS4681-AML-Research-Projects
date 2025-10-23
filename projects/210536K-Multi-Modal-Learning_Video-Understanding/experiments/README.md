# Experiments

## Overview

This directory contains all experiments conducted during the research on temporal action localization in untrimmed videos. For ease of reference, the experiments are organized into two main categories based on their purpose and stage in the research process.

---

## 1. Preliminary Experiments

**[View Preliminary Experiments →](./preliminary_experiments/)**

These experiments were conducted in the early stages of research to explore different architectural components and establish baselines. They helped us understand the behavior of various model configurations and informed the design decisions for our final approach.

### Experiments Included:

- **Experiment 1: Effect of Training Epochs on ActionFormer**

  - Studied how the number of training epochs (1, 4, 8, 15, 25 epochs) affects model performance and convergence
  - Examined overfitting/underfitting behavior with different training durations
  - [Details →](./preliminary_experiments/exp_1.md)

- **Experiment 2: Temporal Feature Pyramid Network (FPN1D)**

  - Extended baseline with a 3-level temporal Feature Pyramid Network
  - Enhanced multi-scale temporal representations for capturing both short and long actions
  - [Details →](./preliminary_experiments/exp_2.md)

- **Experiment 3: Convolutional Backbone + FPN1D**

  - Tested convolutional backbone architecture combined with FPN1D neck
  - Explored alternative to Transformer-based backbone (30 epochs + 5 warm-ups)
  - [Details →](./preliminary_experiments/exp_3.md)

- **Experiment 4: SGP Backbone + FPN1D + Normal Head**

  - Implemented Scalable-Granularity Perception (SGP) backbone with FPN1D neck
  - Evaluated multi-granularity context capture with standard prediction head
  - [Details →](./preliminary_experiments/exp_4.md)

- **Experiment 5: SGP Backbone + Identity Neck + Normal Head**

  - Tested SGP backbone with identity neck (no feature pyramid)
  - Analyzed the necessity of feature pyramid with SGP architecture
  - [Details →](./preliminary_experiments/exp_5.md)

- **Experiment 6: Transformer + Identity Neck + TriDet Head**
  - Explored Transformer backbone with identity neck and TriDet-style prediction head
  - Investigated alternative head architectures for boundary prediction
  - Implementation notebook: [exp_6_transformer_identity_tridet_head.ipynb](./preliminary_experiments/notebooks/exp_6_transformer_identity_tridet_head.ipynb)

### Additional Resources:

- **Notebooks**: Detailed Jupyter notebooks with code implementations for each experiment
- **TensorBoard Logs**: Training logs for experiments 2-6 with loss curves and metrics
- **Visualizations**: Result visualizations stored in the `imgs/` folder

---

## 2. Ablation Studies

**[View Ablation Studies →](./ablation_studies/)**

These are systematic, controlled experiments conducted for the final paper to validate the contribution of each proposed component and optimize hyperparameters. Each ablation study isolates specific design choices to quantify their individual impact on model performance.

### Studies Included:

- **Ablation Study 1: Component-wise Contributions**

  - Validates individual impact of Scaled Backbone, Cross-Scale FPN (CS-FPN), and Boundary Distribution Regression (BDR) Head
  - Quantifies synergistic performance boost when all components are combined
  - [Details →](./ablation_studies/ablation_01_component_contributions.md)

- **Ablation Study 2: Effect of Local Attention Window Size**

  - Analyzes optimal window size (19, 25, 30, 37) for local attention mechanism
  - Tests hypothesis that enhanced backbone can leverage larger temporal context
  - [Details →](./ablation_studies/ablation_02_window_size.md)

- **Ablation Study 3: Design of the Feature Pyramid**

  - Determines optimal depth of feature pyramid (1, 3, 5, 6, 7 levels)
  - Validates importance of multi-scale representation for temporal action localization
  - [Details →](./ablation_studies/ablation_03_fpn_design.md)

- **Ablation Study 4: Analysis of Regression Loss Weight ($\lambda_{reg}$)**

  - Analyzes model sensitivity to regression loss weight (0.2, 0.5, 1.0, 2.0, 5.0)
  - Identifies optimal balance between classification and boundary regression objectives
  - [Details →](./ablation_studies/ablation_04_loss_weight.md)

- **Ablation Study 5: Alternative Backbone Architectures**
  - Compares Transformer vs. Hybrid Attention-Mamba vs. SGP (convolutional) backbones
  - Explores viability of modern alternatives for temporal action localization
  - [Details →](./ablation_studies/ablation_05_alternative_backbones.md)

---

## Experimental Pipeline

All experiments follow a consistent pipeline:

1. **Dataset**: Primarily THUMOS14, with additional validation on ActivityNet and EPIC Kitchens 100
2. **Features**: Pre-extracted I3D features (2048-dimensional)
3. **Training**: Adam optimizer with cosine annealing learning rate schedule
4. **Evaluation**: Mean Average Precision (mAP) at multiple IoU thresholds (0.3, 0.4, 0.5, 0.6, 0.7)
5. **Logging**: TensorBoard for loss curves and training metrics

---

## Navigating the Experiments

- For understanding the **research journey and exploration process** → Start with [Preliminary Experiments](./preliminary_experiments/)
- For **rigorous validation and paper results** → Go to [Ablation Studies](./ablation_studies/)
- For **implementation details** → Check the Jupyter notebooks in `preliminary_experiments/notebooks/`
- For **training logs and metrics** → See TensorBoard logs in `preliminary_experiments/logs_tensorboard/`
