# Methodology: Exploring Hyperparameters and Training Strategies in Plenoxels: Radiance Fields without Neural Networks

**Student:** 210302P <br>
**Research Area:** 3D Vision:Neural Radiance Fields<br>
**Date:** 2025-09-01<br>

## 1. Overview

This document outlines the methodology for systematically exploring and evaluating the impact of various hyperparameters, loss functions, and training strategies on the Plenoxels framework. The core of this research is to move beyond the standard Plenoxel configuration to identify modifications that enhance reconstruction quality, training stability, and perceptual fidelity. The methodology is designed as a series of controlled experiments where individual components of the training pipeline are modified and their effects are measured against a consistent baseline, providing clear insights into the strengths and limitations of each approach.

## 2. Research Design

The research follows a quantitative, experimental design. A baseline Plenoxel model is first established using its default configuration. Subsequently, a series of experiments are conducted, where one key aspect of the model or training process is systematically varied at a time. This approach allows for the isolated analysis of the impact of:

- Learning Rate Schedules: Comparing fixed rates, cosine annealing, and cyclical learning rates.
- Regularization: Investigating the effects of uniform vs. separate Total Variation (TV) weights and a progressive TV decay schedule.
- Model Parameters: Tuning the spherical harmonics (SH) order.
- Initialization: Comparing the default initialization with a Gaussian distribution.

The performance of each modification is measured using a consistent set of metrics, enabling a direct comparison of convergence behavior, final reconstruction quality, and computational efficiency.

## 3. Data Collection

### 3.1 Data Sources

The primary data source for this research is the NeRF Synthetic Dataset, a standard benchmark for novel view synthesis tasks.

### 3.2 Data Description

The dataset contains multiple synthetic scenes captured from a variety of viewpoints, providing RGB images and their corresponding camera poses. For this study, the Lego scene was selected for all experiments. This scene is ideal as it exhibits moderate geometric complexity and diverse visual features, making it a suitable testbed for evaluating the reconstruction capabilities of Plenoxels.

### 3.3 Data Preprocessing

To ensure uniform input across all experimental models and runs, all images from the dataset were resized to a consistent resolution. The training and evaluation splits followed the original dataset configuration to maintain consistency with standard benchmarks.

## 4. Model Architecture

The core architecture is Plenoxels, a framework that represents 3D scenes using a sparse voxel grid without a neural network. Each voxel in the grid encodes density (opacity) and color, which is modeled using spherical harmonic (SH) coefficients. Optimization is performed directly on these voxel parameters via gradient-based methods.
This research does not propose a single new model but rather investigates a series of modifications to the baseline Plenoxel training framework:

- Learning Rate Schedulers: Implementation of cosine annealing and cyclical learning rate schedules to dynamically adjust learning rates during training.
- Progressive Regularization: A training strategy where the weight of the Total Variation (TV) regularization is gradually decreased over the course of training to balance coarse structure capturing with fine detail preservation.
- Gaussian Initialization: An alternative weight initialization strategy where both the SH coefficients and density values are initialized from a Gaussian distribution instead of the default.

### 4.1. Implementation of Experimental Modifications

The following code snippets, using Python's argparse library, define the command-line flags used to control the different experimental configurations tested in this research.

#### 4.1.1. Gaussian Initialization

This set of arguments allows for switching the initialization of the model's parameters from the default to a Gaussian distribution. This is central to the experiment on the "Effect of Initialization," allowing for analysis of how different starting conditions affect convergence.

- use_gaussian_init: A boolean flag to enable this initialization strategy.
- gaussian_mean and gaussian_std: Control the mean and standard deviation of the Gaussian distribution for both SH coefficients and density, allowing for precise control over the initial parameter space.

```python
group.add_argument('--use_gaussian_init', action='store_true', default=False,
help='Initialize SH coefficients and density with Gaussian distribution')
group.add_argument('--gaussian_mean', type=float, default=0.0, help='Mean for Gaussian initialization')
group.add_argument('--gaussian_std', type=float, default=0.01, help='Standard deviation for Gaussian initialization')
group.add_argument('--gaussian_sigma', type=float, default=0.1, help='Initial sigma for density if using Gaussian')
group.add_argument('--gaussian_sigma_bg', type=float, default=0.1, help='Initial sigma for background density if using Gaussian')
```

#### 4.1.2. Cyclical Learning Rate Scheduler

These arguments enable the use of a cyclical learning rate schedule, an alternative to the default exponential or cosine decay. This modification is a core part of the "Hyperparameter & Scheduler Experiments."

- use_cyclic_lr: A boolean flag to activate the cyclical learning rate scheduler.
- cyclic_base_lr and --cyclic_max_lr: Define the lower and upper bounds of the learning rate cycle.
- cyclic_step_size: Controls the duration of a half-cycle, allowing for adjustment of the learning rate's oscillation frequency.
- cyclic_mode: Allows testing of different cycle shapes, such as 'triangular' or 'triangular2'.
  - `triangular`: A simple up-and-down oscillation with constant amplitude.
  - `triangular2`: Similar to `triangular`, but each cycle’s amplitude is halved, leading to a gradual reduction in oscillation range.
  - `exp_range`: Uses an exponentially decaying upper bound (scaled by gamma), gradually lowering the entire learning rate envelope over time.

```python
group = parser.add_argument_group("cyclic_lr")
group.add_argument('--use_cyclic_lr', action='store_true', default=False,
                   help='If set, use cyclic learning rate schedule instead of exponential/cosine')
group.add_argument('--cyclic_base_lr', type=float, default=1e-4,
                   help='Lower bound of the cyclic learning rate')
group.add_argument('--cyclic_max_lr', type=float, default=1e-2,
                   help='Upper bound of the cyclic learning rate')
group.add_argument('--cyclic_step_size', type=int, default=2000,
                   help='Number of iterations per half cycle (so full cycle = 2 * step_size)')
group.add_argument('--cyclic_mode', choices=["triangular", "triangular2", "exp_range"], default="triangular",
                   help='Cyclic mode: triangular, triangular2, or exp_range')
group.add_argument('--cyclic_gamma', type=float, default=1.0,
                   help='Scaling factor for exp_range mode')
```

#### 4.1.3. Cosine Annealing Scheduler

These arguments enable the use of a **cosine annealing learning rate schedule**, which gradually decreases the learning rate following a cosine curve instead of exponential decay. This helps achieve smoother convergence and can improve stability during long training runs.

- use_cosine_annealing: A boolean flag to activate the cosine annealing scheduler. When enabled, all learning rates will follow a cosine decay pattern instead of the default exponential schedule.
- cosine_T_max: Defines the total number of steps for one complete cosine cycle. If set to `0`, the learning rate remains constant after the warmup phase.
- cosine_eta_min: Specifies the minimum learning rate value reached at the end of a cosine cycle.
- cosine_warmup_steps: Number of initial steps used for linear warmup before starting the cosine decay, allowing the model to stabilize early in training.

```python
group = parser.add_argument_group("cosine_annealing")
group.add_argument('--use_cosine_annealing', action='store_true', default=False,
                   help='If set, use cosine annealing schedule (instead of exponential) for ALL LRs')
group.add_argument('--cosine_T_max', type=int, default=250000,
                   help='Number of steps for the cosine cycle (T_max). If 0, cosine will behave like constant after warmup')
group.add_argument('--cosine_eta_min', type=float, default=0.0,
                   help='Minimum learning rate reached by cosine annealing')
group.add_argument('--cosine_warmup_steps', type=int, default=0,
                   help='Linear warmup steps from 0 to initial lr before cosine annealing starts')
```

#### 4.1.4. Progressive TV and Sparsity Scheduling

This code enables the progressive regularization training strategy. It allows the strength of the Total Variation (TV) and sparsity regularizations to decay over time, balancing initial smoothness with final detail preservation.

- use_progressive_tv: A flag to activate this decay schedule.
- progressive_tv_start and --progressive_tv_end: Define the initial and final values for the main TV regularization weight, allowing for a controlled decay.
- progressive_tv_sh_start and --progressive_tv_sh_end: Similarly control the decay for the TV regularization on the SH coefficients.
- progressive_sparsity_start and --progressive_sparsity_end: Control the decay of the sparsity fraction used in regularization.

```python

group = parser.add_argument_group("progressive_tv")
group.add_argument('--use_progressive_tv', action='store_true', default=False,
                   help='Enable exponential progressive TV and sparsity decay')
group.add_argument('--progressive_tv_start', type=float, default=1e-3,
                   help='Initial lambda_tv value for progressive decay')
group.add_argument('--progressive_tv_end', type=float, default=1e-5,
                   help='Final lambda_tv value for progressive decay')
group.add_argument('--progressive_tv_sh_start', type=float, default=1e-3,
                   help='Initial lambda_tv_sh value for progressive decay')
group.add_argument('--progressive_tv_sh_end', type=float, default=1e-6,
                   help='Final lambda_tv_sh value for progressive decay')
group.add_argument('--progressive_sparsity_start', type=float, default=0.01,
                   help='Initial sparsity fraction for TV regularization')
group.add_argument('--progressive_sparsity_end', type=float, default=0.001,
                   help='Final sparsity fraction for TV regularization')
```

## 5. Experimental Setup

### 5.1 Evaluation Metrics

The performance of all models was evaluated using a combination of quantitative metrics to assess both pixel-level accuracy and perceptual quality:

- PSNR (Peak Signal-to-Noise Ratio): Measures the quality of the reconstruction by comparing the pixel differences between the rendered and ground-truth images.
- MSE (Mean Squared Error): A direct measure of the average squared difference between pixel values.
- LPIPS (Learned Perceptual Image Patch Similarity): A metric that measures perceptual similarity based on deep feature embeddings, which better aligns with human judgment of image quality.
- training time

### 5.2 Baseline Models

To establish a clear benchmark for all subsequent experiments, the standard Plenoxel model was trained for 10 epochs using its default configuration on the Lego scene. The performance was logged at each epoch, measuring both Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE). These results, detailed below, serve as the baseline against which all proposed modifications are compared. The total training time for this baseline run was 1434.22 seconds.

| Epoch | PSNR (dB) | MSE      |
| :---: | :-------: | :------- |
|   1   |  10.3942  | 0.093092 |
|   2   |  30.2549  | 0.001068 |
|   3   |  31.2994  | 0.000870 |
|   4   |  30.9720  | 0.000922 |
|   5   |  33.5443  | 0.000595 |
|   6   |  33.9279  | 0.000561 |
|   7   |  34.1127  | 0.000546 |
|   8   |  34.2212  | 0.000538 |
|   9   |  34.2870  | 0.000532 |
|  10   |  34.3300  | 0.000529 |
| Final |  34.3596  | 0.000527 |

### 5.3 Hardware/Software Requirements

- Hardware: All experiments were performed on a Google Colab T4 GPU.
- Software: The implementation is based on Python. The full codebase, including all training and evaluation scripts, is publicly available in a GitHub repository to ensure reproducibility.

## 6. Implementation Plan

| Phase   | Tasks                                                                                                                                                           | Duration | Deliverables                                                                           |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------- |
| Phase 1 | Setup and Baseline Establishment: Prepare the dataset, implement the baseline Plenoxel model, and run initial experiments to establish a performance benchmark. | 2 weeks  | A clean, preprocessed dataset and documented baseline performance metrics (PSNR, MSE). |
| Phase 2 | Hyperparameter & Scheduler Experiments: Implement and test different learning rate schedules (cosine, cyclical) and TV regularization strategies.               | 3 weeks  | Comparative results and graphs showing the impact of different schedulers.             |
| Phase 3 | Advanced Training Strategy Experiments: Implement and evaluate the effects of Gaussian initialization and progressive TV regularization.                        | 3 weeks  | Analysis of convergence behavior and final quality for advanced strategies.            |
| Phase 4 | Perceptual Analysis & Reporting: Calculate LPIPS for key experiments, analyze all results, generate final figures, and write the final report and conclusion.   | 2 week   | Final report with comprehensive analysis, figures, and conclusion.                     |

## 7. Risk Analysis

| Potential Risk                                                                                                                      | Mitigation Strategy                                                                                                                                                                                                           |
| :---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Marginal Performance Gains:** Advanced strategies may not significantly outperform simpler, well-tuned baselines.                 | Acknowledge this as a valid scientific finding. The conclusion of the research will focus on the trade-offs between complexity, runtime, and reconstruction quality, highlighting cases where simpler methods are preferable. |
| **Computational Limitations:** Training multiple configurations can be time-consuming, even with an efficient model like Plenoxels. | Utilize efficient coding practices and leverage cloud GPU resources (Google Colab). Carefully select the number and length of experimental runs to balance thoroughness with available compute time.                          |
| **Reproducibility Issues:** Minor differences in software versions or hardware could affect results.                                | Maintain a detailed `requirements.txt` file. Make the entire codebase publicly available on GitHub, including scripts for reproducing all experiments, as mentioned in the paper.                                             |

## 8. Expected Outcomes

This methodology is designed to produce the following outcomes:

- A comprehensive empirical analysis of how different hyperparameters and training strategies affect the performance of Plenoxels.
- Clear insights into the trade-offs between reconstruction fidelity (PSNR, MSE), perceptual quality (LPIPS), training time, and model complexity.
- Actionable recommendations for researchers and practitioners on how to best tune Plenoxels for their specific needs, including when simpler baselines may be more effective than complex training schemes.
- A reproducible codebase that allows others to verify the findings and build upon this research.

---
