# Methodology: Space AI:Autonomous Spacecraft

**Student:** 210460V
**Research Area:** Space AI:Autonomous Spacecraft
**Date:** 2025-09-01

## 1. Overview
While the baseline **Autonomous Rendezvous Transformer (ART)** achieves high fidelity in imitating expert demonstrations, it remains inherently data-driven and unconstrained by physical laws of motion.  
As a result, its predictions can occasionally become dynamically inconsistent or infeasible under extrapolated orbital regimes.  

To address this, the **Physics-Aware ART (PA-ART)** introduces physics-consistency principles directly into the learning process.  
This enhancement ensures that the predicted control commands produce physically valid state transitions according to orbital dynamics, improving both *trajectory feasibility* and *long-horizon stability*.  
PA-ART retains the same Transformer architecture as ART but incorporates additional physics-aware loss terms and stability mechanisms during training.

## 2. Research Design
This study follows a **model-centric enhancement approach**, combining imitation learning with physics-informed regularization.

1. **Baseline Reproduction:** Implement and train the original ART to establish benchmark performance.  
2. **Physics-Aware Integration:** Extend the loss function with physics-consistency and multi-step rollout regularization terms.  
3. **Progressive Weight Scheduling:** Introduce tunable coefficients (`λ_dyn`, `α_roll`) to gradually balance physical consistency and imitation accuracy.  
4. **Stability Enhancement:** Apply Exponential Moving Average (EMA) parameter smoothing to stabilize training and improve generalization.  

The design ensures that PA-ART learns both data fidelity (from demonstrations) and adherence to physical laws (from analytical dynamics).

## 3. Data Collection

### 3.1 Data Sources
Synthetic datasets were generated using a convex optimization-based orbital control solver.  
Each trajectory adheres to linearized orbital dynamics under the Hill–Clohessy–Wiltshire (HCW) and $J_2$-perturbed motion equations.

### 3.2 Data Description
Each trajectory $\tau_i = \{x_i(t), u_i(t)\}$ spans **T = 100 timesteps**, where:  
- **State ($x_t$):** 6D vector (relative position and velocity in the RTN frame).  
- **Control ($u_t$):** 3D vector (thrust commands or $\Delta v$ in x, y, z).  
- **Auxiliary Variables:** Reward-to-go ($r_t$) and constraint-to-go ($c_t$) values to encode mission objectives.

### 3.3 Data Preprocessing
1. **Normalization:**  
   Each variable $z$ was normalized using the training set statistics:  
   
   $z_{\mathrm{norm}} = \frac{z - \mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}$
   
   to prevent data leakage.
2. **Splitting:**  
   Data were split into 90% training and 10% validation subsets.

## 4. Model Architecture
The **PA-ART** retains the baseline ART’s Transformer encoder–decoder structure, which models sequential dependencies between states and control actions.  
No architectural changes are made to the Transformer itself.  
Instead, *physics-awareness* is introduced entirely through the **loss formulation** and **training procedure**.

### Physics-Aware Loss

The total training loss combines imitation accuracy with two physics-consistency regularization terms:

$\mathcal{L}$ = $\mathcal{L}_{\text{imit}}$ + $\lambda_{\text{dyn}}$ $\mathcal{L}_{\text{dyn}}$ + $\alpha_{\text{roll}}$ $\mathcal{L}_{\text{roll}}$

where:

- $\mathcal{L}_{\text{imit}}$: Baseline reconstruction loss for state and control matching.  
- $\mathcal{L}_{\text{dyn}}$: One-step **physics-consistency loss**, enforcing the next-state prediction to align with orbital propagation $f(x_t, \hat{u}_t)$.  
- $\mathcal{L}_{\text{roll}}$: Multi-step **rollout consistency loss**, penalizing long-horizon deviations.

#### Physics-Consistency Loss
At each timestep, the predicted control $\hat{u}_t$ is applied to the true current state $x_t$ through the analytical dynamics model $f(\cdot)$:
\[
$x_{t+1}^{\text{phys}}$ = $f(x_t, \hat{u}_t)$
\]
The loss measures deviation between the physically propagated state and the ground truth:
\[
$\mathcal{L}_{\text{dyn}}$ = 
$\frac{1}{T-1}\sum_{t=1}^{T-1}$ 
$\|x_{t+1} - f(x_t, \hat{u}_t)\|_2^2$
\]
This ensures that the Transformer’s predicted controls generate dynamically consistent transitions under orbital mechanics.

#### Rollout Consistency Loss
To improve long-horizon stability, predicted control sequences are recursively applied over a horizon of $H$ steps: $x_{t+k+1}^{\text{phys}} = f(x_{t+k}^{\text{phys}}, \hat{u}_{t+k})$, 
$[k = 0, 1, \dots, H-1]$ and compared against ground-truth rollouts: $\mathcal{L}_{\text{roll}} = \sum_{k=0}^{H-1} \|x_{t+k+1} - x_{t+k+1}^{\text{phys}}\|_2^2$

This penalizes accumulated drift, encouraging physically feasible control trajectories over extended horizons.

#### Progressive Weight Scheduling
The loss coefficients are gradually increased during training:

$\begin{aligned}
\lambda_{\text{dyn}} &\leftarrow 
\min(\lambda_{\text{dyn}} \times 10, 
\lambda_{\text{dyn}}^{\max})\\
\alpha_{\text{roll}} &\leftarrow 
\min(\alpha_{\text{roll}} \times 10, 
\alpha_{\text{roll}}^{\max})
\end{aligned}$

This incremental schedule allows the model to first learn data-driven behavior before emphasizing physics constraints, preventing unstable optimization early in training.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- **Mean Squared Error (MSE):** Measures reconstruction accuracy of states and controls.  
- **Physics Residual ($L_{\text{dyn}}$):** Quantifies physical consistency under orbital equations.  
- **Rollout Drift:** Measures cumulative trajectory deviation over long horizons.  

### 5.2 Baseline Models
- **Baseline ART:** Original transformer trained with imitation-only loss.  
- **Physics-Aware ART (PA-ART):** Proposed model with dynamics and rollout constraints.  

### 5.3 Hardware/Software Requirements
- **Hardware:** NVIDIA RTX 3090 GPU, 32 GB RAM.  
- **Software:**  
  - Python 3.10  
  - PyTorch ≥ 2.0  
  - CUDA 11.8  
  - NumPy, SciPy, Matplotlib 

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Data preprocessing | 2 weeks | Clean dataset |
| Phase 2 | Model implementation | 3 weeks | Working model |
| Phase 3 | Experiments | 2 weeks | Results |
| Phase 4 | Analysis | 1 week | Final report |

## 7. Risk Analysis

| **Risk** | **Description** | **Mitigation** |
|-----------|----------------|----------------|
| **Training Instability** | Multiple weighted loss terms may cause divergence. | Apply gradient clipping (0.5) and progressive coefficient scaling. |
| **Physics Loss Dominance** | Overweighting physics term may hinder data fitting. | Tune λ<sub>dyn</sub> and α<sub>roll</sub> using validation loss balance. |
| **Compounding Errors** | Long-horizon predictions may drift. | Use multi-step rollout training and curriculum horizon scheduling. |
| **Overfitting** | Model may memorize specific orbital regimes. | Employ Exponential Moving Average (EMA) and early stopping. |


## 8. Expected Outcomes
The Physics-Aware ART is expected to:
- Generate **dynamically feasible** trajectories consistent with orbital mechanics.  
- Exhibit **improved long-horizon stability** via multi-step rollout training.  
- Reduce **compounding prediction errors** common in purely data-driven models.  
- Achieve **smoother convergence** and better generalization through EMA parameter averaging.  

The integration of physics-consistent objectives thus provides a principled inductive bias, guiding the Transformer toward physically valid and stable spacecraft rendezvous trajectories while retaining the flexibility of deep sequence modeling.

---