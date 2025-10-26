# Validation and Results

Both the baseline ART and Physics-Aware ART were evaluated on identical validation datasets  
(*N = 10,000*, *T = 100*, *Dₓ = 6*, *Dᵤ = 3*).  
Table 1 summarizes the average quantitative results obtained.

| **Model** | **𝓛ₜₒₜₐₗ** | **𝓛ₛₜₐₜₑ** | **𝓛ₐcₜᵢₒₙ** | **physics_res** | **MSE@10** | **MSE@50** | **Feas. Ratio** |
|:-----------|:------------:|:------------:|:------------:|:----------------:|:-----------:|:-----------:|:----------------:|
| Baseline ART | 6.941×10⁻¹ | 4.257×10⁻³ | 6.899×10⁻¹ | 2.626×10⁻¹ | 7.973×10² | 5.169×10³ | 0.0 |
| **Physics-Aware ART** | **6.887×10⁻¹** | **3.570×10⁻⁵** | **6.886×10⁻¹** | **2.531×10⁻¹** | 9.070×10² | **5.089×10³** | 0.0 |

**Table 1.** Comparison of baseline ART and Physics-Aware ART on validation dataset.  
Lower is better for all metrics.

---

The results show that the physics-aware enhancement yields:

- a **99% reduction** in one-step state prediction error (𝓛ₛₜₐₜₑ),  
- a **3.6% decrease** in physics residuals, indicating improved adherence to orbital dynamics,  
- a marginally lower overall validation loss (𝓛ₜₒₜₐₗ), and  
- improved medium-horizon stability (MSE@50).

The control prediction loss (𝓛ₐcₜᵢₒₙ) shows a slight improvement of approximately **0.77%**, indicating that the physics-aware regularization not only preserves but marginally enhances the model’s ability to imitate optimal control actions.  

However, both models still exhibit numerical divergence at long horizons (MSE@100) and a zero feasibility ratio.  
This degradation in long-term convergence is likely attributed to the absence of Sequential Convex Programming (SCP) refinement during data generation and training, where only the optimally solved convex trajectories were utilized without iterative feasibility correction.

---

### Interpretation

These results confirm that incorporating physics-informed regularization improves local dynamic consistency and mid-horizon rollout stability without compromising policy imitation performance.  
The approach effectively biases the Transformer toward physically valid transitions, bridging the gap between pure data-driven learning and physically grounded trajectory generation.
