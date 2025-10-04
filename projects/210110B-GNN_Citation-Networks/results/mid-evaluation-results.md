# Mid Evaluation Results: GNN:Citation Networks

**Student:** 210110B  
**Research Area:** GNN:Citation Networks  
**Date:** 2025-09-30  

---

## Iteration 2: H-UniMP Core (Relation-Aware)

### Approach
This version extends Iteration-1 UniMP by making the model **heterogeneity-aware**:
- Separate linear projections for each edge type:
  - Citation (paper → paper),
  - Author → Paper,
  - Paper → Author.
- Typed mean aggregation per relation, then combined into a unified representation.
- Masked label injection remains the same as Iter-1.

### Motivation
Citation networks are inherently heterogeneous, with multiple node and edge types.  
Treating edges differently allows the model to capture semantic differences and improves representation learning, particularly for predicting venues and modeling author contributions.

---

## Results (Citation-Network V1, title classification)

| Model             | Accuracy | Macro-F1 | Micro-F1 |
|-------------------|----------|----------|----------|
| Iter-1 UniMP      | 70.20%   | 68.70%   | 69.40%   |
| Iter-2 H-UniMP    | 73.71%   | 72.50%   | 70.20%   |

---

## Observations
- Incorporating heterogeneous edge semantics gives a **clear performance gain (+3–4%)** over Iter-1.
- The model achieves better balance across classes, improving **Macro-F1**, which indicates stronger robustness for less frequent venues.
- Even under CPU-only training, the relation-aware model provides measurable benefits.

---

## Next Step
Iteration 3 will address **label uncertainty** by gating how much label information is injected per node.  
This aims to reduce the impact of noisy labels during propagation and further improve stability.

---
