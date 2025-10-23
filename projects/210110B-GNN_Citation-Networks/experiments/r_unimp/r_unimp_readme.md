# Iteration 2: H-UniMP Core (Relation-Aware)

### Approach
This version extends Iter-1 by making the model **heterogeneity-aware**:
- Separate linear projections for each edge type:
  - Citation (paper → paper),
  - Author → Paper,
  - Paper → Author.
- Typed mean aggregation per relation, then combined.
- Label injection remains the same as Iter-1.

### Motivation
Citation networks are heterogeneous. Treating edges differently should improve representation learning, especially for venues and authors.

### Results
Citation-network V1 (title classification):

| Model          | Accuracy | Macro-F1 | Micro-F1 |
|----------------|----------|----------|----------|
| Iter-1 UniMP | 70.20%    | 68.70%    | 69.40%    |
| Iter-2 R-UniMP    | 73.71%    | 72.50%    | 70.20%    |

### Observations
- Clear gain (+3-4%) from incorporating heterogeneous edge semantics.
- The model is more robust across classes with few labeled papers.

### Next Step
Iteration 3 will address **label uncertainty** by gating how much label information is injected per node.
