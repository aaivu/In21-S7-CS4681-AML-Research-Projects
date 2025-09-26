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
Citation-network V1 (venue classification):

| Model          | Accuracy | Macro-F1 | Micro-F1 |
|----------------|----------|----------|----------|
| Iter-1 UniMP-lite | 61.2%    | 58.4%    | 60.7%    |
| Iter-2 H-UniMP    | 66.9%    | 64.5%    | 66.2%    |

### Observations
- Clear gain (+5–6%) from incorporating heterogeneous edge semantics.
- The model is more robust across classes with few labeled papers.

### Next Step
Iteration 3 will address **label uncertainty** by gating how much label information is injected per node.
