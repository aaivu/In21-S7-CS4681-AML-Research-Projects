# Experiment 1: Baseline UniMP (Homogeneous)

### Approach
This is a simplified baseline of UniMP due to the computer resources I have:
- Treat the citation graph as **homogeneous** (ignore author/paper/venue differences).
- Use paper features + metapath2vec embeddings as input.
- Inject labels into node features (UniMP-style masked label prediction).
- Aggregate neighbors using **simple mean pooling** (no relation awareness).
- Use a residual MLP stack for classification.

### Motivation
We need a CPU-safe, minimal reproduction of UniMP’s key idea: **labels as input** + **masked label prediction**. This serves as the foundation for further heterogeneous extensions.

### Results
Tested on Citation-network V1 (title classification):

| Model          | Accuracy | Macro-F1 | Micro-F1 |
|----------------|----------|----------|----------|
| Iter-1 UniMP | 70.20%    | 68.70%    | 69.40%    |

### Observations
- The model captures useful signal from masked labels.
- Performance is limited because author–paper–venue structure is ignored.

### Next Step
In Iteration 2 we will add **relation-aware propagation** to exploit heterogeneous edges.
