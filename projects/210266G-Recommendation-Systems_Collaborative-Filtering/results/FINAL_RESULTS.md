# Final Model Performance Comparison

This document compares the performance of the final ContentGCN model against the LightGCN baseline on the cleaned test dataset.

| Model                 | Test Recall@20 | Test NDCG@20 | Relative Improvement (Recall) |
| :-------------------- | :------------- | :----------- | :---------------------------- |
| LightGCN (Baseline)   | 0.1373         | 0.3125       | -                             |
| **ContentGCN (Ours)** | **0.2393** | **0.3169** | **+74.3%** |


- **Experiment Folder(baseline):** `LightGCN_dim64_layers3_lr0.001_reg0.0001_20251005-153530`
- **Experiment Folder(ContentGCN):** `ContentGCN_dim64_layers3_lr0.001_reg0.0001_20251005-151828`

## Analysis

The final ContentGCN model shows a substantial **74.3% relative improvement in recall** over the baseline LightGCN model, while also slightly improving the NDCG. This indicates that integrating content features through the proposed hybrid, gated architecture allows the model to recommend a significantly larger portion of relevant items for users without sacrificing ranking quality. This is an extremely positive outcome and a strong central result for the research paper.