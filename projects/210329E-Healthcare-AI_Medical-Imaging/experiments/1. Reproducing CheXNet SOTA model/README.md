# Experiment 1 - Original Paper Version - CheXNet Reproduction

In here, I attempted to reproduce CheXNet as described in the original research paper:  

**CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning**  
https://doi.org/10.48550/arXiv.1711.05225

I used the hyperparameters reported in the paper. However, there was a limitation when comparing the AUROC scores for each class because the original CheXNet test set is not publicly available.  

As a remedy, I used the NIH Chest X-ray dataset available on Kaggle for testing:  
https://www.kaggle.com/datasets/nih-chest-xrays/data

## AUROC Comparison (Original vs. Reproduced Baseline)

| Pathology           | CheXNet (Paper) | Reproduced CheXNet (Original) |
|--------------------|----------------|-------------------------------|
| Atelectasis        | 0.8094         | 0.7676                        |
| Cardiomegaly       | 0.9248         | 0.8836                        |
| Effusion           | 0.8638         | 0.8237                        |
| Infiltration       | 0.7345         | 0.6993                        |
| Mass               | 0.8676         | 0.8179                        |
| Nodule             | 0.7802         | 0.7593                        |
| Pneumonia          | 0.7680         | 0.6951                        |
| Pneumothorax       | 0.8887         | 0.8528                        |
| Consolidation      | 0.7901         | 0.7472                        |
| Edema              | 0.8878         | 0.8439                        |
| Emphysema          | 0.9371         | 0.9056                        |
| Fibrosis           | 0.8047         | 0.8279                        |
| Pleural_Thickening | 0.8062         | 0.7680                        |
| Hernia             | 0.9164         | 0.9011                        |
 
## Summary

**Mean AUC (14 classes) — Reproduced CheXNet (Original): 0.8066**

## Conclusion

The reproduced CheXNet model achieves a mean AUC of **0.8066** across the 14 pathologies when evaluated on the NIH Chest X-ray dataset. Per-class AUROC scores are generally slightly lower than the original CheXNet paper's reported values — in many cases by a few percentage points — though some classes (e.g., Fibrosis) show comparable or slightly better performance.

Because the original CheXNet test set used in the paper is not publicly available, direct one-to-one comparison is limited. Differences in test set composition, labeling, and preprocessing can substantially affect AUROC numbers; therefore these reproduced scores should be interpreted as an approximate baseline rather than a definitive replication of the original results.

For a stronger comparative study, access to the original test set or a carefully matched external test set would be necessary. Until then, the public NIH dataset provides a useful, consistent benchmark for measuring relative performance and reproducibility.

### Reproducibility / Environment

- This run was performed in the Kaggle environment (import the notebook into a Kaggle kernel). Set the accelerator to GPU (T4) and enable 2x GPUs if available (T4 * 2).
- Dataset: NIH Chest X-ray (link above).
- Estimated runtime: approximately 10–11 hours on the Kaggle T4*2 setup for the full training/evaluation run.

