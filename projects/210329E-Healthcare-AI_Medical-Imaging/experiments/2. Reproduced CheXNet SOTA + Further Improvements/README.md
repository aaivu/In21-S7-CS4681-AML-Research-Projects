# CheXNet Reproduction with Additional Improvements (V1)

In this folder, I extended the original CheXNet reproduction by adding advanced data augmentations and a modified loss function.

## Data Augmentation

The original CheXNet paper only applied `RandomHorizontalFlip`.  
In this version, I applied the following augmentations:

```python
transforms.RandomHorizontalFlip(),
transforms.RandomRotation(degrees=15),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),  
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
````

## Loss Function

Instead of the original `BCEWithLogitsLoss`, I used **Focal Loss** to better handle class imbalance and focus on hard-to-classify examples.

## AUROC Comparison

| Pathology           | CheXNet (Paper) | Reproduced CheXNet (Original) | Reproduced CheXNet + Improvements (V1) |
| ------------------- | --------------- | ----------------------------- | -------------------------------------- |
| Atelectasis         | **0.8094**      | 0.7676                        | 0.7762                                 |
| Cardiomegaly        | **0.9248**      | 0.8836                        | 0.8921                                 |
| Effusion            | **0.8638**      | 0.8237                        | 0.8316                                 |
| Infiltration        | **0.7345**      | 0.6993                        | 0.6876                                 |
| Mass                | **0.8676**      | 0.8179                        | 0.8227                                 |
| Nodule              | **0.7802**      | 0.7593                        | 0.7609                                 |
| Pneumonia           | **0.7680**      | 0.6951                        | 0.7228                                 |
| Pneumothorax        | **0.8887**      | 0.8528                        | 0.8569                                 |
| Consolidation       | **0.7901**      | 0.7472                        | 0.7530                                 |
| Edema               | **0.8878**      | 0.8439                        | 0.8474                                 |
| Emphysema           | **0.9371**      | 0.9056                        | 0.9179                                 |
| Fibrosis            | 0.8047          | 0.8279                        | **0.8282**                             |
| Pleural\_Thickening | **0.8062**      | 0.7680                        | 0.7738                                 |
| Hernia              | 0.9164          | 0.9011                        | **0.9290**                             |

This table summarizes the per-class AUROC scores comparing the original CheXNet paper results, the reproduced CheXNet (baseline), and the version with additional improvements (V1).

## Notes

* The test set used is from the [NIH Chest X-ray Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) since the original CheXNet test set is not publicly available.
* The improvements focus on better data augmentation and handling class imbalance with focal loss, leading to slightly improved AUROC for several classes.

### Reproducibility / Environment

- This run was performed in the Kaggle environment (import the notebook into a Kaggle kernel). Set the accelerator to GPU (T4) and enable 2x GPUs if available (T4 * 2).
- Dataset: NIH Chest X-ray (link above).
- Estimated runtime: approximately 10–12 hours on the Kaggle T4*2 setup for the full training/evaluation run.

## Summary

- **Reproduced CheXNet (Original) — Mean AUC (14 classes): 0.8066**
- **Reproduced CheXNet + Improvements (V1) — Mean AUC (14 classes): 0.8143**

## Conclusion

The improved reproduction (V1) shows overall AUROC improvements for several pathologies compared to my reproduced baseline — the mean AUC increased to **0.8143**. However, despite these gains, the reproduced and improved models still do not consistently surpass the original CheXNet paper's reported AUROC values. Direct comparison remains limited because the original CheXNet test set is not publicly available; differences in test set composition, labeling, and preprocessing can change AUROC outcomes. The NIH dataset continues to serve as a consistent public benchmark for these experiments.

