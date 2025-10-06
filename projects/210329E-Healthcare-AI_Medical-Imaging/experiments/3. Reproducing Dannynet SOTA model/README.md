# Reproducing DannyNet (SOTA) — Notes and Results

This implementation reproduces the DannyNet method described in the paper:

DannyNet: (see) https://www.arxiv.org/abs/2505.06646

This reproduction respectfully acknowledges the DacNet implementation as an implementation reference: https://github.com/dstrick17/DacNet

This repository contains the notebook `Reproduced_Dannynet.ipynb` which implements training, validation and testing for the DannyNet architecture and evaluation on the NIH Chest X-ray dataset.

## AUROC Comparison (Original vs DannyNet vs Reproduced DannyNet)

| Pathology           | Original CheXNet (2017) | DannyNet (paper)  | Reproduced DannyNet (this work) |
|---------------------|-------------------------:|----------------:|---------------------------------:|
| Atelectasis         | 0.809                    | 0.817           | **0.8181**                        |
| Cardiomegaly        | 0.925                    | **0.932**       | 0.9280                           |
| Consolidation       | **0.790**               | 0.783           | 0.7810                           |
| Edema               | 0.888                    | **0.896**       | 0.8782                           |
| Effusion            | 0.864                    | **0.905**       | 0.8975                           |
| Emphysema           | 0.937                    | **0.963**       | 0.9606                           |
| Fibrosis            | 0.805                    | 0.814           | **0.8216**                        |
| Hernia              | 0.916                    | **0.997**       | 0.9951                           |
| Infiltration        | **0.735**               | 0.708           | 0.6986                           |
| Mass                | 0.868                    | **0.919**       | 0.9047                           |
| Nodule              | 0.780                    | **0.789**       | 0.7736                           |
| Pleural Thickening  | **0.806**               | 0.801           | 0.7988                           |
| Pneumonia           | **0.768**               | 0.740           | 0.7209                           |
| Pneumothorax        | **0.889**               | 0.875           | 0.8831                           |

> Values are AUROC only. "DannyNet" numbers are taken from the DannyNet paper; "Reproduced DannyNet" numbers are produced by `Reproduced_Dannynet.ipynb` evaluation.


## Per-class F1 scores (DannyNet paper vs Reproduced)

| Disease              | DannyNet (paper) | Reproduced DannyNet (F1) |
|---------------------:|-----------------:|-------------------------:|
| Atelectasis          | **0.421**        | 0.4119                  |
| Cardiomegaly         | **0.532**        | 0.5149                  |
| Consolidation        | **0.226**        | 0.2222                  |
| Edema                | **0.286**        | 0.2460                  |
| Effusion             | **0.623**        | 0.5997                  |
| Emphysema            | **0.516**        | 0.5082                  |
| Fibrosis             | **0.127**        | 0.1261                  |
| Hernia               | **0.750**        | 0.6667                  |
| Infiltration         | 0.395            | **0.4018**              |
| Mass                 | 0.477            | **0.5000**              |
| Nodule               | **0.352**        | 0.3382                  |
| Pleural Thickening   | **0.258**        | 0.2453                  |
| Pneumonia            | **0.082**        | 0.0640                  |


## Summary metrics

| Metric | DannyNet (paper) | This work (Reproduced DannyNet) |
|-------:|------------------:|--------------------------------:|
| Loss   | **0.0416**        | 0.0419                          |
| AUC    | **0.8527**        | 0.8471                          |
| F1     | **0.3861**        | 0.3705                          |


## Conclusion

- The implemented DannyNet (this work) closely reproduces the paper's DannyNet performance; AUROC and F1 values are comparable for most classes. Some classes (e.g., Hernia, Emphysema) show very high AUROC in the reproduced run.
- Overall, DannyNet (paper) outperforms the original CheXNet on average AUC in the reported numbers; our reproduced DannyNet closely matches this improvement (Avg AUROC 0.8471 vs paper 0.8527).


### Reproducibility

- Notebook: `Reproduced_Dannynet.ipynb` (this folder)
- Data: NIH Chest X-ray dataset (Kaggle) — `https://www.kaggle.com/datasets/nih-chest-xrays/data`. Adjust `CONFIG['data_dir']` to point to your local/kaggle dataset.
- To reproduce: run the notebook end-to-end; the notebook saves best checkpoints under `models/` and prints final test statistics.

Notes on the environment used for reproduction:

- This run used the Kaggle environment (import the notebook into a Kaggle kernel). Set accelerator to GPU (T4) and enable 2x GPUs if available (T4 * 2). 
- Dataset: NIH Chest X-ray (link above). 
- Estimated runtime: approximately 10–11 hours on the Kaggle T4*2 setup for the full training/evaluation run.

## Research note

For downstream experiments and comparisons in this project, we will treat DannyNet as the working SOTA baseline. The primary reason is that DannyNet's dataloader creation and preprocessing steps are visible and somewhta closely reproducible from the implementation, which improves comparability and reduces ambiguity when evaluating new methods against a known pipeline.

