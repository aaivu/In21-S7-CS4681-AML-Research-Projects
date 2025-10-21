# Experiments Initialization

This document describes how to organize, prepare, and use the `experiments` folder.  
All model runs including **training from scratch** and **finetuning runs** will be stored here in a hierarchical manner.  
Results, logs, and checkpoints will be kept under model-specific directories for reproducibility and clarity.

---

## Folder Setup

Create two subdirectories under `experiments` to separate runs for the two datasets:

```bash
mkdir -p temporal
mkdir -p sentinel

# Create directories for the updated configurations
mkdir -p temporal_v2
mkdir -p sentinel_v2
```

- `temporal/` → runs on the **fMoW Temporal dataset**
- `sentinel/` → runs on the **fMoW-Sentinel dataset**
- `temporal_v2/` → runs on the **Temporal V2 model**
- `sentinel_v2/` → runs on the **Sentinel V2 model**

---

## Training vs. Finetuning

- **Training (from scratch):** The model is initialized randomly and trained on the dataset. This requires large compute and full dataset access.
- **Finetuning:** Starts from **pretrained SatMAE model weights** and adapts them to the task.

**Recommendation:** Use finetuning if you have **limited compute resources**, since training full models from scratch is prohibitively expensive.

---

## Temporal - Pretrained Weights

Model weights are hosted on Zenodo: [https://zenodo.org/records/7369797](https://zenodo.org/records/7369797)

**Available weights:**
- Pretrain Temporal, ViT-L, 800 epochs
- Finetune Temporal, ViT-L, 25 epochs
- Pretrain Non-Temporal, ViT-L, 800 epochs
- Finetune Non-Temporal, ViT-L, 30 epochs

**Setup:**
```bash
cd temporal

# Pretrain weights (recommended starting point for finetuning)
wget https://zenodo.org/records/7369797/files/pretrain_fmow_temporal.pth

# Finetune weights (optional, useful for direct evaluation)
wget https://zenodo.org/records/7369797/files/finetune_fmow_temporal.pth
```

---

## Sentinel - Pretrained Weights

Model weights are hosted on Zenodo: [https://zenodo.org/records/7338613](https://zenodo.org/records/7338613)

**Available weights:**
- Pretrain, ViT-L, 200 epochs
- Finetune, ViT-L, 30 epochs (best accuracy at epoch 7)
- Pretrain, ViT-B, 200 epochs
- Finetune, ViT-B, 30 epochs (best accuracy at epoch 7)

**Setup (ViT-Base, recommended for limited compute):**
```bash
cd sentinel

# Pretrain weights
wget https://zenodo.org/records/7338613/files/pretrain-vit-base-e199.pth

# Optional: Finetune weights (best model at epoch 7)
wget https://zenodo.org/records/7338613/files/finetune-vit-base-e7.pth
```

If you have access to **larger compute**, you may download the **ViT-Large** weights from the same Zenodo record.<br>
**Be aware:** ViT-Large significantly increases memory and compute requirements.

---

## Temporal V2 - Pretrained Weights

Model weights are hosted on Zenodo: [https://zenodo.org/records/17277063](https://zenodo.org/records/7369797)

**Available weights:**
- Finetune Temporal, ViT-L, 5 epochs

**Setup:**
```bash
cd temporal_v2

wget https://zenodo.org/records/17277063/files/temporal-finetune-vit-large.pth
```

---

## Sentinel V2 - Pretrained Weights

Model weights are hosted on Zenodo: [https://zenodo.org/records/17277063](https://zenodo.org/records/7369797)

**Available weights:**
- Finetune Sentinel, ViT-B, 5 epochs

**Setup:**
```bash
cd sentinel_v2

wget https://zenodo.org/records/17277063/files/sentinel-finetune-vit-base.pth
```

**Note:** You can run evaluations with the V2 models to validate the results. Change the model run commands to use these checkpoint files.

---

## Final Expected Structure

After downloading the model weights, the `experiments` directory should look like:
```bash
/experiments
├── temporal
│   ├── pretrain_fmow_temporal.pth
│   └── finetune_fmow_temporal.pth
│
└── temporal_v2
│   └── temporal-finetune-vit-large.pth
│
└── sentinel
│   ├── pretrain-vit-base-e199.pth
│   └── finetune-vit-base-e7.pth
│
└── sentinel_v2
    └── sentinel-finetune-vit-base.pth
```
When you run training or finetuning experiments, **results, logs, and checkpoints** will be saved under these directories in subfolders named by the runtime configuration.

---

## Acknowledgements

Baseline model weights are accessed from the **SatMAE repository:** [https://github.com/sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE)