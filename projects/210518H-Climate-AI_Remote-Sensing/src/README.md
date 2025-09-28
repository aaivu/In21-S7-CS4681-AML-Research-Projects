# Model Initialization

This document explains how to prepare the environment and run training / finetuning / evaluation for the models. It is intentionally concise and practical, follow the steps exactly and adjust the runtime flags to match your hardware.

**Key points:**
- Create and activate the conda environment `sat_env` (instructions below).  
- Configure `/data` and `/experiments` according to their README files before running anything.  
- Two model families are supported: `temporal` (time sequences) and `sentinel` (multi-spectral).  
- Use pretrained SatMAE weights for finetuning whenever compute is limited.

---

## 1. Environment setup (conda)

This repository contains an `env.yml` file that defines the required environment. The environment name used here is `sat_env`.

1. Ensure you have Miniconda / Anaconda installed and that `conda` is on your `PATH`.
2. From the repository root run:

```bash
# Create the environment from the env.yml file
conda env create -f env.yml

# Activate it
conda activate sat_env
```

---

## 2. Configure `/data` and `/experiments`

Before running any model:
1. Read and follow the README in `/data` to prepare dataset folders (`temporal` and `sentinel`) and metadata CSVs.
2. Read and follow the README in `/experiments` to create `experiments/temporal` and `experiments/sentinel` and to download the baseline weights (SatMAE weights) if you plan to finetune.

---

## 3. Models Overview

- `temporal` - integrates time sequences for the same geolocations (improves temporal dependency learning).
- `sentinel` - uses multi-spectral bands (beyond RGB) to improve spatial/ spectral representations.

Available **pretrain** backbone models:
- `mae_vit_base_patch16`
- `mae_vit_large_patch16`
- `mae_vit_large_patch16_samemask`
- `mae_vit_huge_patch14`

Available **finetune** backbone models:
- `vit_base_patch16`
- `vit_large_patch16`
- `vit_huge_patch14`

---

## 4. How to Run

### 4.1 Temporal Pretraining

Command:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=1 --master_port=1234 \
  main_pretrain.py \
  --batch_size 8 \
  --accum_iter 8 \
  --norm_pix_loss \
  --epochs 5 \
  --blr 1.5e-4 \
  --mask_ratio 0.75 \
  --input_size 224 \
  --patch_size 16 \
  --model mae_vit_large_patch16 \
  --model_type temporal \
  --dataset_type temporal \
  --train_path ../data/temporal/train_62classes.csv \
  --output_dir ../experiments/temporal/pretrain \
  --log_dir ../experiments/temporal/pretrain \
  --num_workers 4
```

**Flags Explained:**
- `--nproc_per_node`: number of processes (GPUs) per node. `1` means single-GPU.
- `--nnodes`: number of nodes (machines) in distributed run. Usually `1` for local runs.
- `--master_port`: port used by the distributed backend; choose an available port.
- `--batch_size`: per-process (per-GPU) batch size.
- `--accum_iter`: gradient accumulation iterations. Accumulating gradients allows a larger effective batch size without more GPU memory.
- `--norm_pix_loss`: use normalized pixel reconstruction loss (MAE-specific).
- `--epochs`: number of training epochs.
- `--blr`: base learning rate.
- `--mask_ratio`: fraction of input patches masked for MAE pretraining.
- `--input_size`: input image size (px).
- `--patch_size`: ViT patch size.
- `--model`: backbone model to use (see list above).
- `--model_type`, `--dataset_type`: set to `temporal` for temporal experiments.
- `--train_path`: path to the CSV listing training images.
- `--output_dir`, `--log_dir`: where checkpoints and logs will be written.
- `--num_workers`: dataloader workers.

**Effective batch size** = `batch_size * accum_iter * (#GPUs)` 

**Recommendation:** The provided example is tuned for a single RTX 4070 Ti Super (16GB). Adjust `batch_size`, `accum_iter`, `num_workers`, and `epochs` to match your memory and time budget.

**W&B:** To use Weights & Biases, run `wandb login` after activating `sat_env` and configure **W&B** into the project. Then append `--wandb <YOUR_PROJECT_NAME>` to the CLI.

---

### 4.2 Temporal Finetuning

Command:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=1 --master_port=1234 \
  main_finetune.py \
  --batch_size 4 \
  --accum_iter 8 \
  --epochs 5 \
  --blr 1e-3 \
  --layer_decay 0.75 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --model vit_large_patch16 \
  --model_type temporal \
  --dataset temporal \
  --finetune ../experiments/temporal/pretrain_fmow_temporal.pth \
  --train_path ../data/temporal/train_62classes_subset.csv \
  --test_path ../data/temporal/val_62classes_subset.csv \
  --output_dir ../experiments/temporal/finetune \
  --log_dir ../experiments/temporal/finetune \
  --dist_eval \
  --num_workers 4
```

**Flags Explained:**
- `--finetune`: path to pretrained weights (SatMAE baseline or your own). If omitted, finetuning will start from the specified model's random init.
- `--resume`: (alternative) resume training from a local checkpoint. Use `--resume <checkpoint.pth>` to continue a run.
- `--layer_decay`: layer-wise learning rate decay (useful for ViT fine-tuning).
- `--weight_decay`: L2 regularization.
- `--drop_path`: stochastic depth rate (regularization).
- `--reprob`: random erasing probability.
- `--mixup`, `--cutmix`: data augmentation mixing parameters.
- `--dist_eval`: run distributed evaluation (synchronizes metrics across ranks).
- `--model`: choose the finetuning backbone (note: `vit_large_patch16` must be used for fine-tuning when pretrained with `mae_vit_large_patch16`. Dimensions must match).

**Resume vs Finetune:**
- Use `--finetune <pretrained_weights>` to initialize from pretrained weights.
- Use `--resume <checkpoint>` to continue a previous finetuning run from a checkpoint.

---

### 4.3 Temporal Evaluation

Command:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=1 --master_port=1234 \
  main_finetune.py \
  --batch_size 128 \
  --model vit_base_patch16 \
  --model_type temporal \
  --dataset temporal \
  --resume ../experiments/temporal/finetune/checkpoint-X.pth \
  --train_path ../data/temporal/train_62classes.csv \
  --test_path ../data/temporal/val_62classes.csv \
  --output_dir ../experiments/temporal/evaluation \
  --log_dir ../experiments/temporal/evaluation \
  --dist_eval \
  --eval \
  --num_workers 4
```

- `--resume` loads the checkpoint to evaluate; `--eval` runs evaluation-only mode.
- Use `--batch_size` large for evaluation if GPU memory allows (no gradients).

---

### 4.4 Sentinel Pretraining

Command:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=1 \
  main_pretrain.py \
  --batch_size 32 \
  --accum_iter 8 \
  --blr 0.0001 \
  --epochs 50 \
  --input_size 96 \
  --patch_size 8 \
  --mask_ratio 0.75 \
  --model mae_vit_base_patch16 \
  --model_type group_c \
  --dataset_type sentinel \
  --dropped_bands 0 9 10 \
  --grouped_bands 0 1 2 6 \
  --grouped_bands 3 4 5 7 \
  --grouped_bands 8 9 \
  --train_path ../data/sentinel/train_subset.csv \
  --output_dir ../experiments/sentinel/pretrain \
  --log_dir ../experiments/sentinel/pretrain \
  --num_workers 4
```

**Bands handling:**
- `--dropped_bands`: list of band indices to drop (excluded from model input).
- `--grouped_bands`: group several bands into a single sequence input (repeat flag multiple times to create multiple groups).
- The Sentinel imagery here contains 13 bands; grouping lets you control how bands are fed into the model.

**Other notes:**
- `--spatial_mask` (optional): enable consistent spatial masking across channels (toggle this to evaluate alternative masking strategies).

---

### 4.5 Sentinel Finetuning

Command:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=1 \
  main_finetune.py \
  --batch_size 128 \
  --accum_iter 8 \
  --blr 0.0002 \
  --epochs 30 \
  --input_size 96 \
  --patch_size 16 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --model vit_base_patch16 \
  --model_type group_c \
  --dataset_type sentinel \
  --dropped_bands 0 9 10 \
  --train_path ../data/sentinel/train_subset.csv \
  --test_path ../data/sentinel/val_subset.csv \
  --output_dir ../experiments/sentinel/finetune \
  --log_dir ../experiments/sentinel/finetune \
  --finetune ../experiments/sentinel/pretrain-vit-base-e199.pth \
  --num_workers 4
```

- Change `--model` to a different backbone (e.g., ViT-Large) if you have enough compute.
- If you use ViT-Large pretraining weights, match the finetune `--model` name to avoid shape mismatches.
- `--dropped_bands` must match the pretraining setting.

---

## 5. Additional Notes

- **Compute tradeoffs:** Increase `accum_iter` to simulate a larger batch size when GPU memory is limited. Increase `num_workers` to accelerate loading (but watch for system RAM pressure).
- **Multi-GPU / cluster:** Set `--nproc_per_node` to GPUs-per-node and `--nnodes` to number of nodes, and ensure `MASTER_ADDR` / `MASTER_PORT` are available or controlled by the cluster launcher.
- **W&B integration:** `wandb login` after activating `sat_env`, then add `--wandb <PROJECT>` to the run command.
- **Logging & checkpoints:** Always set `--output_dir` and `--log_dir` to subfolders under `experiments/<dataset>/<run_name>`.
- **Use pretrained SatMAE weights** for finetuning when compute or time is limited, this yields much faster convergence and strong baselines.
- **Pretrain and Finetune models:** You must use the same model configurations used for pretraining during finetune (Pretrain: `mae_vit_large_patch16` → Finetune: `vit_large_patch16`).
- **`vanilla` runs:** To run the baseline (no temporal grouping or multi-spectral grouping), set `--model_type vanilla`.

## 6. Acknowledgement

Model configurations and setup are derived from the SatMAE research: [https://github.com/sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE)