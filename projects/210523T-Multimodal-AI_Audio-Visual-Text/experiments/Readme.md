# OmniQ – Experiments

This folder contains experiment outputs (checkpoints, logs, plots) for the **OmniQ** project.  
Each subfolder corresponds to one training run with a specific fusion method / fine-tuning regime.

## Pre-flight checklist

1) **Activate environment**
- Windows (PowerShell):
  ```powershell
  cd "C:\Users\Hashini\Desktop\Sem 7\Adavanced Machine Learning\Research Assignment\OmniQ"
  .\.venv\Scripts\activate
  ```

2) **Dataset in place**  
   `data/UCF101/videos/…` and `data/UCF101/splits/…` (or `ucfTrainTestlist/…`) must exist and match the official split-1 lists.

3) **Configs present**  
   These commands assume the repo has:
    - `configs/pretrain_mvp.yaml`
    - `configs/finetune_ucf101.yaml`  
      (This finetune config supports toggling **fusion** (omnivore baseline / transformer / mamba) and **LoRA** via keys inside the YAML. If the YAMLs are split per variant, use the closest matching file and keep the run names shown below.)



## 1) `pretrain_mvp` — Dual-masked warm-up (video MSE + text MLM)

**Goal:** Warm up the fusion module using (i) video-time feature regression and (ii) MLM with Qwen2.5-Omni tokenizer/embeddings.  

**Backbone:** Frozen Swin-T (per-frame)

- Windows (PowerShell):
  ```powershell
  python -m omniq.train.pretrain --config configs/pretrain_mvp.yaml
  ```


**Output:** `pretrain_mvp/` (fusion warm-start checkpoint, logs, pretrain curves)

---

## 2) `omnivore_ucf101_t` — Visual baseline (temporal avg / Omnivore-style)

**Goal:** Strong, simple baseline with 2D Swin per-frame features + temporal average pooling + linear head (no temporal transformer / mamba).  

**Backbone:** Frozen Swin-T (typical) or as per config

- Windows:
  ```powershell
  python -m omniq.train.finetune --config configs/finetune_ucf101.yaml
  ```

> In the YAML, set the fusion to “baseline/avg” (or equivalent), and `train.lora: false`.  
> Ensure `train.run_name: omnivore_ucf101_t` so outputs land in this folder.

**Output:** `omnivore_ucf101_t/`

---

## 3) `omnivore_ucf101_t_smoke` — Smoke test (fast, tiny run)

**Goal:** Quick sanity check (few steps/epochs, small batch) to validate dataloading, shapes, loss wiring.  
 
**Backbone:** Frozen Swin-T

- Windows:
  ```powershell
  python -m omniq.train.finetune --config configs/finetune_ucf101.yaml
  ```


> In the YAML, set `train.max_steps` (or `epochs`) very small, reduce `batch_size`, and set `train.run_name: omnivore_ucf101_t_smoke`.

**Output:** `omnivore_ucf101_t_smoke/`

---

## 4) `omniq_transformer_ucf101_t_lora` — Thin Transformer fusion + LoRA

**Goal:** Replace temporal average with a **2-layer Transformer** fusion (8 heads), adapt only LoRA + classifier.  
 
**Backbone:** Frozen Swin-T  
**LoRA:** ✅ on fusion + classifier

- Windows:
  ```powershell
  python -m omniq.train.finetune --config configs/finetune_ucf101.yaml
  ```

> In the YAML: `model.fusion: transformer`, `train.lora: true` (e.g., `rank: 8, alpha: 16`).  
> Set `train.run_name: omniq_transformer_ucf101_t_lora`.

**Output:** `omniq_transformer_ucf101_t_lora/`

---

## 5) `omniq_mamba_ucf101_t` — Mamba fusion (no LoRA)

**Goal:** Swap temporal fusion to **bidirectional Mamba** (depth=2) and train without LoRA (full fusion params).  
 
**Backbone:** Frozen Swin-T  
**LoRA:** ❌

- Windows:
  ```powershell
  python -m omniq.train.finetune --config configs/finetune_ucf101.yaml
  ```

> In the YAML: `model.fusion: mamba`, `train.lora: false`, `fusion.depth: 2`, `mamba.bidirectional: true`.  
> Set `train.run_name: omniq_mamba_ucf101_t`.

**Output:** `omniq_mamba_ucf101_t/`

---

## 6) `omniq_mamba_ucf101_t_lora` — Mamba fusion + LoRA (recommended)

**Goal:** Same as above but with **LoRA adapters** only on the fusion (and classifier).  

**Backbone:** Frozen Swin-T  
**LoRA:** ✅ (`rank=8`, `alpha=16` by default)

- Windows:
  ```powershell
  python -m omniq.train.finetune --config configs/finetune_ucf101.yaml
  ```
- Linux/macOS:
  ```bash
  python -m omniq.train.finetune --config configs/finetune_ucf101.yaml
  ```

> In the YAML: `model.fusion: mamba`, `train.lora: true` (rank/alpha as desired), `mamba.bidirectional: true`.  
> Set `train.run_name: omniq_mamba_ucf101_t_lora`.

**Output:** `omniq_mamba_ucf101_t_lora/`

---

## Optional: warm-start fine-tuning from `pretrain_mvp`

If the finetune config supports it, point to the pretrain fusion checkpoint:

```yaml
# inside configs/finetune_ucf101.yaml
train:
  init_from_pretrain: "pretrain_mvp/checkpoints/fusion_pretrain.pt"  # example path
  freeze_backbone: true
  lora: true  # for _lora runs
```

Then rerun the finetune command for the desired variant (Transformer or Mamba).

---

## What ’ll find in each run folder

```
<RUN_NAME>/
├─ checkpoints/
│  ├─ last.pt
│  ├─ best.pt
│  └─ (optional) fusion_pretrain.pt
├─ logs/
│  ├─ events.tfevents...        # TensorBoard
│  └─ train.log                 # plain text log
├─ cfg/
│  └─ resolved.yaml             # frozen copy of the used config
├─ metrics.json                 # scalar summary (acc, loss, latency, VRAM, params)
└─ plots/                       # optional: per-run plots
```

- **TensorBoard**
    - Windows:
      ```powershell
      tensorboard --logdir runs
      ```

---

## Reproducing the *exact* folders  have

If the training scripts do not accept CLI overrides for the run name, set it **inside the YAML** before launching:

```yaml
train:
  run_name: omnivore_ucf101_t           # or any of the names below
  # …
model:
  fusion: baseline                       # baseline | transformer | mamba
  # …
lora:
  enabled: false                         # true for *_lora runs
  rank: 8
  alpha: 16
fusion:
  depth: 2
mamba:
  bidirectional: true
```

Then run the generic command:

```powershell
# Windows
python -m omniq.train.finetune --config configs/finetune_ucf101.yaml
```

---

## Quick troubleshooting

- **Accuracy very low (e.g., 3–10%)**  
  Try: (1) sanity overfit 100 clips; (2) verify `T=32, stride=2` actually reaches fusion; (3) disable heavy aug for a run; (4) confirm LoRA is applied to fusion only; (5) warm-start from `pretrain_mvp`.

- **OOM / VRAM spikes**  
  Reduce batch size, set `AMP: true`, verify no attention KV tensors in Mamba fusion (no transformer layers accidentally enabled).

- **Path issues on Windows**  
  Use raw strings or quotes; keep all paths relative to the repo root.

---

### Commands Summary 

```powershell
# Windows (PowerShell, after activating .venv)
python -m omniq.train.pretrain --config configs/pretrain_mvp.yaml
python -m omniq.train.finetune --config configs/finetune_ucf101.yaml   # set YAML to: omnivore_ucf101_t
python -m omniq.train.finetune --config configs/finetune_ucf101.yaml   # set YAML to: omnivore_ucf101_t_smoke
python -m omniq.train.finetune --config configs/finetune_ucf101.yaml   # set YAML to: omniq_transformer_ucf101_t_lora
python -m omniq.train.finetune --config configs/finetune_ucf101.yaml   # set YAML to: omniq_mamba_ucf101_t
python -m omniq.train.finetune --config configs/finetune_ucf101.yaml   # set YAML to: omniq_mamba_ucf101_t_lora
```


> If  prefer separate YAMLs per run, create small files (e.g., `configs/finetune_ucf101_mamba_lora.yaml`) that set the right `train.run_name`, `model.fusion`, and `lora.enabled`, and call the same command pointing to that file.
