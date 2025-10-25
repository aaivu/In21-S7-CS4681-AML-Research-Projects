# Student Usage Instructions

[Instructions will be loaded from template]
# DPCRN-TwoStage (MiniVCTK × DEMAND)

Research-ready implementation of a lightweight DPCRN variant with:

- **Spectral Compression Mapping (SCM)**
- **Two-stage enhancement** (Stage-1 magnitude mask → Stage-2 complex RI refinement)

This README targets a **local Git workflow** (Linux/macOS/Windows). No Colab steps.

---

## 1) Repo Layout

dpcrn_project/
├── configs/
│ └── base_config.yaml
├── data/
│ ├── extract_demand_zips.py
│ └── prepare_dataset.py
├── experiments/
│ └── train.py
├── requirements.txt
├── scripts/
│ └── download_from_gdrive.py
└── src/
├── dataset.py
├── loss.py
├── metrics.py
├── model/
│ ├── dpcrn_two_stage.py
│ ├── dual_path_rnn.py
│ └── spectral_compression.py
└── utils.py

## 2) Quickstart

### 2.1 Clone and set up environment

```bash
git clone https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd projects/210112H-NLP_Speech-Enhancement

conda create -n dpcrn python=3.10 -y
conda activate dpcrn
pip install -r requirements.txt
```

## 3) Datasets

We use two datasets:

MiniVCTK (clean speech subset)

DEMAND_noise (selected 48 kHz environments)

### 3.1 Download via Google Drive (script)

We provide a script that pulls your exact Drive files and extracts them under data/:

python scripts/download_from_gdrive.py --root ./data
It creates:

/data/MiniVCTK/...
/data/DEMAND_noise/...
If you already have the datasets, place them at the same locations.

Expected DEMAND layout example:
data/DEMAND_noise/DKITCHEN_48k/DKITCHEN/ch01.wav ... ch16.wav

## 4) Configuration

Minimal configs/base_config.yaml (edit to taste):

yaml

# Audio / STFT

sample_rate: 48000
n_fft: 1200
hop_length: 600
window: hann

# SCM / model

scm:
n_bins: 601 # n_fft//2 + 1
compressed_bins: 256
fixed_bins: 64

model:
enc_channels: [16, 32, 48] # conv encoder
dp_hidden: 127 # DPRNN hidden size

# Training

batch_size: 2
epochs: 3
lr: 1.0e-4
seed: 1337

# Data roots (relative or absolute)

paths:
clean_root: ./data/MiniVCTK
noise_root: ./data/DEMAND_noise

# Output

out_dir: ./outputs

## 5) Training

From the repo root:

python -m experiments.train

The trainer uses configs/base_config.yaml and prints progress. Modify epochs/lr in the YAML or add simple CLI flags in train.py if you want overrides.

Outputs:

outputs/
logs/
checkpoints/
dpcrn_two_stage_ep<...>.pt

## 6) Evaluation

From the repo root:

python /experiments/eval_metrics.py --num_samples 120 --seed 42
python /experiments/results.py
