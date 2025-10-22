# Usage Instructions

This guide provides a step-by-step procedure to set up the environment, generate datasets, preprocess data, train both the baseline and physics-aware ART models, and evaluate their performance.

---

## 1. Environment Setup

### Prerequisites
- **Python:** 3.10 or above  
- **CUDA:** 11.8+ (for GPU acceleration)  
- **Git:** installed and added to PATH  

### Clone Only This Project Folder (Using Git Sparse-Checkout)

Since the repository contains multiple projects, you can selectively clone only this project’s directory using **Git sparse-checkout**:

```bash
# Clone the repository without checking out files
git clone --no-checkout https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd In21-S7-CS4681-AML-Research-Projects

# Enable sparse-checkout mode
git sparse-checkout init --cone

# Specify the project folder to include
git sparse-checkout set projects/210460V-Space-AI_Autonomous-Spacecraft

# Checkout the main branch
git checkout main
```
Only the contents of the specified project folder will now be downloaded.

### Create a Virtual Environment
```bash
python -m venv venv
```

Activate it:
- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **Linux/Mac**
  ```bash
  source venv/bin/activate
  ```

### Install Dependencies
```bash
pip install -r requirements.txt
```

If a requirements file is missing, use:
```bash
pip install torch torchvision torchaudio tqdm numpy scipy matplotlib pandas
pip install transformers accelerate seaborn
```

### Verify Installation
```bash
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import transformers; print('Transformers:', __import__('transformers').__version__)"
```

---

## 2. Dataset Generation

The dataset is generated using **convex optimization** for orbital rendezvous trajectories.  
Each trajectory consists of:
- **ROE states (6D)**  
- **RTN states (6D)**  
- **Actions (Δv in RTN frame)**  
- **Auxiliary parameters:** time, orbital elements, reward-to-go (RTG), constraint-to-go (CTG)

### Run Dataset Generator
```bash
python src/dataset-generation/generate_data_art_physicsaware.py
```

### Output
- Directory: `dataset-seed-{seed}/`
  - `dataset-rpod-cvx.npz`
  - `dataset-rpod-cvx-param.npz`

*Tip:* Modify parameters like `TARGET_SAMPLES`, `RNG_SEED`, or `HORIZON_GRID` at the top of the generator code file to customize dataset size and range.

---

## 3. Data Preprocessing

### Normalize Dataset
> Replace `dataset-seed-2030` with the actual directory containing your dataset.

```bash
python src/dataset-generation/preprocess_art_physicsaware_norm.py     --data_dir dataset-seed-2030
```

If using saved scaling factors:
```bash
# Give your actual paths here:
python src/dataset-generation/preprocess_art_physicsaware_norm_from_scaler.py     --data_dir dataset-seed-2030     --scaler_path scaler_stats.npz
```
> Replace `dataset-seed-2030` with the actual directory containing your dataset, and `scaler_stats.npz` with the actial path of scaler file.

### Output
- `torch_states_rtn_cvx_norm.pth`
- `torch_actions_cvx_norm.pth`
- `torch_rtgs_cvx_norm.pth`
- `torch_ctgs_cvx_norm.pth`
- `torch_oe_cvx_norm.pth`

These files are ready for model training and validation.

---

## 4. Training the Models

### (A) Train Baseline ART

> Replace `processed_data` with the actual directory containing your preprocessed dataset.

```bash
# Give actual data driectory path:
python src/train/main_train_baseline.py     --data_dir processed_data     --seed 42     --epochs 3     --batch_size 8
```

### (B) Train Physics-Aware ART

Before starting the training, configure the following hyperparameters in the code file (`src/train/main_train_physicsaware.py`):

| Parameter | Description | Value |
|------------|--------------|--------|
| `lambda_dyn` | Initial physics-consistency weight | `1e-4` |
| `alpha_roll` | Initial rollout loss weight | `1e-5` |
| `lambda_dyn_max` | Maximum value for physics weight scheduler | `0.01` |
| `alpha_roll_max` | Maximum value for rollout weight scheduler | *(set desired value, e.g., `0.001`)* |
| `ema_decay` | Exponential moving average decay factor | `0.999` |
| `use_ema` | Enable EMA for stable validation | `True` |

---

### Run the Training Script

> Replace `processed_data` with the actual directory containing your preprocessed dataset.

```bash
python train/main_train_physicsaware.py \
    --seed 42 \
    --data_dir processed_data \
    --save_dir saved_files/checkpoints/physicsaware \
    --epochs 3 \
    --batch_size 8
```

✅ *Tip:* EMA (Exponential Moving Average) improves stability — enabled by default in physics-aware training.

---

## 5. Testing and Model Comparison

Before running the evaluation script, make sure to configure the following paths in the code file (`eval_art_vs_physicsaware.py`):

| Variable | Description | Example Path |
|-----------|--------------|---------------|
| `data_dir` | Directory containing the **preprocessed dataset** | `os.path.join(root_folder, "dataset")` |
| `ckpt_dir` | Directory containing the **Physics-Aware ART** checkpoint | `os.path.join(root_folder, "saved_files/checkpoints/physicsaware")` |
| `baseline_ckpt_dir` | Directory containing the **Baseline ART** checkpoint | `os.path.join(root_folder, "saved_files/checkpoints/baseline")` |

Define the checkpoint file paths:

```python
baseline_ckpt = os.path.join(baseline_ckpt_dir, "baseline_art.pt")
physics_ckpt  = os.path.join(ckpt_dir, "best_model.pt")
```

### Compare Baseline vs Physics-Aware Models

Once the paths are set, run the evaluation script:
```bash
python src/train/eval_art_vs_physicsaware.py
```

This will compare both models using validation metrics such as state prediction loss, control loss, physics residuals, and long-horizon MSE.
### Output Metrics
- `total`, `state`, `action`, `physics_res`
- `MSE@10`, `MSE@50`, `MSE@100`
- `Feasibility_Ratio`

Example output:
```
PhysicsAware_ART:
   total = 6.90e-01
   state = 3.82e-05
   action = 6.90e-01
   physics_res = 2.53e-01
   MSE@10 = 8.86e+02
   MSE@50 = 5.35e+03
   MSE@100 = nan
   Feasibility_Ratio = 0.00e+00
```

---

## 7. Recommended Configurations

| Parameter | Symbol | Typical Value |
|------------|---------|----------------|
| Learning Rate | η | 3e-5 |
| Batch Size | B | 8 |
| Epochs | E | 3 |
| λ_dyn (Physics Weight) | λ_dyn | 1e-4 → 1e-3 |
| α_roll (Rollout Weight) | α_roll | 1e-5 → 1e-4 |
| Rollout Horizon | H | 3 |
| EMA Decay | β_ema | 0.999 |
| Gradient Clip | ‖∇‖_max | 0.5 |

---