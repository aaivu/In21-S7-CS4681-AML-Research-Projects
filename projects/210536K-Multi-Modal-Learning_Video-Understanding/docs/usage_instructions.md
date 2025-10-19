# TBT-Former Usage Instructions

This document provides detailed instructions for setting up, training, and evaluating TBT-Former on various datasets.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Setup](#dataset-setup)
   - [THUMOS14](#thumos14)
   - [EPIC-Kitchens 100 & ActivityNet](#epic-kitchens-100--activitynet)
3. [Running on Kaggle](#running-on-kaggle)
4. [Training](#training)
5. [Monitoring Training](#monitoring-training)
6. [Evaluation](#evaluation)
7. [Additional Resources](#additional-resources)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

**System Requirements:**

- A machine with CUDA support (GPU with at least 12GB memory is recommended)
- Python 3.x

**Python Dependencies:**

All necessary Python packages are listed in [`requirements.txt`](../requirements.txt). Install them using:

```shell
pip install -r requirements.txt
```

### Compiling NMS Module

Part of NMS is implemented in C++. The code must be compiled before use:

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

**Important:** The code should be recompiled every time you update PyTorch.

---

## Dataset Setup

### THUMOS14

#### Download Features and Annotations

- Download `thumos.tar.gz` (`md5sum 375f76ffbf7447af1035e694971ec9b2`) from:
  - [Box Link](https://uwmadison.box.com/s/glpuxadymf3gd01m1cj6g5c3bn39qbgr)
  - [Google Drive Link](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing)
- The file includes I3D features, action annotations, and external classification scores.

**Feature Details:**

- Extracted from two-stream I3D models pretrained on Kinetics
- Clips of `16 frames` at video frame rate (`~30 fps`)
- Stride of `4 frames`
- Results in one feature vector per `4/30 ~= 0.1333` seconds

For more information about THUMOS14, see the [THUMOS14 dataset documentation](../data/THUMOS14.md).

#### Unpack Features and Annotations

Unpack the file under `./data` (or create a symlink to `./data`). The folder structure should look like:

```
TBT-Former/
│   README.md
│   ...
│
└───data/
    └───thumos/
        └───annotations/
        └───i3d_features/
```

---

- For EPIC-Kitchens 100 & ActivityNet dataset setup, follow the same procedure as THUMOS14. Refer to the [EPIC-Kitchens 100 dataset documentation](../data/EPIC_Kitchens_100.md) for more details about the EPIC-Kitchens 100 dataset.
  For more information about ActivityNet, see the [ActivityNet dataset documentation](../data/ActivityNet.md).

---

## Running on Kaggle

If you want to run TBT-Former on Kaggle, there are some additional steps required due to Kaggle's environment limitations.

### Prerequisites for Kaggle

Kaggle doesn't permit installing different Python versions directly, but this code requires compatibility with TensorFlow 1.x. To work around this, you need to create a conda environment.

### Step 1: Set Up Conda Environment

#### Option A: Upload and Run the Setup Script (Recommended)

The easiest way to set up the environment on Kaggle is to use the provided [`kaggle-setup.sh`](../src/kaggle-setup.sh) script.

1. Download the [`kaggle-setup.sh`](../src/kaggle-setup.sh) file from the `src` folder
2. Upload it to your Kaggle notebook session (you can drag and drop it into the Kaggle file browser)
3. Run the script in a Kaggle code cell:

```python
!bash kaggle-setup.sh
```

This automated script will:

- Install Miniconda on the Kaggle instance
- Create a new conda environment named `myenv` with Python 3.8
- Install PyTorch 1.11 with CUDA 11.3 support
- Install torchvision and torchaudio
- Verify the installation

#### Option B: Manual Setup

Alternatively, you can manually execute the setup commands. Here's what the script does:

```bash
#!/bin/bash
set -e  # exit if any command fails

# Install Miniconda
wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
export PATH="/usr/local/bin:$PATH"
export PYTHONPATH="/usr/local/lib/python3.8/site-packages:$PYTHONPATH"

# Create conda env
conda create -y -n myenv python=3.8 -c conda-forge --override-channels

# Check env
conda info --envs
conda run -n myenv python --version

# Install PyTorch
conda install -y -n myenv pytorch=1.11 torchvision=0.12 torchaudio=0.11 cudatoolkit=11.3 -c pytorch -c conda-forge --override-channels

# Force correct CUDA wheels
conda run -n myenv python -m pip uninstall -y torch torchvision torchaudio || true
conda run -n myenv python -m pip install -U \
  "torch==1.11.0+cu113" "torchvision==0.12.0+cu113" "torchaudio==0.11.0+cu113" \
  --extra-index-url https://download.pytorch.org/whl/cu113

# Verify installation
conda run -n myenv python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Note:** The environment is named `myenv` in the script. You can modify the script to use a different name if preferred

### Step 2: Upload Source Code as Dataset

1. Navigate to the [`src`](../src/) folder in this repository
2. Compress the entire `src` folder into a `.zip` or `.tar.gz` file
3. Upload it to Kaggle as a dataset
4. Attach this dataset to your Kaggle notebook/script

### Step 3: Running Commands in Kaggle

**Important:** Kaggle doesn't permit you to activate your conda environment throughout the entire session. Instead, you must invoke the environment on a **per-command basis**.

#### Standard Command (Won't Work on Kaggle):

```shell
# This won't work on Kaggle
python ./train.py ./configs/tbtformer_thumos_i3d.yaml --output reproduce_tbtformer
```

#### Kaggle-Compatible Command (Use This Instead):

```shell
# Use this format on Kaggle
conda run -n myenv python -u ./train.py ./configs/tbtformer_thumos_i3d.yaml --output reproduce_tbtformer
```

**Command Format:**

```shell
conda run -n <environment_name> python -u <script> <arguments>
```

Where:

- `<environment_name>`: Your conda environment name (e.g., `myenv` if using the provided script)
- `python -u`: Runs Python in unbuffered mode (recommended for real-time output)
- `<script>`: The Python script to run
- `<arguments>`: Script arguments

### Example Commands for Kaggle

**Training:**

```shell
conda run -n myenv python -u ./train.py ./configs/tbtformer_thumos_i3d.yaml --output reproduce_tbtformer
```

**Evaluation:**

```shell
conda run -n myenv python -u ./eval.py ./configs/tbtformer_thumos_i3d.yaml ./ckpt/tbtformer_thumos_i3d_reproduce_tbtformer
```

**Compiling NMS Module:**

```shell
cd ./libs/utils
conda run -n myenv python setup.py install --user
cd ../..
```

### Tips for Kaggle Usage

1. **Session Persistence:** Save your checkpoints and results to Kaggle's output directory to persist them across sessions.

2. **GPU Selection:** Ensure you've enabled GPU acceleration in your Kaggle notebook settings (Kaggle provides free GPU access).

3. **Memory Management:** Monitor your memory usage, as Kaggle has memory limits. Consider reducing batch size if needed.

4. **Output Buffering:** The `-u` flag in `python -u` ensures unbuffered output, so you can see training progress in real-time.

5. **Environment Verification:** Before running training, verify your environment is set up correctly:
   ```shell
   conda run -n myenv python --version
   conda run -n myenv pip list
   ```

---

## Training

### Training on THUMOS14

Train TBT-Former with I3D features. This will create an experiment folder under `./ckpt` that stores the config, logs, and checkpoints.

```shell
python ./train.py ./configs/tbtformer_thumos_i3d.yaml --output reproduce_tbtformer
```

**Hardware Requirements:**

- Training requires ~5GB GPU memory
- Inference may require over 10GB GPU memory
- A GPU with at least 12GB memory is recommended

---

## Monitoring Training

You can monitor the training process in real-time using TensorBoard:

```shell
tensorboard --logdir=./ckpt/tbtformer_thumos_i3d_reproduce_tbtformer/logs
```

Replace the path with the appropriate checkpoint directory for your experiment.

---

## Evaluation

### Evaluating on THUMOS14

Evaluate the trained model:

```shell
python ./eval.py ./configs/tbtformer_thumos_i3d.yaml ./ckpt/tbtformer_thumos_i3d_reproduce_tbtformer
```

**Expected Results:**

- Average mAP: **68.0%**

Detailed results (mAP at different tIoU thresholds):

| Method         | 0.3  | 0.4  | 0.5  | 0.6  | 0.7  | **Avg**  |
| -------------- | ---- | ---- | ---- | ---- | ---- | -------- |
| **TBT-Former** | 82.5 | 79.0 | 72.4 | 60.6 | 45.3 | **68.0** |

---

## Additional Resources

- **Code Structure:** See [Code Overview](../src/README.md#code-overview) for details on the repository structure
- **Dataset Information:**
  - [THUMOS14 Documentation](../data/THUMOS14.md)
  - [ActivityNet Documentation](../data/ActivityNet.md)
  - [EPIC-Kitchens 100 Documentation](../data/EPIC_Kitchens_100.md)
- **Experiments:** See the [experiments folder](../experiments/README.md) for details on ablation studies and preliminary experiments

---

## Troubleshooting

### Common Issues

1. **NMS Compilation Errors:**

   - Ensure you have a compatible C++ compiler installed
   - Recompile the NMS module after any PyTorch update

2. **Out of Memory Errors:**

   - Reduce batch size in the configuration file
   - Ensure you have at least 12GB GPU memory
   - Close other GPU-intensive applications

3. **Dataset Loading Errors:**
   - Verify the folder structure matches the expected layout
   - Check that all files were extracted correctly
   - Verify file permissions

---

## Contact

For questions or issues, please contact:

- Thisara Rathnayaka (thisara.21@cse.mrt.ac.lk)
