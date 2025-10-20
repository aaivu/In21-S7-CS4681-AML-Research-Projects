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