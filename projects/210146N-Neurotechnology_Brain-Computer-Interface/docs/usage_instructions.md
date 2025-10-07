# Usage Instructions

## eeg-expy Installation Guide: Recommended Virtual Environment Setup

It is **highly recommended** to use a virtual environment (**conda** or **virtualenv**) for installing **eeg-expy**.

If Python is not already installed, obtain the latest **miniconda** version for your operating system from the official documentation: https://docs.conda.io/en/latest/miniconda.html.

The following commands guide through downloading the repository, creating, and activating a virtual environment.

### Conda Installation

**Windows, Linux, or macOS (Intel)**
```
git clone https://github.com/NeuroTechX/eeg-expy

cd eeg-expy

# Specify newer python than 3.8 version if needed.
conda create -v -n eeg-expy-full python=3.8

conda activate eeg-expy-full

# install only necessary dependencies
conda env update -f environments/eeg-expy-full.yml
```
**macOS arm64 (M1, M2, etc.)**
```
# clone the repo
git clone https://github.com/NeuroTechX/eeg-expy

# navigate to the repo
cd eeg-expy

# for audio to be supported, osx-64 runtime is currently required,
# drop the '--platform osx-64' parameter if audio is not needed, to use the native runtime.
# Specify newer python than 3.8 version if needed.
conda create -v --platform osx-64 -n eeg-expy-full python=3.8

# activate the environment
conda activate eeg-expy-full

# install only necessary dependencies
conda env update -f environments/eeg-expy-full.yml
```
### Virtualenv Installation

**Windows**
```
mkdir eegnb_dir

python3 -m venv eegnb-env

git clone https://github.com/NeuroTechX/eeg-expy

eegnb-env\Scripts\activate.bat

cd eeg-expy

pip install -e .
```
**Linux or macOS**
```
mkdir eegnb_dir

python3 -m venv eegnb-env

git clone https://github.com/NeuroTechX/eeg-expy

source eegnb-env/bin/activate

cd eeg-expy

pip install -e .
```
### Post-Installation Step (Jupyter Kernel)

For some operating systems, the following command is necessary to make the new eeg-expy environment available from the Jupyter Notebook landing page:

```
python -m ipykernel install --user --name eeg-expy
```

Check **https://neurotechx.github.io/EEG-ExPy** for further details.
And experiments are available at the same documentation, that can be run with the above mentioned setup