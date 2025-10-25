# Usage Instructions

## Overview
This document provides instructions for using the Network Intrusion Detection System with Attack-Specialized Deep Learning models.

## Prerequisites

### System Requirements
- Python 3.8+
- TensorFlow 2.x
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn (for visualization)

### Hardware Requirements
- Minimum 8GB RAM
- GPU recommended for faster training (optional for inference)

## Installation

1. Clone the repository
2. Navigate to the project directory:
```bash
cd projects/210144G-Cybersecurity-AI_Threat-Detection
```
3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Place the NSL-KDD dataset files in the `data/` directory:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`
   - `KDDTest-21.txt` (optional, for generalizability testing)

## Model Training

### Individual Attack-Specific Models
Run the Jupyter notebooks in the `experiments/` directory to train individual models:

1. **DoS Detection**: `ndd_DoS_.ipynb`
2. **Probe Detection**: `ndd_probe_.ipynb`
3. **R2L Detection**: `ndd_R2L_.ipynb`
4. **U2R Detection**: `ndd_U2R_.ipynb`

These notebooks will save trained models to the `base_models/` directory.

### Meta-Classifier Training
After training individual models, train the meta-classifiers for ensemble methods:

```bash
# Run from project root directory
python train_meta.py
```

This will create meta-classifier files in the `meta_models/` directory.

## Running Evaluations

### Full Ensemble Evaluation
```bash
# Run from project root directory
python run_ensemble.py
```

### Sample-based Evaluation
```bash
python run_ensemble.py --sample-size 1000
```

### Generalizability Testing
```bash
python run_ensemble.py --test-dataset KDDTest-21
```

### Alternative: Run from src directory
If you prefer to run from the src directory:
```bash
cd src
python main.py --sample-size 1000
```

## Output

Results will be saved in the `results/` directory and displayed in the terminal, including:
- Individual model performance metrics
- Ensemble method comparisons
- Confusion matrices
- Classification reports

## Troubleshooting

### Getting Help

- Check the individual notebook outputs for training issues
- Verify file paths in `src/config.py`
- Ensure you're running scripts from the project root directory


## Recommended Workflow

1. **Setup**: Install dependencies and place data files
2. **Train Models**: Run all Jupyter notebooks in `experiments/`
3. **Train Meta-Classifiers**: `python train_meta.py`
4. **Evaluate**: `python run_ensemble.py --sample-size 1000`
5. **Full Evaluation**: `python run_ensemble.py`