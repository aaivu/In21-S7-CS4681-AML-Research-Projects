# AI Evaluation:Task Completion - 210706H

## Student Information

- **Index Number:** 210706H
- **Research Area:** AI Evaluation:Task Completion
- **GitHub Username:** @JaneeshaJ2001
- **Email:** janeesha.21@cse.mrt.ac.lk

# CAFE: A Context-Aware and Fairness-Weighted Framework for Toxicity Evaluation in Language Models

This repository contains the official implementation for the paper "CAFE: A Context-Aware and Fairness-Weighted Framework for Toxicity Evaluation in Language Models." It provides the code to train, evaluate, and reproduce our transparent "glass-box" toxicity evaluator and audit the commercial Perspective API.

## Overview

The de facto standard for evaluating toxic degeneration in LLMs, the RealToxicityPrompts (RTP) benchmark, relies on the flawed, black-box Perspective API. This classifier suffers from systemic bias, context insensitivity, and non-stationarity, undermining reproducible science.

This project introduces CAFE, a transparent framework built by fine-tuning a state-of-the-art DeBERTa-v3-large model on the human-annotated Jigsaw Unintended Bias dataset. CAFE employs a multi-task learning objective and a fairness-weighting scheme to explicitly disentangle genuine toxicity from the mere mention of sensitive identity terms. The final artifact is a robust, reproducible, and demonstrably fairer evaluation tool used to perform a direct comparative audit of the Perspective API.


## ✨ Key Features

- **State-of-the-Art Model**: Utilizes microsoft/deberta-v3-large for superior nuance and context understanding.

- **Context-Aware Training**: Employs a multi-task learning head to predict 7 toxicity subtypes simultaneously, forcing the model to learn richer text representations.

- **Fairness-Weighted Loss**: Implements a custom sample weighting scheme that up-weights critical examples (BPSN/BNSP cases) to directly mitigate bias against identity groups.

- **Robust Ensembling**: Uses a 5-fold cross-validation strategy with a Power-3.5 weighted average to produce a stable and high-performing final model.

- **Comprehensive Bias Audit**: Includes the full suite of Jigsaw fairness metrics (Overall AUC, Subgroup AUC, BPSN AUC, BNSP AUC) to rigorously evaluate model performance.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd 210706H-AI-Evaluation_Task-Completion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## 2. Prerequisites: Kaggle API Setup

This project requires the Jigsaw dataset, which is downloaded via the Kaggle API.

1. Create a Kaggle account at https://www.kaggle.com and Visit https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules, accept the rules for the Jigsaw Unintended Bias Competition.

2. Generate an API Token from your Kaggle account page (Go to https://www.kaggle.com/account -> Create New API Token). This will download a kaggle.json file.

3. Place the credentials file in the correct location (~/.kaggle/kaggle.json on Linux/macOS).


## 3. Step-by-Step Execution

Execute the scripts in the src/ directory in order.

```bash
# Step 1: Download and preprocess the Jigsaw dataset (~30-40 minutes)
python src/preprocessing.py

# Step 2: Train the 5-fold ensemble (~10-12 hours on an A100 GPU)
# This is the most computationally intensive step.
python src/train.py

# Step 3: Run intrinsic evaluation on the Jigsaw test set (~30-60 minutes)
# This generates the results for Table II in the paper.
python src/evaluate.py

# Step 4: Run the extrinsic audit on RealToxicityPrompts (~1-2 hours)
# NOTE: This requires setting up your Perspective API key first (see below).
python src/run_rtp_audit.py

# Step 5: Generate all figures and tables for the paper
python src/generate_results.py
```

## 4. 💬 Interactive Demo

After training is complete, you can use the inference.py script to interact with your trained CAFE model.

```bash
python src/inference.py
```

The script will load the 5-fold ensemble and provide a prompt where you can enter text to get a full toxicity analysis.


## 🔧 Perspective API Setup
To run the full comparative audit in run_rtp_audit.py, you must use your own Perspective API key.

1. Obtain an API key by following the instructions at the [Perspective API website](https://perspectiveapi.com/).

2. Install the client library: 
   ```bash
   pip install google-api-python-client
   ```

3. Add your key to the designated variable inside the src/run_rtp_audit.py script.


## Project Structure
```
210706H-AI-Evaluation:Task-Completion/
├── src/                     # Source code
│   ├── preprocessing.py     # Data loading and text preprocessing
│   ├── DeBERTaForToxicity.py  # Model architecture
│   ├── JigsawDataset.py     # Custom dataset and sample weighting
│   ├── train.py             # Training with 5-fold CV
│   ├── jigsaw_metrics.py    # Bias-aware evaluation metrics
│   ├── evaluate.py          # Model evaluation and ensembling
│   ├── run_rtp_experiment.py  # RealToxicityPrompts experiment
│   └── generate_results.py  # Generate figures and tables
├── data/                    # Datasets
│   ├── raw/                 # Original datasets
│   ├── processed/           # Processed datasets
│   └── augmented/           # Augmented datasets
├── experiments/             # Experiment configurations
│   ├── configs/             # YAML configuration files
│   └── run_experiments.py   # Main experiment runner
├── results/                 # Experimental results
│   ├── models/revised_cafe_[timestamp]/  # Trained model checkpoints
│   │   ├── best_model.pt
│   │   └── history.json
│   ├── metrics/             # Evaluation metrics
│   │   ├── rtp_evaluation_[timestamp].json
│   │   └── jigsaw_evaluation_[timestamp].json
│   └── plots/               # Visualization plots
└── requirements.txt         # Python dependencies
```

## 🔧 Configuration

### Training Hyperparameters (in `train.py`)

```python
config = {
    'model_name': 'microsoft/deberta-v3-large',
    'max_length': 512,
    'batch_size': 8,          # Adjust based on GPU memory
    'epochs': 3,               # Increase for better performance
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'dropout': 0.1,
    'n_folds': 5
}
```

### Memory Requirements
- **Training:** ~40GB GPU memory (batch_size=8)
- **Reduce batch_size** if you encounter OOM errors
- **Gradient accumulation** can be added for smaller GPUs


## Research Paper

This implementation accompanies the research paper:

*"CAFE: A Context-Aware and Fairness-Weighted Framework for Toxicity Evaluation in Language Models"*

By Wickramasinghe J.J., Department of Computer Science and Engineering, University of Moratuwa

## Citation

```bibtex
@article{wickramasinghe2025cafe,
  title={CAFE: A Context-Aware and Fairness-Weighted Framework for Toxicity Evaluation in Language Models},
  author={Wickramasinghe, J.J.},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

- Jigsaw/Google for the Unintended Bias dataset
- Microsoft for DeBERTa
- Allen AI for RealToxicityPrompts
- HuggingFace for transformers library