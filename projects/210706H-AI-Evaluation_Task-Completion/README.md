# AI Evaluation:Task Completion - 210706H

## Student Information

- **Index Number:** 210706H
- **Research Area:** AI Evaluation:Task Completion
- **GitHub Username:** @JaneeshaJ2001
- **Email:** janeesha.21@cse.mrt.ac.lk

# CAFE: Context-Aware and Fairness-Weighted Framework for Toxicity Evaluation

A comprehensive framework for evaluating toxicity in language models that addresses context-insensitivity and fairness issues in existing approaches like the Perspective API.

## Overview

CAFE (Context-Aware Fairness-Weighted Toxicity Evaluator) is a novel framework that integrates:

- **Contextual Embeddings**: RoBERTa-based embeddings to capture nuanced linguistic context
- **Fairness-Aware Loss**: Multi-objective optimization that reduces bias against marginalized groups
- **Data Augmentation**: Paraphrasing and adversarial crafting to improve robustness

## Key Features

- 🎯 **Context-Aware**: Distinguishes between literal and non-literal expressions (sarcasm, slang)
- ⚖️ **Fairness-Weighted**: Reduces systematic bias against minority dialects
- 🚀 **Robust Evaluation**: Extended dataset with adversarial and paraphrased prompts
- 📊 **Comprehensive Metrics**: F1 score, fairness gap, Expected Maximum Toxicity, and more

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

## Quick Start

### 1. Run Full Experiment Pipeline

```bash
python experiments/run_experiments.py
```

### 2. Run Quick Experiment (for testing)

```bash
python experiments/run_experiments.py --quick
```

### 3. Train Individual Components

```bash
# Train CAFE model
python src/train.py

# Evaluate models
python src/evaluate.py

# Augment data
python src/data_augmentation.py
```

## Project Structure
```
210706H-AI-Evaluation:Task-Completion/
├── data/                     # Datasets
│   ├── raw/                  # Original datasets
│   ├── processed/            # Processed datasets
│   └── augmented/            # Augmented datasets
├── docs/
│   ├── research_proposal.md  # Your research proposal (Required)
│   ├── literature_review.md  # Literature review and references
│   ├── methodology.md        # Detailed methodology
│   └── progress_reports/     # Weekly progress reports
├── experiments/              # Experiment configurations
│   ├── configs/              # YAML configuration files
│   └── run_experiments.py    # Main experiment runner
├── results/                  # Experimental results
│   ├── models/               # Trained model checkpoints
│   ├── metrics/              # Evaluation metrics
│   └── plots/                # Visualization plots
├── src/                      # Source code
│   ├── data_augmentation.py  # Data augmentation utilities
│   ├── model.py              # CAFE and baseline models
│   ├── loss_functions.py     # Multi-objective loss functions
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation pipeline
│   └── utils.py              # Utility functions
├── README.md                 # This file
└── requirements.txt          # Project dependencies
```
## Configuration

Modify `experiments/configs/base_config.yaml` to customize:

- Model parameters (RoBERTa variant, dimensions)
- Training settings (batch size, learning rate, epochs)
- Loss function weights (α, β, γ)
- Data augmentation parameters

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

- RealToxicityPrompts dataset by Gehman et al.
- Perspective API by Google Jigsaw
- Transformers library by Hugging Face
- University of Moratuwa, Department of Computer Science and Engineering