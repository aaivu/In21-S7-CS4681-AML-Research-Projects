# Ensemble-Based Network Intrusion Detection System Using Deep Learning

**Student ID:** 210144G  
**Research Area:** Cybersecurity AI - Threat Detection  
**Author:** Nisith Divantha  
**Email:** nisith.21@cse.mrt.ac.lk  

## Abstract

This project implements a novel ensemble-based network intrusion detection system that combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks with traditional machine learning algorithms. The system employs a stacking meta-classifier approach to achieve superior performance in detecting network intrusions across multiple attack categories on the NSL-KDD dataset.

## Project Architecture

### Dataset
- **Primary Dataset:** NSL-KDD (KDD Cup 1999 improved version)
- **Training Set:** KDDTrain+ (125,973 samples)
- **Test Sets:** KDDTest+ (22,544 samples), KDDTest-21 (11,850 samples)
- **Attack Categories:** DoS, Probe, R2L, U2R, Normal

### Model Components
1. **Deep Learning Models:** CNN-LSTM architectures for each attack type combined with Attention mechanism for some
2. **Traditional ML Models:** Random Forest, Gradient Boosting, Logistic Regression
3. **Meta-Classifier:** Random Forest stacking ensemble
4. **Class Imbalance Handling:** SMOTE oversampling and Focal Loss

## Project Structure

```
210144G-Cybersecurity-AI_Threat-Detection/
├── README.md                           # This documentation
├── requirements.txt                    # Project dependencies
├── data/                              # NSL-KDD dataset files
│   ├── KDDTrain+.txt                  # Training dataset
│   ├── KDDTest+.txt                   # Primary test dataset
│   └── KDDTest-21.txt                 # Secondary test dataset
├── src/                               # Source code
│   ├── main.py                        # Single dataset evaluation
│   ├── evaluate_generalizability.py   # Cross-dataset evaluation
│   ├── evaluate_ensemble.py           # Core ensemble evaluation
│   ├── data_preprocessing.py          # Data preprocessing pipeline
│   ├── models.py                      # Model definitions
│   ├── ensemble.py                    # Ensemble methods
│   ├── train_meta_classifier.py       # Meta-classifier training
│   ├── load_models.py                 # Model loading utilities
│   ├── config.py                      # Configuration settings
│   └── utils.py                       # Utility functions
├── base_models/                       # Pre-trained individual models
│   ├── DoS.h5, Probe.h5, R2L.h5, U2R.h5    # Deep learning models
│   └── *_probe_model.pkl              # Traditional ML models
├── meta_models/                       # Ensemble meta-classifiers
│   ├── anomaly_meta_*.pkl             # Trained meta-classifiers
│   └── best_meta_classifier_info.pkl  # Best model metadata
├── docs/                              # Documentation
│   ├── research_proposal.md           # Research proposal
│   ├── literature_review.md           # Literature review
│   └── methodology.md                 # Detailed methodology
│            
└── results/                           # Experimental results
```

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Minimum 8GB RAM (16GB recommended)

### Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- TensorFlow >= 2.8.0
- scikit-learn >= 1.0.2
- pandas >= 1.4.0
- numpy >= 1.21.0
- imbalanced-learn >= 0.8.0

## Reproduction Instructions

### 1. Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd 210144G-Cybersecurity-AI_Threat-Detection
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify data files are present:**
   ```bash
   ls data/
   # Should show: KDDTrain+.txt, KDDTest+.txt, KDDTest-21.txt
   ```

### 2. Model Evaluation (Using Pre-trained Models)

The repository includes pre-trained models for immediate result reproduction.

#### Option A: Single Dataset Evaluation
```bash
cd src
python main.py --sample-size 1000
```

**Parameters:**
- `--sample-size`: Number of samples to evaluate (default: 1000)
- Uses KDDTest+ dataset by default

#### Option B: Cross-Dataset Generalizability Analysis
```bash
cd src
python evaluate_generalizability.py --sample-size 500
```

**For full dataset evaluation (all samples):**
```bash
cd src
python evaluate_generalizability.py  # No sample-size parameter = full datasets
```

**Parameters:**
- `--sample-size`: Number of samples to evaluate per dataset (optional)
- **No sample-size parameter:** Evaluates complete datasets (KDDTest+: 22,544 samples, KDDTest-21: 11,850 samples)
- **With sample-size:** Evaluates specified number of samples from each dataset for faster testing

**Features:**
- Evaluates on both KDDTest+ and KDDTest-21 datasets
- Provides comparative performance analysis
- Generates comprehensive generalizability report
- **Full dataset mode:** Most comprehensive and accurate results for research purposes

#### Option C: Direct Ensemble Evaluation
```bash
cd src
python evaluate_ensemble.py
```

**Advanced usage with custom parameters:**
- Modify `config.py` for dataset paths and model settings
- Adjust sample sizes and evaluation parameters as needed

### 3. Expected Results

#### Full Dataset Evaluation Results
Running `python evaluate_generalizability.py` (without sample-size parameter) produces the following results:

**KDDTest+ Dataset (22,544 samples):**
- Accuracy: 99.52%
- Precision: 99.33%
- Recall: 99.82%
- F1-Score: 99.58%
- ROC-AUC: 99.95%

**Detailed Classification Report (KDDTest+):**
```
              precision    recall  f1-score   support

      Normal       1.00      0.99      0.99      9711
      Attack       0.99      1.00      1.00     12833

    accuracy                           1.00     22544
   macro avg       1.00      0.99      1.00     22544
weighted avg       1.00      1.00      1.00     22544
```

**KDDTest-21 Dataset (11,850 samples):**
- Accuracy: 99.10%
- Precision: 99.14%
- Recall: 99.76%
- F1-Score: 99.45%
- ROC-AUC: 99.74%

**Detailed Classification Report (KDDTest-21):**
```
              precision    recall  f1-score   support

      Normal       0.99      0.96      0.97      2152
      Attack       0.99      1.00      0.99      9698

    accuracy                           0.99     11850
   macro avg       0.99      0.98      0.98     11850
weighted avg       0.99      0.99      0.99     11850
```

**Generalizability Summary:**
```
STACKING ENSEMBLE PERFORMANCE COMPARISON
Dataset         Accuracy   Precision  Recall     F1-Score   ROC-AUC   
---------------------------------------------------------------------------
KDDTest+        0.9952     0.9933     0.9982     0.9958     0.9995    
KDDTest-21      0.9910     0.9914     0.9976     0.9945     0.9974    
```

#### Output Format
Each evaluation produces:
1. **Console Output:** Performance summary, classification report, and metrics table
2. **JSON Results:** Detailed results saved to `/results/` directory with timestamp
3. **Text Reports:** Human-readable evaluation reports in `/results/` directory
4. **Confusion Matrix Plots:** Visual plots saved to `/results/plots/` directory
5. **Latest Results:** `latest_generalizability_results.json` for easy access

**Saved Files Structure:**
```
results/
├── single_evaluation_kddtest_plus_full_dataset_YYYYMMDD_HHMMSS.json
├── single_evaluation_report_kddtest_plus_full_dataset_YYYYMMDD_HHMMSS.txt
├── generalizability_results_YYYYMMDD_HHMMSS.json
├── generalizability_report_YYYYMMDD_HHMMSS.txt
├── latest_generalizability_results.json
└── plots/
    ├── confusion_matrix_stacking_ensemble_kddtest_plus.png
    └── confusion_matrix_stacking_ensemble_kddtest_21.png
```

### 4. Accessing Saved Results

All evaluation results are automatically saved to the `/results` directory with timestamps for version control.

#### Result Files
```bash
# View latest generalizability results
cat results/latest_generalizability_results.json

# View detailed text reports
ls results/*_report_*.txt

# View timestamped JSON results
ls results/*_results_*.json

# Check generated plots
ls results/plots/
```

#### Key Result Files
- **`latest_generalizability_results.json`**: Most recent cross-dataset evaluation
- **`generalizability_report_YYYYMMDD_HHMMSS.txt`**: Human-readable evaluation report
- **`single_evaluation_*.json`**: Individual dataset evaluation results
- **`plots/confusion_matrix_*.png`**: Performance visualization plots

### 5. Model Retraining (Optional)

To retrain models from scratch:

1. **Train individual base models:**
   ```bash
   cd experiments
   jupyter notebook ndd_DoS_.ipynb    # Train DoS detection model
   jupyter notebook ndd_Probe_.ipynb  # Train Probe detection model
   jupyter notebook ndd_R2L_.ipynb    # Train R2L detection model
   jupyter notebook ndd_U2R_.ipynb    # Train U2R detection model
   ```

2. **Train ensemble meta-classifier:**
   ```bash
   cd src
   python train_meta_classifier.py
   ```

**Note:** Retraining requires significant computational resources and time (2-4 hours on GPU).

## Research Documentation

### Academic Papers and References
- **Literature Review:** `docs/literature_review.md`
- **Methodology:** `docs/methodology.md`
- **Research Proposal:** `docs/research_proposal.md`

### Key Contributions
1. **Novel Architecture:** CNN-LSTM hybrid models for each attack type
2. **Ensemble Method:** Stacking meta-classifier with heterogeneous base learners
3. **Class Imbalance Solution:** SMOTE oversampling with Focal Loss optimization
4. **Generalizability Analysis:** Cross-dataset evaluation framework

## Contact

For questions, issues, or collaboration:
- **Email:** nisith.21@cse.mrt.ac.lk
- **GitHub:** @NisithDivantha
- **Institution:** University of Moratuwa, Department of Computer Science and Engineering

---
