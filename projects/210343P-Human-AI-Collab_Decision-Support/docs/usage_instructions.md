# Usage Instructions

## 1. Clone Repository & Open Notebook

This project is part of the **Human–AI Collaboration: Decision Support** research repository.  
To clone only this project folder using **Git sparse-checkout**, run:

```bash
git clone --no-checkout https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd In21-S7-CS4681-AML-Research-Projects/

# Enable sparse-checkout
git sparse-checkout init --cone

# Set the project directory to checkout
git sparse-checkout set projects/210343P-Human-AI-Collab_Decision-Support

# Checkout the main branch
git checkout main
```

Alternatively, you can open the main Jupyter notebook directly in **Google Colab** or **Kaggle** by uploading the `.ipynb` file found in this project folder.

---

## 2. Install Dependencies

Install the required Python libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn interpret optuna torch joblib
```

| **Library**       | **Purpose**                                                  |
| ----------------- | ------------------------------------------------------------ |
| **numpy, pandas** | Data manipulation and preprocessing                          |
| **scikit-learn**  | Data splitting, scaling, metrics, and baselines              |
| **interpret**     | Training and visualizing Explainable Boosting Machines (EBM) |
| **optuna**        | Bayesian hyperparameter optimization                         |
| **torch**         | Self-supervised autoencoder pretraining                      |
| **joblib**        | Saving and loading trained models and Optuna studies         |
| **matplotlib**    | Generating performance and fairness plots                    |

📌 *Standard Python modules used (no installation needed):*  
`os`, `random`, `argparse`, `typing`

---

## 3. Prepare Datasets

The experiments use three publicly available benchmark datasets:

1. **UCI Adult Income** – Financial fairness benchmarking  
2. **Credit Card Fraud Detection** – Security domain classification  
3. **UCI Heart Disease** – Medical decision-support dataset  

You can download them from the **UCI Machine Learning Repository** or **Kaggle**.

Place them in the following directory structure:

```
/data/
 ├── adult.csv
 ├── creditcard.csv
 ├── heart.csv
```

If stored elsewhere, update the dataset paths inside the notebook.

---

## 4. Run the Notebook

The workflow follows the same stages as described in the research paper:

### **A. Baseline EBM Training**
- Loads the dataset and performs **StratifiedShuffleSplit**.
- Trains a baseline **ExplainableBoostingClassifier** using default parameters.
- Evaluates metrics: ROC AUC, F1-score, confusion matrix, and feature importance.

### **B. Bayesian Hyperparameter Optimization**
- Uses **Optuna** for two-stage optimization:
  - **Stage 1:** Maximizes predictive performance (ROC AUC).  
  - **Stage 2:** Adds a fairness penalty based on **Demographic Parity Difference (DP)**.  
- Best hyperparameter configurations are stored as Optuna study files.

### **C. Self-Supervised Pretraining**
- Trains a **tabular autoencoder** using PyTorch to learn feature embeddings.  
- Extracts latent features and uses them to generate **`init_scores`** for the EBM:
  
  $$
  \text{init\_scores} = \hat{p}
  $$

- Initializes EBM training using these warm-start scores to improve convergence and performance in low-label conditions.

Run all cells sequentially (`Run All`) to reproduce the baseline, optimized, and pretrained models.

---

## 5. Visualizing Results

Use the **InterpretML dashboard** to explore global and local explanations:

```python
from interpret import show
show(ebm_model)
```

Visual outputs include:
- Feature importance before and after fairness optimization  
- Confusion matrices for all datasets  
- Pareto front showing the **accuracy–fairness trade-off**

---

## 6. Saved Artifacts

After successful execution, the following artifacts are generated automatically:

| **File Name**                  | **Description**                         |
| ------------------------------ | --------------------------------------- |
| `baseline_ebm.joblib`          | Default trained EBM                     |
| `optuna_study_performance.pkl` | Optuna study (performance-only)         |
| `optuna_study_fairness.pkl`    | Optuna study (fairness-aware)           |
| `autoencoder_pretrain.pth`     | Self-supervised PyTorch autoencoder     |
| `init_scores.npy`              | Pretraining-based initialization scores |
| `final_ebm.joblib`             | Combined pretrained + optimized model   |

---

## 7. Reproducibility Notes

- Random seeds are fixed for consistent dataset splits.  
- Experiments use **3 repeated stratified splits** for reliability.  
- Each Optuna trial logs `ROC`, `DP`, and all hyperparameters for later analysis.  
- The repository includes saved studies for reproducibility and comparison.

---

## 8. System Requirements

| **Component**      | **Requirement**                             |
| ------------------ | ------------------------------------------- |
| **Python Version** | 3.10 or higher                              |
| **RAM**            | ≥ 16 GB                                     |
| **CPU**            | Multicore (parallel EBM training supported) |
| **GPU**            | Optional (used for pretraining only)        |

---

