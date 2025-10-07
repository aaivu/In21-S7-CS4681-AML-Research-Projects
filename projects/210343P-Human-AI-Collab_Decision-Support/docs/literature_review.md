# Literature Review: Explainable Boosting Machines (EBM) and Interpretable Machine Learning

**Student:** 210343P  
**Research Area:** Interpretable Machine Learning / Explainable AI (XAI)  
**Date:** 2025-10-07  

---

## Abstract

This literature review explores the development of **Explainable Boosting Machines (EBMs)** within the broader domain of **interpretable machine learning**. It traces their theoretical roots in Generalized Additive Models (GAMs), their architectural advancements, and their implementation through the **InterpretML** framework. EBMs aim to bridge the gap between interpretability and predictive power, performing competitively with black-box models like **XGBoost** while maintaining transparency. The review also compares EBM performance across standard benchmark datasets such as **Adult Income**, **UCI Heart Disease**, and **Credit Card Fraud Detection**. Key findings highlight EBMs’ robustness, fairness adaptability, and their potential to enhance accountability in AI decision systems.

---

## 1. Introduction

As AI models become more complex, the **“black-box problem”** has emerged  high-performing models such as deep neural networks and ensemble methods often lack interpretability. This poses challenges in high-stakes fields like healthcare, finance, and law, where understanding model reasoning is critical.

Explainable Boosting Machines (EBMs) represent a **modern interpretable alternative** that balances accuracy and explainability. EBMs belong to the family of **Generalized Additive Models (GAMs)** but integrate advanced machine learning techniques like cyclic gradient boosting, bagging, and automated feature interaction detection. This allows them to achieve **state-of-the-art (SOTA)** results on structured datasets while maintaining complete transparency.

---

## 2. Search Methodology

### Search Terms Used
- “Explainable Boosting Machine (EBM)”
- “Generalized Additive Models (GAM)”
- “Interpretable machine learning”
- “Glassbox models”
- “Post-hoc explainability”
- “Fairness in XAI”

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  
- [ ] Other: Microsoft Research Documentation  

### Time Period
1986–2024, with focus on modern developments from 2018–2024.

---

## 3. Key Areas of Research

### 3.1 Evolution from Generalized Additive Models (GAMs)

EBMs are an advanced extension of **Generalized Additive Models (GAMs)**, originally developed by *Hastie & Tibshirani (1986)*. GAMs maintain an additive model structure but use non-linear smooth functions to capture feature effects, making them interpretable yet flexible.

**Key Papers:**
- Hastie & Tibshirani (1986) – Introduced the GAM framework allowing non-linear feature modeling while retaining interpretability.  
- Ding et al. (2021) – Reviewed EBM as a modern successor to GAMs incorporating machine learning advancements.  

---

### 3.2 Modern Enhancements in EBMs

EBMs advance GAMs by integrating:
- **Cyclic Gradient Boosting:** Sequential training on feature subsets to control collinearity.  
- **Automatic Interaction Detection:** Inclusion of pairwise feature interactions for enhanced accuracy.  
- **Bagging:** Ensemble averaging to improve robustness and reduce variance.

**Key Papers:**
- Caruana et al. (2015) – Demonstrated EBM’s interpretability in healthcare applications.  
- Chen et al. (2021) – Showed EBM’s role in identifying data quality issues.  
- Microsoft Research (2021) – Documented EBM’s structure and open-source implementation via *InterpretML*.  

---

### 3.3 InterpretML Framework

**InterpretML** is an open-source library by Microsoft Research that implements EBMs and other interpretability tools under a unified API. It supports both **glassbox** (inherently interpretable) and **black-box** explanation methods.

![](../Images/Screenshot_Interpret.png)

**Key Papers:**
- Nori et al. (2019) – Presented *InterpretML: A Unified Framework for Machine Learning Interpretability*.  
- Microsoft Documentation (n.d.) – Provided official usage and performance benchmarks for EBM.  

---

### 3.4 Benchmark Comparisons: EBM vs. Competitors

EBM has been benchmarked against **Logistic Regression**, **Random Forest**, and **XGBoost** on multiple datasets. Results show that EBM performs competitively while maintaining interpretability.

| Dataset       | Domain   | Logistic Regression | Random Forest | XGBoost       | EBM               |
| ------------- | -------- | ------------------- | ------------- | ------------- | ----------------- |
| Adult Income  | Finance  | 0.907 ± 0.003       | 0.903 ± 0.002 | 0.927 ± 0.001 | **0.928 ± 0.002** |
| Heart Disease | Medical  | 0.895 ± 0.030       | 0.890 ± 0.008 | 0.851 ± 0.018 | **0.898 ± 0.013** |
| Credit Fraud  | Security | 0.979 ± 0.002       | 0.950 ± 0.007 | 0.981 ± 0.003 | **0.981 ± 0.003** |

> **Table:** Baseline AUROC comparison across benchmark datasets.

---

## 4. Research Gaps and Opportunities

### Gap 1: Limited exploration of fairness in EBM training  
**Why it matters:** Fairness is critical for real-world AI applications in finance and healthcare.  
**How your project addresses it:** Incorporates **fairness-aware hyperparameter optimization** using demographic parity metrics.

### Gap 2: Lack of self-supervised initialization in EBM  
**Why it matters:** Cold-start training limits model performance on small datasets.  
**How your project addresses it:** Implements **autoencoder-based pretraining** to initialize EBMs using self-supervised representations.

---

## 5. Theoretical Framework

The study builds upon the **Generalized Additive Model (GAM)** foundation and integrates concepts from **boosting**, **ensemble learning**, and **fairness-aware optimization**. The theoretical basis ensures interpretability through additive modeling while leveraging data-driven non-linear feature learning.

---

## 6. Methodology Insights

Common methodologies used in EBM research include:
- **Cyclic Gradient Boosting** for interpretable function learning  
- **Bayesian Hyperparameter Optimization** (Optuna)  
- **Fairness-aware objective functions** using Demographic Parity (Dwork et al., 2012)  
- **Self-supervised Pretraining** using autoencoders (Kingma & Welling, 2013)

These approaches enhance performance, fairness, and model stability without sacrificing interpretability.

---

## 7. Conclusion

Explainable Boosting Machines represent a pivotal development in interpretable AI — combining transparency, fairness, and competitive accuracy. Through innovations such as self-supervised pretraining and fairness-aware optimization, EBMs bridge the gap between **responsible AI** and **real-world deployment**. Future research should expand these methods to larger datasets and multi-objective fairness formulations.

---

## References

1. Addactis. (2022). *Explainable Boosting Machine: a new model for car insurance.* Addactis Blog.  
2. Caruana, R. et al. (2015). *Intelligible models for healthcare: predicting pneumonia risk and hospital 30-day readmission.* ACM SIGKDD.  
3. Chamola, V. et al. (2023). *A Review of Trustworthy and Explainable Artificial Intelligence (XAI).* IEEE Access.  
4. Chen, Z. et al. (2021). *Using Explainable Boosting Machines (EBMs) to Detect Common Flaws in Data.* PKDD.  
5. Ding, J. et al. (2021). *Explainable Boosting Machines: A Review of the Method and Applications.* MDPI Remote Sensing.  
6. Hastie, T. & Tibshirani, R. (1986). *Generalized Additive Models.* Statistical Science.  
7. Nori, A. et al. (2019). *InterpretML: A Unified Framework for Machine Learning Interpretability.* arXiv:1909.09223.  
8. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.  
9. Microsoft Research. (n.d.). *InterpretML Documentation.* Retrieved from https://interpret.ml/  
10. Microsoft. (n.d.). *Explainable Boosting Machines (EBM) regression.* Microsoft Learn.  
11. Kingma, D. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* arXiv preprint.  
12. Dwork, C. et al. (2012). *Fairness Through Awareness.* ITCS.  
13. Ribeiro, M. T. et al. (2016). *Why Should I Trust You?: Explaining the Predictions of Any Classifier.* ACM SIGKDD.  
14. Schug, D. et al. (2023). *Extending Explainable Boosting Machines to Scientific Image Data.* NeurIPS Workshop.  
15. Xu, F. et al. (2019). *Explainable AI: A Brief Survey on History, Research Areas, Approaches and Challenges.* NLPCC.  
16. Zhao, D. & Zhu, J. (2021). *The Judicial Demand for Explainable Artificial Intelligence.* Columbia Law Review.  
17. Zhou, Y. et al. (2021). *Interpretable Recidivism Prediction using Machine Learning Models.* ACM TIST.  
18. Akiba, T. et al. (2019). *Optuna: A Next-Generation Hyperparameter Optimization Framework.* ACM SIGKDD.

---

**Notes:**  
- Total References: 18  
- Focused on recent developments (2018–2024) with seminal works from 1986 and 2015.  
- Combines academic papers, technical reports, and applied case studies for a balanced review.
