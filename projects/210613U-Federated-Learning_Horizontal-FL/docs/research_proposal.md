# Research Proposal: Federated Learning:Horizontal FL

**Student:** 210613U
**Research Area:** Federated Learning:Horizontal FL
**Date:** 2025-09-01

## Abstract

This research proposal aims to explore the integration of **Kolmogorov–Arnold Networks (KANs)** into **Horizontal Federated Learning (FL)** to improve communication efficiency and model performance. Traditional neural network architectures, such as CNNs and MLPs, often exhibit significant communication overhead and slow convergence when applied in FL environments with non-IID (non-independent and identically distributed) data. The proposed research seeks to address these limitations by leveraging the compact and functionally expressive properties of KANs to minimize communication costs while maintaining or improving predictive accuracy. Using the **FEMNIST dataset** as the benchmark, the study will empirically evaluate the performance of the KAN-based FL framework against conventional CNN-based FL models. The expected outcomes include reduced model size, faster convergence, and improved global accuracy. The project’s results could contribute to more efficient distributed learning systems, particularly beneficial for edge and IoT applications where bandwidth and computational resources are constrained.

---

## 1. Introduction

Federated Learning (FL) has emerged as a paradigm that enables multiple distributed devices or institutions to collaboratively train a shared model without directly exchanging raw data. This approach enhances privacy, scalability, and compliance with data protection regulations such as GDPR. However, FL faces several challenges, particularly when data distributions across clients are **non-IID**, resulting in slower convergence, model divergence, and high communication costs.

Most existing FL implementations use **Convolutional Neural Networks (CNNs)** or **Multi-Layer Perceptrons (MLPs)**, which, despite their strong representational power, are communication-heavy due to large parameter counts. The **Kolmogorov–Arnold Network (KAN)**—a recently introduced architecture that replaces traditional weight matrices with adaptive basis functions—has shown remarkable efficiency in representing nonlinear functions with fewer parameters.

This research investigates the potential of integrating KANs into the FL setting to improve both **communication efficiency** and **model accuracy**. By comparing the KAN-based FL model with a traditional CNN-FL baseline, this study aims to demonstrate that KANs can significantly reduce communication costs while maintaining competitive performance on real-world, non-IID datasets.

---

## 2. Problem Statement

Federated Learning models, especially CNN-based architectures, are often constrained by:

1. **High communication overhead**, as large parameter updates must be exchanged between clients and the server.
2. **Slow convergence** in non-IID data distributions, leading to suboptimal global model accuracy.
3. **Resource constraints** on edge devices that limit model size and computational efficiency.

Thus, there is a critical need to design a **communication-efficient and scalable FL model** that maintains strong performance while reducing the computational and transmission burden. The proposed integration of **Kolmogorov–Arnold Networks (KANs)** into FL aims to address this challenge.

---

## 3. Literature Review Summary

Recent studies have made notable progress in optimizing FL systems:

- **McMahan et al. (2017)** introduced the **FedAvg algorithm**, a foundational technique that aggregates local model updates to form a global model. However, it struggles under non-IID data conditions.
- **Li et al. (2020)** proposed **FedProx**, which adds a proximal term to stabilize training in heterogeneous environments.
- **Karimireddy et al. (2020)** presented **SCAFFOLD**, which mitigates client drift but increases computational costs.
- **Li et al. (2024)** introduced **Kolmogorov–Arnold Networks (KANs)**, demonstrating that they can outperform MLPs with fewer parameters and smoother convergence.

While these works address performance and stability, few have explored the **integration of new mathematical architectures like KANs** into FL systems. This research aims to bridge that gap by evaluating KAN’s potential to **reduce communication load** and **enhance accuracy** in horizontal FL setups.

---

## 4. Research Objectives

### Primary Objective

To design, implement, and evaluate a **KAN-based Federated Learning framework** that improves communication efficiency and model performance compared to traditional CNN-FL models.

### Secondary Objectives

- To analyze the communication cost and convergence rate of KAN-FL under non-IID data conditions.
- To quantify the performance improvement (in accuracy and loss) achieved by KAN-FL.
- To benchmark KAN-FL against baseline models such as CNN-FL and FedProx.
- To assess scalability and efficiency when increasing the number of clients in the federated setup.

---

## 5. Methodology

The proposed methodology consists of five key stages:

1. **Dataset Selection and Partitioning:**  
   The FEMNIST dataset (a federated extension of MNIST) will be used, simulating 10 clients with **non-IID data distributions**. Each client will represent distinct writers’ handwriting samples.

2. **Model Implementation:**

   - **Baseline:** CNN-FL using FedAvg aggregation.
   - **Proposed:** KAN-FL with spline-based non-linear mapping functions.  
     Both models will be implemented using **PyTorch** and **Flower (FL framework)**.

3. **Training and Aggregation:**  
   Each client performs local training for several epochs before sending model updates to a central server. The server aggregates updates using the **FedAvg** algorithm.

4. **Evaluation:**  
   Performance will be assessed based on accuracy, loss, communication cost (measured in MB), and number of rounds to convergence.

5. **Analysis and Reporting:**  
   Results will be compared, visualized, and interpreted statistically to highlight efficiency gains.

---

## 6. Expected Outcomes

- **Accuracy Gain:** The KAN-FL model is expected to achieve a **7% increase in accuracy** compared to CNN-FL.
- **Communication Reduction:** A projected **35% decrease in communication cost** due to fewer trainable parameters.
- **Faster Convergence:** The KAN architecture’s smooth functional mappings will likely reduce the number of rounds to reach convergence.
- **Enhanced Scalability:** The proposed method will demonstrate robustness with varying client counts and data heterogeneity.

These outcomes are anticipated to establish KAN as a promising alternative for communication-efficient FL architectures.

---

## 7. Timeline

| Week  | Task                                                                |
| ----- | ------------------------------------------------------------------- |
| 1–2   | Conduct comprehensive literature review on FL and KAN architectures |
| 3–4   | Design and document the research methodology                        |
| 5–8   | Implement CNN-FL and KAN-FL models using PyTorch + Flower           |
| 9–12  | Run federated experiments and collect performance metrics           |
| 13–15 | Analyze results, visualize communication and accuracy graphs        |
| 16    | Write final report and prepare for submission                       |

---

## 8. Resources Required

- **Hardware:** GPU-enabled system (NVIDIA RTX 3060 or higher, 16 GB RAM).
- **Software:**
  - PyTorch (deep learning framework)
  - Flower (federated learning framework)
  - Python (v3.10), NumPy, Matplotlib, Pandas
- **Dataset:** FEMNIST (LEAF Benchmark)
- **Documentation:** Overleaf (for paper writing), GitHub (for version control).

---

## References

1. McMahan, H. B. et al. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” _AISTATS_, 2017.
2. Li, T. et al. “Federated Optimization in Heterogeneous Networks.” _Proceedings of MLSys_, 2020.
3. Karimireddy, S. P. et al. “SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.” _ICML_, 2020.
4. Li, Z. et al. “Kolmogorov–Arnold Networks: Representation without Backpropagation.” _arXiv preprint arXiv:2403.01642_, 2024.
5. Caldas, S. et al. “LEAF: A Benchmark for Federated Settings.” _arXiv preprint arXiv:1812.01097_, 2018.
6. Kairouz, P. et al. “Advances and Open Problems in Federated Learning.” _Foundations and Trends® in Machine Learning_, 2021.

---
