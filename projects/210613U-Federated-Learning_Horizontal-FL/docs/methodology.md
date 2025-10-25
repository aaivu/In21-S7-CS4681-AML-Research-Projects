# Methodology: Federated Learning:Horizontal FL

**Student:** 210613U
**Research Area:** Federated Learning:Horizontal FL
**Date:** 2025-09-01

## 1. Overview

This methodology outlines the experimental and analytical procedures used to evaluate the impact of **Kolmogorov–Arnold Network (KAN)** architectures on **communication efficiency and model performance** in a horizontal federated learning (FL) environment. The study adopts a **quantitative, simulation-based research design**, emphasizing reproducibility, scalability, and empirical validation using the **FEMNIST dataset**.

The primary objective is to compare the proposed **KAN-FL framework** against a **Convolutional Neural Network (CNN)-based FL baseline** under non-IID conditions to demonstrate:

- Improved model accuracy.
- Reduced communication overhead.
- Enhanced convergence stability.

## 2. Research Design

This research follows an **experimental simulation approach** using a controlled FL environment.  
The design consists of three major components:

1. **Data Partitioning:** Simulating real-world horizontal FL by distributing the FEMNIST dataset across 10 clients in a **non-IID manner**, where each client receives data from a unique subset of writers.
2. **Model Training:** Clients locally train their models (CNN or KAN) for a fixed number of epochs before sending updates to a central server.
3. **Aggregation & Evaluation:** The server aggregates model updates using **Federated Averaging (FedAvg)**, evaluates global performance, and measures communication cost.

The research is iterative and comparative — emphasizing statistical validation of results through multiple experimental runs.

## 3. Data Collection

### 3.1 Data Sources

- **Primary Dataset:** FEMNIST (Federated Extended MNIST) dataset from the LEAF benchmark suite.
- **Nature of Data:** Handwritten character images (62 classes, 3500+ writers).
- **Data Type:** Grayscale images (28×28 pixels).

### 3.2 Data Description

The dataset is well-suited for horizontal FL because it naturally represents **user-partitioned data** (each writer as a client).

- Total samples: ~800,000 images.
- Each client: ~2,000–5,000 samples.
- Number of classes per client: Variable (non-IID distribution).

### 3.3 Data Preprocessing

1. **Normalization:** Pixel values scaled to [0,1].
2. **Reshaping:** Converted into tensors of shape (1×28×28).
3. **Encoding:** Labels one-hot encoded into 62 classes.
4. **Partitioning:** Custom Python scripts to simulate **non-IID client distributions**.
5. **Batching:** Mini-batches of size 32 used for local training.

## 4. Model Architecture

### 4.1 Baseline CNN Model

The baseline FL model is a lightweight **Convolutional Neural Network (CNN)** consisting of:

- Two convolutional layers (ReLU activation, kernel size = 3×3).
- One max-pooling layer (2×2).
- Fully connected dense layer with dropout (0.5).
- Output layer using Softmax activation for 62-class classification.

**Advantages:** Simple, effective for image data, widely used in prior FL research.  
**Limitations:** High parameter count → increased communication cost.

### 4.2 Proposed KAN Model

The **Kolmogorov–Arnold Network (KAN)** replaces dense layers with spline-based univariate function layers, drastically reducing parameter counts while maintaining non-linear representational capacity.  
Each neuron learns a **basis function** through cubic B-splines, offering flexible yet efficient mappings.

**Architecture Summary:**

- Input Layer: 784 nodes (flattened image).
- Hidden Layers: Two spline-based function layers with adaptive activation.
- Output Layer: 62 nodes (Softmax).
- Total parameters reduced by ~35% compared to CNN.

**Key Benefits:**

- Compact representation → less communication per round.
- Faster convergence in FL due to reduced overfitting.
- Smooth, continuous mappings yield stable gradients.

## 5. Experimental Setup

### 5.1 Evaluation Metrics

The following quantitative metrics are used for assessment:

- **Accuracy (%):** Classification accuracy of the global model.
- **Communication Cost (MB):** Total data transmitted between clients and server.
- **Convergence Rounds:** Number of communication rounds required to reach 90% of final accuracy.
- **Loss (Cross-Entropy):** Average training and test loss per round.
- **Computation Time (s):** Total local training time per client.

### 5.2 Baseline Models

| Model             | Description                       | Key Features                                |
| ----------------- | --------------------------------- | ------------------------------------------- |
| CNN-FL            | Standard CNN with FedAvg          | High communication cost, strong performance |
| KAN-FL (Proposed) | Spline-based KAN model            | Compact, efficient, adaptive nonlinearity   |
| FedProx           | FedAvg variant with proximal term | Improved stability in non-IID setups        |
| Scaffold          | Control variate method            | Reduces variance between client updates     |

### 5.3 Hardware/Software Requirements

- **Frameworks:** PyTorch, Flower (FL framework).
- **Programming Language:** Python 3.10.
- **Hardware:**
  - CPU: Intel Core i7 or AMD Ryzen 7.
  - GPU: NVIDIA RTX 3060 (6 GB VRAM).
  - RAM: 16 GB minimum.
- **Software:**
  - Ubuntu 22.04 LTS
  - CUDA 12.1, cuDNN 8.x
  - Python packages: `torch`, `flwr`, `numpy`, `matplotlib`, `pandas`.

---

## 6. Implementation Plan

| Phase   | Tasks                                | Duration | Deliverables                       |
| ------- | ------------------------------------ | -------- | ---------------------------------- |
| Phase 1 | Data preprocessing and partitioning  | 2 weeks  | Clean, non-IID FEMNIST dataset     |
| Phase 2 | Model implementation (CNN & KAN)     | 3 weeks  | Working local training scripts     |
| Phase 3 | Federated simulation & experiments   | 2 weeks  | Aggregation logs, results          |
| Phase 4 | Performance analysis & visualization | 1 week   | Accuracy/Loss graphs, final report |

---

## 7. Risk Analysis

| Risk                        | Description                                   | Mitigation Strategy                               |
| --------------------------- | --------------------------------------------- | ------------------------------------------------- |
| Data imbalance              | Clients with highly uneven data distributions | Apply stratified sampling or weighted aggregation |
| Overfitting in local models | Small local datasets lead to bias             | Use early stopping and dropout                    |
| Communication delay         | High network latency during FL simulation     | Asynchronous update simulation                    |
| Model divergence            | Non-IID data causing unstable updates         | Integrate FedProx-style regularization            |
| Hardware limitations        | GPU memory constraints for KAN training       | Reduce batch size or model width                  |

---

## 8. Expected Outcomes

1. **Accuracy Improvement:**  
   The proposed KAN-FL model is expected to achieve approximately **7% higher accuracy** than CNN-FL on FEMNIST.

2. **Communication Reduction:**  
   Due to its compact spline-based structure, KAN-FL is projected to reduce communication costs by **~35%** per training round.

3. **Convergence Behavior:**  
   KAN’s continuous basis functions are expected to yield smoother convergence curves and fewer oscillations.

4. **Scalability:**  
   The architecture should scale efficiently up to 50 clients without significant performance degradation.

5. **Scientific Contribution:**  
   This research provides an empirical demonstration of how **mathematical network architectures (KANs)** can enhance the **communication–performance trade-off** in Federated Learning.

---
