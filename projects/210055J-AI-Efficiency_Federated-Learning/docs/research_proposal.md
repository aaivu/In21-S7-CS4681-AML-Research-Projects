# Research Proposal: AI Efficiency:Federated Learning

**Student:** 210055J  
**Research Area:** AI Efficiency:Federated Learning  
**Date:** 2025-09-01

---

## Abstract

Federated Averaging (FedAvg) has become the most widely adopted baseline for federated learning, due to its simple design and communication efficiency. In FedAvg, clients perform local training over their private data and share model updates with the server, which then aggregates them to form the global model. Despite its elegance, FedAvg is highly vulnerable to non-IID data distributions, partial participation, and client heterogeneity, leading to unstable convergence, client drift, and degraded global accuracy. This work explores a minimal yet effective modification to FedAvg: a client-side *local--global knowledge regularizer*. During local training, each client duplicates the global model into a frozen teacher and a trainable student. The training objective combines standard cross-entropy with a knowledge-distillation loss that anchors the student to the teacher’s predictions. This approach reduces the divergence of local updates while leaving the server and communication protocol unchanged. The proposed enhancement requires only one additional forward pass per batch, incurring negligible overhead while significantly improving convergence under heterogeneous data.  

This method can be positioned as a step towards more robust federated learning. It is compatible with existing deployments, privacy-preserving by design, and able to deliver faster rounds-to-accuracy, higher stability, and better fairness compared to baseline FedAvg. 

---

## 1. Introduction

Federated learning is an emerging paradigm that enables the collaborative training of machine learning models across a network of distributed clients without centralizing their raw data. This setup is increasingly relevant in domains such as mobile devices, healthcare, and finance, where sensitive data cannot be shared due to privacy regulations or bandwidth constraints.

The canonical algorithm in Federated Learning is Federated Averaging (FedAvg). In each round, a subset of clients receives the global model, performs several local epochs of stochastic gradient descent (SGD) on their private data and sends their updated parameters to the server. The server aggregates these updates, typically by weighted averaging to produce a new global model. FedAvg’s appeal lies in its simplicity and communication efficiency: it reduces the number of rounds needed compared to one-step SGD and does not alter the underlying communication protocol.

However, FedAvg struggles in practice when faced with *non-IID data*. In real-world deployments, clients rarely have identically distributed datasets. For example, in mobile keyboard prediction, each user’s typing patterns, vocabulary, and frequency vary significantly. In healthcare, hospitals differ in patient demographics, equipment, and record-keeping practices. Under such heterogeneity, local updates can drift strongly toward client-specific optima, which conflict when averaged at the server. This phenomenon known as *client drift* causes unstable training dynamics, oscillations, and degraded global performance. The issue becomes worse when clients perform more local epochs or when only a small fraction of clients participate in each round.

Several attempts have been made to fix these problems, including proximal objectives, variance-reduction schemes, adaptive optimization at the server and personalization methods. Yet, many of these solutions require modifying the server, adding communication overhead, or maintaining per-client state making them difficult to deploy in constrained environments.

In this work, a different method is explored: keep FedAvg *exactly the same* on the server and communication side, but improve the client’s local training with a regularization mechanism. The method *local--global knowledge regularization* treats the received global model as a frozen teacher and the trainable copy as the student. By aligning the student’s predictions with the teacher’s, each client is nudged toward the global distribution thereby reducing harmful divergence. Importantly, this modification is lightweight: communication cost is unchanged, privacy assumptions are preserved, and the only extra computation is a forward pass of the frozen teacher per batch.

The central research questions motivating this study are as follows.  
1. Does guiding each client with the global model’s predictions reduce drift and improve global accuracy compared to FedAvg under label-skew data?  
2. What are the effective settings for the regularization weight λ, distillation temperature T, and confidence threshold τ across different datasets?  
3. How does the method behave under varying client participation rates, local epochs, and dataset modalities (vision, text, handwriting)?  
4. Can refinements such as confidence-based masking and λ warm-up yield further gains in stability and fairness?  

---

## 2. Problem Statement

FedAvg exhibits degraded performance under non-IID data, leading to unstable convergence and fairness issues due to client drift. Existing methods like FedProx, SCAFFOLD, and server-side optimizers improve stability but often add complexity or require communication protocol changes. The problem is to develop a **lightweight, privacy-preserving, client-side mechanism** that mitigates drift and enhances robustness without modifying the FedAvg protocol or increasing communication cost.

---

## 3. Literature Review Summary

The limitations of Federated Averaging under heterogeneous data distributions have inspired methods such as FedProx, SCAFFOLD, and FedNova, which penalize deviations or normalize updates to mitigate drift. Personalization approaches like Per-FedAvg, pFedMe, FedPer, and FedRep improve per-client adaptation but increase complexity. Server-side improvements (FedOpt, FedAdam, FedYogi) and communication compression (QSGD, signSGD) enhance efficiency but may exacerbate drift.  

Knowledge Distillation (KD) methods such as FedMD and FedDF apply distillation across clients or at the server. In contrast, the proposed approach applies KD *locally* within the client, using the global model as a teacher to stabilize learning. Existing works either modify the server or require extra communication, but the proposed technique achieves stability without altering FedAvg’s structure.  

The research gap lies in developing a **client-only, communication-neutral, and computationally minimal** method that reduces drift and improves fairness under heterogeneous data.

---

## 4. Research Objectives

### Primary Objective
To enhance the robustness and convergence stability of Federated Averaging under non-IID data through a lightweight local–global knowledge regularization mechanism.

### Secondary Objectives
- To reduce client drift and performance oscillation during training.
- To maintain communication and computational efficiency comparable to FedAvg.
- To analyze the sensitivity of hyperparameters (λ, T, τ) and refine them for improved stability and fairness.

---

## 5. Methodology

The proposed enhancement introduces a **local–global knowledge regularization** at the client side while keeping the FedAvg server unchanged.

- Each client receives the global model and duplicates it into a frozen *teacher* and a trainable *student*.  
- The local loss combines supervised cross-entropy with a knowledge-distillation term, controlled by λ and temperature T, applied only when the teacher’s confidence exceeds threshold τ.  
- After local training, only the updated student weights are returned to the server, maintaining the same communication cost as FedAvg.  
- Refinements include confidence thresholding and a warm-up schedule for λ to avoid early over-regularization.  

The methodology involves experimental validation on MNIST, CIFAR-10, and Shakespeare datasets, comparing convergence, stability, and fairness metrics with baseline FedAvg.

---

## 6. Expected Outcomes

- Improved convergence stability and reduced client drift under heterogeneous data distributions.  
- Higher or comparable accuracy to baseline FedAvg across benchmarks.  
- Reduced variance in per-client performance, implying fairer learning outcomes.  
- Minimal computational and communication overhead, ensuring deployment practicality.  
- Empirical validation that small client-side regularization can enhance global performance without server modifications.

---

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review |
| 3-4  | Methodology Development |
| 5-8  | Implementation |
| 9-12 | Experimentation |
| 13-15| Analysis and Writing |
| 16   | Final Submission |

---

## 8. Resources Required

- Python-based federated learning framework (e.g., Flower, FedML, or TensorFlow Federated)  
- Datasets: MNIST, CIFAR-10, Shakespeare  
- GPU computing resources for training  
- Visualization and logging tools (Matplotlib, TensorBoard)  
- Libraries: PyTorch, NumPy, Scikit-learn  

---

## References

1. H.~B. McMahan *et al.*, “Communication-efficient learning of deep networks from decentralized data,” *AISTATS*, 2017.  
2. T.~Li *et al.*, “Federated optimization in heterogeneous networks,” *MLSys*, 2020.  
3. S.~P. Karimireddy *et al.*, “SCAFFOLD: Stochastic controlled averaging for on-device federated learning,” *ICML*, 2020.  
4. J.~Wang *et al.*, “Tackling objective inconsistency in heterogeneous federated optimization,” *NeurIPS*, 2020.  
5. A.~Fallah *et al.*, “Personalized federated learning with theoretical guarantees,” *NeurIPS*, 2020.  
6. C.~T. Dinh, T.~Q. Tran, and J.~Nguyen, “Personalized federated learning with Moreau envelopes,” *NeurIPS*, 2020.  
7. L.~Collins *et al.*, “Exploiting shared representations for personalized federated learning,” *arXiv:2102.07078*, 2021.  
8. M.~G. Arivazhagan *et al.*, “Federated learning with personalization layers,” *arXiv:1912.00818*, 2019.  
9. S.~J. Reddi *et al.*, “Adaptive federated optimization,” *ICLR*, 2021.  
10. T.-M.~H. Hsu, H.~Qi, and M.~Brown, “Measuring the effects of non-identical data distribution on federated learning,” *arXiv:1909.06335*, 2019.  
11. D.~Alistarh *et al.*, “QSGD: Communication-efficient SGD via gradient quantization and encoding,” *NeurIPS*, 2017.  
12. J.~Bernstein *et al.*, “signSGD: Compressed optimisation for non-convex problems,” *ICML*, 2018.  
13. S.~U. Stich *et al.*, “Sparsified SGD with memory,” *NeurIPS*, 2018.  
14. G.~Hinton, O.~Vinyals, and J.~Dean, “Distilling the knowledge in a neural network,” *arXiv:1503.02531*, 2015.  
15. D.~Li and J.~Wang, “FedMD: Heterogeneous federated learning via model distillation,” *arXiv:1910.03581*, 2019.  
16. T.~Lin *et al.*, “Ensemble distillation for robust model fusion in federated learning,” *NeurIPS*, 2020.  

---
