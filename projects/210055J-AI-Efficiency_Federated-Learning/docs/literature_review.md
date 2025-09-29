# Literature Review: AI Efficiency – Federated Learning

**Student:** 210055J  
**Research Area:** AI Efficiency: Federated Learning  
**Date:** 2025-09-01  

---

## Abstract
This literature review examines the efficiency challenges in federated learning (FL), with a focus on Federated Averaging (FedAvg) and its limitations under heterogeneous data. Key themes include client drift, personalization, server-side optimization, communication efficiency, and knowledge distillation. The review highlights how existing methods address these issues, their drawbacks in terms of complexity and overhead, and the potential of lightweight client-side regularization mechanisms such as local–global knowledge distillation. Findings suggest that while personalization and server modifications improve stability and accuracy, they often come at deployment costs, whereas minimal client-side enhancements may offer a balance of efficiency, fairness, and robustness.  

---

## 1. Introduction
Federated learning enables distributed model training across multiple clients without sharing raw data, making it attractive for privacy-sensitive domains like healthcare, finance, and mobile applications. FedAvg, the canonical algorithm, is valued for its simplicity and communication efficiency. However, its performance degrades under non-IID (non-identically distributed) data, limited client participation, and heterogeneous devices. This review explores research efforts to enhance FL efficiency and robustness, with a particular emphasis on methods that reduce client drift, improve personalization, optimize server aggregation, and leverage knowledge transfer.  

---

## 2. Search Methodology

### Search Terms Used
- “Federated Learning,” “FedAvg,” “non-IID data,” “client drift”  
- “personalized federated learning,” “federated optimization”  
- “communication efficiency,” “knowledge distillation in FL”  

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  

### Time Period
2017–2024, emphasizing recent developments while including seminal works (e.g., McMahan et al., 2017).  

---

## 3. Key Areas of Research

### 3.1 Drift Mitigation
Non-IID data causes client updates to diverge, leading to unstable training.  
- **FedProx (Li et al., 2020):** Introduced a proximal term to reduce inconsistency in local updates.  
- **SCAFFOLD (Karimireddy et al., 2020):** Used control variates to correct update variance, improving convergence speed.  
- **FedNova (Wang et al., 2020):** Normalized local updates to account for varying local steps.  

These methods improve stability but increase complexity and require careful hyperparameter tuning.  

### 3.2 Personalization Approaches
A single global model may not serve diverse clients equally.  
- **Per-FedAvg (Fallah et al., 2020):** Used meta-learning for client personalization.  
- **pFedMe (Dinh et al., 2020):** Adopted bi-level optimization with Moreau envelopes.  
- **FedPer (Arivazhagan et al., 2019) & FedRep (Collins et al., 2021):** Shared representation layers with personalized heads.  
- **IFCA:** Clustered clients into groups with separate global models.  

Personalization enhances per-client performance but complicates deployment.  

### 3.3 Server-Side Optimization & Communication Efficiency
Server optimization stabilizes aggregation while reducing communication cost.  
- **FedOpt (Reddi et al., 2021):** Applied adaptive optimizers like FedAdam and FedYogi.  
- **FedAvgM (Hsu et al., 2019):** Added momentum for stability.  
- **Compression methods (Alistarh et al., 2017; Bernstein et al., 2018; Stich et al., 2018):** Quantization, sign-based updates, and sparsification improved efficiency but worsened drift under non-IID data.  

### 3.4 Knowledge Distillation (KD) in FL
KD enables knowledge transfer without raw data sharing.  
- **FedMD (Li & Wang, 2019):** Server-driven heterogeneous model distillation.  
- **FedDF (Lin et al., 2020):** Aggregated models using ensemble distillation.  
- **Proposed Study (Aththanayake, 2025):** Local–global KD, where clients use the global model as a teacher to regularize local training, reducing drift while maintaining FedAvg’s communication protocol.  

---

## 4. Research Gaps and Opportunities

### Gap 1: Complexity vs. Deployability
Most existing drift-mitigation and personalization methods require server-side changes, added state, or communication overhead.  
- **Why it matters:** Practical deployments (e.g., mobile, IoT) often require minimal computation and communication.  
- **How this project addresses it:** Explore lightweight client-side regularization (teacher–student distillation) without server modifications.  

### Gap 2: Stability Under Heterogeneity
Current methods often fail to balance accuracy, fairness, and stability across diverse datasets.  
- **Why it matters:** Real-world FL systems must work reliably across domains (vision, text, healthcare).  
- **How this project addresses it:** Investigate hyperparameter tuning of local–global KD (λ, T, τ) for robust cross-domain performance.  

---

## 5. Theoretical Framework
The work builds upon:  
- **Federated Averaging (McMahan et al., 2017):** Baseline FL algorithm.  
- **Regularization Theory:** Proximal constraints (FedProx) and drift correction (SCAFFOLD).  
- **Knowledge Distillation (Hinton et al., 2015):** Teacher–student frameworks for knowledge transfer.  

Together, these theories support the hypothesis that aligning client updates with the global distribution improves stability.  

---

## 6. Methodology Insights
- **Common Approaches:** Proximal objectives, variance reduction, server-side adaptive optimizers, communication-efficient updates, and KD.  
- **Promising for this work:** Lightweight client-side KD, which preserves privacy, does not alter communication, and only adds minimal overhead.  

---

## 7. Conclusion
Federated learning research has advanced from FedAvg to more sophisticated techniques tackling non-IID data, personalization, server-side optimization, and communication bottlenecks. However, many solutions trade simplicity for complexity. The proposed client-side local–global knowledge regularization offers a balanced approach: modest yet consistent improvements in convergence, accuracy, and fairness without altering the server or communication protocol. This positions it as a practical enhancement for real-world FL deployments.  

---

## References
1. McMahan, H. B., et al. (2017). *Communication-efficient learning of deep networks from decentralized data*. AISTATS.  
2. Li, T., et al. (2020). *Federated optimization in heterogeneous networks*. MLSys.  
3. Karimireddy, S. P., et al. (2020). *SCAFFOLD: Stochastic controlled averaging for on-device federated learning*. ICML.  
4. Wang, J., et al. (2020). *Tackling objective inconsistency in heterogeneous federated optimization*. NeurIPS.  
5. Fallah, A., et al. (2020). *Personalized federated learning with theoretical guarantees*. NeurIPS.  
6. Dinh, C. T., et al. (2020). *Personalized federated learning with Moreau envelopes*. NeurIPS.  
7. Arivazhagan, M. G., et al. (2019). *Federated learning with personalization layers*. arXiv:1912.00818.  
8. Collins, L., et al. (2021). *Exploiting shared representations for personalized federated learning*. arXiv:2102.07078.  
9. Reddi, S. J., et al. (2021). *Adaptive federated optimization*. ICLR.  
10. Hsu, T.-M. H., Qi, H., & Brown, M. (2019). *Measuring the effects of non-identical data distribution on federated learning*. arXiv:1909.06335.  
11. Alistarh, D., et al. (2017). *QSGD: Communication-efficient SGD via gradient quantization and encoding*. NeurIPS.  
12. Bernstein, J., et al. (2018). *signSGD: Compressed optimization for non-convex problems*. ICML.  
13. Stich, S. U., et al. (2018). *Sparsified SGD with memory*. NeurIPS.  
14. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. arXiv:1503.02531.  
15. Li, D., & Wang, J. (2019). *FedMD: Heterogeneous federated learning via model distillation*. arXiv:1910.03581.  
16. Lin, T., et al. (2020). *Ensemble distillation for robust model fusion in federated learning*. NeurIPS.  
17. Aththanayake, M. A. S. N. (2025). *Enhancing Federated Averaging with Local–Global Knowledge Regularization*. IEEE Conference Paper.  
