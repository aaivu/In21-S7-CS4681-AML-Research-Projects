# Research Proposal: Exploring Fine-Tuning Enhancements for MolCLR

**Student:** 210306G
**Research Area:** Healthcare AI:Drug Discovery
**Date:** 2025-09-01

## Abstract

MolCLR is a leading framework for **self-supervised molecular representation learning**, leveraging graph neural networks (GNNs) and contrastive learning to encode rich molecular features from unlabelled data, which is highly effective for downstream chemical applications. Despite its success, standard optimisation and architectural enhancements are often proposed to further improve performance. This proposal outlines a systematic investigation into the impact of these common enhancements on the MolCLR fine-tuning process. The investigation explores advanced optimisation strategies, including the **AdamW optimiser** and **cosine annealing learning rate schedules**, as well as architectural modifications like replacing global pooling with a **Transformer-based readout** and **ensembling GIN and GCN backbones**. The expected outcome is to quantify the efficacy of these enhancements and determine if significant performance gains can be achieved over the robust MolCLR baseline.

## 1. Introduction

Accurate molecular embeddings are a cornerstone in modern drug discovery and computational chemistry, enabling models to generalize across tasks from predicting physicochemical properties to assessing bioactivity and toxicity. **MolCLR** represents a significant advancement by using contrastive learning with GNNs to learn rich, transferable representations from unlabelled molecular graphs. These embeddings are then fine-tuned for diverse downstream tasks, addressing the challenge of data scarcity. The project is motivated by the common assumption that this effective framework can be further improved by integrating modern, standard enhancements from the wider deep learning field.

## 2. Problem Statement

MolCLR's pre-trained molecular representations are highly effective, yet the practical impact of standard deep learning enhancements on its fine-tuning performance is not well-quantified. The problem is to systematically investigate and quantify whether theoretically promising, common optimisation techniques (e.g., AdamW, cosine annealing) and architectural modifications (e.g., Transformer readout, GNN ensemble) yield significant and consistent performance gains over the robust and well-tuned baseline MolCLR model on established benchmarks.

## 3. Literature Review Summary

Self-supervised representation learning is crucial due to the limited availability of high-quality labelled data in chemistry. MolCLR, which combines **GIN-based GNNs** with graph augmentations (atom masking, bond deletion, subgraph removal) and contrastive learning, has emerged as a prominent framework. Its embeddings are highly transferable across tasks like chemical toxicity classification and solubility regression. MolCLR builds on traditional molecular representations (ECFP, SMILES) and GNN variants (GCNs, GINs), addressing the limitation of fully capturing molecular graph structure. Prior studies suggest optimisers like AdamW and schedulers like cosine annealing can improve generalization. Architectural improvements like **graph transformer blocks** and **ensemble methods** also show promise. The gap addressed here is the lack of systematic evaluation of these specific enhancements applied to the fine-tuning stage of the MolCLR model.

## 4. Research Objectives

### Primary Objective
To systematically investigate and quantify the impact of standard optimisation and architectural enhancements on the performance of the fine-tuned MolCLR framework on MoleculeNet benchmarks.

### Secondary Objectives
- To implement and evaluate advanced optimisation strategies, including the **AdamW optimiser** and **cosine annealing learning rate schedules**.
- To implement and evaluate architectural modifications, specifically a **Transformer-based readout mechanism** and a **GIN/GCN ensemble model**.
- To conduct a **32-experiment grid search** on hyperparameters (including differential learning rates and decoupled weight decay) to assess model sensitivity and establish optimal baseline configurations.

## 5. Methodology

The approach involves the following steps:

1.  **Baseline Setup:** Establish a reproducible baseline by fine-tuning the pre-trained MolCLR model (GIN backbone) on the target MoleculeNet datasets (e.g., BBBP) using standard configurations.
2.  **Optimisation Enhancement:** Replace the standard Adam optimiser with **AdamW** and implement **Cosine Annealing LR** schedules to assess training stability and convergence.
3.  **Architectural Modification:**
    * **Transformer Readout:** Replace the global pooling layer with a **GraphTransformerPool** module, which uses a learnable `[CLS]` token and a Transformer encoder to generate the graph representation.
    * **Ensemble Method:** Implement a feature-level ensemble using two parallel, pre-trained backbones (**GIN** and **GCN**), concatenating their outputs before passing them to a new prediction head.
4.  **Hyperparameter Tuning:** Conduct a systematic grid search, including exploring differential learning rates (low rate for the GNN base, high rate for the prediction head) and decoupled weight decay, to find optimal fine-tuning settings.
5.  **Evaluation:** Systematically evaluate all modifications and the baseline on MoleculeNet tasks using relevant metrics like **ROC-AUC** and **MAE**.

## 6. Expected Outcomes

* A fully functional and reproducible codebase for the MolCLR baseline and all enhancement strategies.
* Conclusive evidence and systematic data quantifying the performance impact of standard fine-tuning strategies on MolCLR.
* Documentation that either establishes an improved fine-tuning protocol or highlights the **inherent robustness** of the pre-trained MolCLR representations, thereby suggesting that future work should pivot to more **fundamental modifications** of pre-training objectives or augmentation strategies.
* A set of rigorously benchmarked results detailing the model's sensitivity to hyperparameters, optimisers, and architectural changes.

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature Review (MolCLR, GNNs, Contrastive Learning, Optimisation Strategies) |
| 3-4  | Methodology Development & Baseline Setup (Reproduce MolCLR baseline and establish experimental structure) |
| 5-8  | Implementation (Coding AdamW, Schedulers, Transformer Readout, Ensemble Model) |
| 9 | Experimentation (32-Experiment Grid Search, Optimiser/Scheduler Testing, Architectural Evaluation) |
| 10| Analysis and Writing (Data interpretation and result summarisation) |
| 11 | Final Submission (Final report preparation and submission) |

## 8. Resources Required

* **Tools/Software:** Python 3.6, PyTorch 1.7, PyTorch Geometric (PyG), RDKit, MolCLR official repository.

## References

L. David, A. Thakkar, R. Mercado, and O. Engkvist, "Molecular representations in Al-driven drug discovery: a review and practical guide," J. Cheminformatics, vol. 12, no. 1, pp. 1-22, 2020.
Y. Wang, J. Wang, Z. Cao, and A. B. Farimani, "Molecular contrastive learning of representations via graph neural networks," arXiv preprint arXiv:2102.10056, 2021.
D. Rogers and M. Hahn, "Extended-connectivity fingerprints," J. Chem. Inf. Model., vol. 50, no. 5, pp. 742-754, 2010.
D. Weininger, "SMILES, a chemical language and information system," J. Chem. Inf. Comput. Sci., vol. 28, no. 1, pp. 31-36, 1988.
M. Krenn, F. Häse, A. Nigam, P. Friederich, and A. Aspuru-Guzik, "SELFIES: A 100% robust molecular string representation," Mach. Learn.: Sci. Technol., vol. 1, no. 4, p. 045024, 2020.
T. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," arXiv preprint arXiv:1609.02907, 2016.
K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How powerful are graph neural networks?" in Proc. ICLR. 2019.
I. Loshchilov and F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," arXiv preprint arXiv:1608.03983, 2016.
I. Loshchilov and F. Hutter, "Decoupled weight decay regularization." in Proc. ICLR, 2019.
S. Yun, M. Jeong, R. Kim, J. Kang. and H. J. Kim, "Graph transformer networks," in Proc. NeurIPS, 2019.
Z. Wu. B. Ramsundar. E. N. Feinberg. J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, and V. Pande, "MoleculeNet: A benchmark for molecular machine learning." Chem. Sci., vol. 9, no. 2, pp. 513-530, 2018.

---
