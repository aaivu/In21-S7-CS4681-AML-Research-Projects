# Research Proposal: Temporal Neural Bellman–Ford Networks (TNBFNet) for Knowledge Graph Reasoning  

**Student:** 210553J  
**Research Area:** Graph Neural Networks – Temporal Knowledge Graphs  
**Date:** 2025-09-01  

---

## Abstract  

Graphs are a fundamental abstraction for representing structured relational data, underpinning domains such as knowledge graphs, social networks, recommender systems, and biological interaction networks. Traditional Graph Neural Networks (GNNs) have achieved strong performance in reasoning over such structures, but they often assume static graphs where entities and relationships remain fixed. Real-world graphs, however, evolve continuously as new interactions occur, entities emerge, and relationships change over time. This introduces critical challenges such as modeling temporal ordering, capturing node state evolution, and learning long-term dependencies.  

Neural Bellman–Ford Networks (NBFNet) [1] represent an important advance in graph learning, embedding the recursive computation of the Bellman–Ford shortest path algorithm into a neural framework to enable efficient, interpretable, path-based reasoning. While NBFNet achieves state-of-the-art results for static link prediction, it is inherently limited to static graphs and cannot incorporate temporal dynamics. Conversely, temporal graph neural networks (e.g., TGAT [2], TGN [3], DyRep [4]) model evolving graphs using temporal embeddings, attention, or memory modules, but often lack explicit, path-based interpretability.  

This research aims to bridge these paradigms by developing **Temporal Neural Bellman–Ford Networks (TNBFNet)**, an extension of NBFNet that incorporates temporal embeddings, decay functions, causal masking, and memory modules inspired by TGN. By combining path-based recurrence with temporal modeling and long-term memory, TNBFNet aims to achieve both **accuracy** and **interpretability** in temporal link prediction. The framework will be benchmarked on widely used temporal knowledge graph datasets, including ICEWS14, ICEWS18, and GDELT, and compared with static and temporal baselines. The expected contributions include a novel temporal reasoning architecture, reproducible implementation, and empirical insights into the trade-offs between interpretability and temporal dynamics.  

---

## 1. Introduction  

Graphs are widely used to represent entities as nodes and their relationships as edges, enabling structured reasoning across diverse application domains such as social networks, biological systems, recommendation engines, and knowledge graphs [2], [5]. Recent advances in graph representation learning have largely been driven by Graph Neural Networks (GNNs), which learn expressive node and edge embeddings through message passing frameworks. In particular, Neural Bellman–Ford Networks (NBFNet) [1] have demonstrated the effectiveness of embedding path-based reasoning into GNN architectures by reformulating the Bellman–Ford recurrence as a differentiable neural process. NBFNet has achieved strong results on static link prediction tasks, combining predictive accuracy with interpretable reasoning paths.  

However, many real-world graphs are not static. Relationships evolve with time, meaning that models must respect temporal causality, prioritize recent interactions, and capture long-term dependencies across evolving structures. Existing temporal GNNs such as TGAT [2], TGN [3], and DyRep [4] address these dynamics through time-aware embeddings, attention mechanisms, or memory updates, but they typically lack the explicit interpretability of path-based models. On the other hand, path-based reasoning models such as NBFNet and MINERVA [6] perform well on static reasoning but do not handle temporal evolution. This gap motivates the central research problem: **How can NBFNet’s path-based recurrence be extended to dynamic graphs, while incorporating temporal modeling and memory mechanisms to preserve both efficiency and interpretability?**  

---

## 2. Problem Statement  

Although NBFNet is highly effective for static link prediction, its applicability to dynamic graphs is restricted by its assumption of static structures. Temporal GNNs capture evolving relationships but often sacrifice explicit interpretability since their predictions are typically based on embeddings or attention weights without clear reasoning chains. Moreover, none of the existing approaches integrate **temporal path reasoning** with **long-term memory modules**. This creates a gap in the field: the lack of a model that combines temporal embeddings and memory updates with explicit, interpretable path-based reasoning.  

The specific research problem addressed in this proposal is therefore the design of a **Temporal NBFNet (TNBFNet)** framework that integrates time-aware embeddings, decay functions, causal masking, and **TGN-style memory modules** to enable interpretable reasoning over dynamic knowledge graphs.  

---

## 3. Literature Review Summary  

Graph learning research has evolved significantly in recent years. Foundations of GNNs include Graph Convolutional Networks (GCNs) [7], Graph Attention Networks (GATs) [8], and GraphSAGE [9], which generalize deep learning to graph-structured data by iteratively aggregating neighborhood information. These approaches are effective but face issues such as oversmoothing and limited receptive fields in deep architectures [10].  

In static link prediction, traditional heuristic methods such as common neighbors and Adamic–Adar [11] provided early solutions, later extended by embedding-based models such as TransE [12], DistMult [13], and ComplEx [14]. While computationally efficient, these models cannot capture multi-hop relational dependencies. GNN-based models, including Relational GCNs (R-GCNs) [15], improved multi-relational reasoning, but still struggled with long-range dependencies. Path-based reasoning approaches such as Neural LP [16], MINERVA [6], and NBFNet [1] addressed this limitation by explicitly modeling relational paths, offering both predictive accuracy and interpretability.  

Dynamic graph learning extends these concepts to evolving graphs. Snapshot-based methods apply static predictors sequentially but lose temporal granularity. Continuous-time models leverage embeddings and memory mechanisms to track changes. TGAT [2] introduced time-aware attention, while TGN [3] added node-level memory modules, and DyRep [4] applied temporal point processes. Temporal Knowledge Graph Completion (TKGC) models such as Know-Evolve [17], HyTE [18], RE-Net [19], and CyGNet [20] extended these ideas to time-stamped facts. However, these models prioritize predictive accuracy over interpretability, often failing to provide explicit reasoning chains.  

The research gap lies in unifying path-based interpretability with temporal modeling. Existing temporal GNNs effectively capture dynamics but lack interpretable reasoning, while path-based models such as NBFNet offer interpretability but ignore temporal evolution. Bridging this gap motivates the development of TNBFNet.  

---

## 4. Research Objectives  

The primary objective of this research is to design and implement a **Temporal NBFNet** that extends path-based reasoning to temporal knowledge graphs through the integration of temporal embeddings, decay mechanisms, causal masking, and memory modules.  

Secondary objectives include:  
1. Extending NBFNet with temporal encodings and query-time masking to respect causality.  
2. Incorporating memory modules, inspired by TGN [3], to capture long-term dependencies across evolving graphs.  
3. Benchmarking the proposed model on temporal knowledge graph datasets (ICEWS14, ICEWS18, GDELT).  
4. Comparing performance against state-of-the-art baselines, including TGAT [2], TGN [3], Know-Evolve [17], HyTE [18], RE-Net [19], and CyGNet [20].  
5. Delivering an open-source implementation with reproducible experimental results.  

---

## 5. Methodology  

The proposed methodology builds upon the foundation of NBFNet while extending it into the temporal domain. Knowledge graphs are represented as quadruples ⟨h, r, t, τ⟩, where τ denotes the timestamp.  

First, temporal encodings will be incorporated into the recurrence. Following TGAT [2], time-dependent encodings of timestamps will be concatenated with relation embeddings to capture temporal patterns. Temporal decay functions will assign higher weights to recent interactions while enforcing causal masking to exclude edges that occur after the query time. This ensures strict temporal causality.  

Second, the Bellman–Ford recurrence will be reformulated as a temporal process. At each hop, only temporally valid neighbors contribute to the node representation. An aggregation mechanism (e.g., weighted path aggregation, optionally comparing with PNA [10]) will combine these messages. After K hops, the model will produce a score for candidate entities using a multilayer perceptron that incorporates both the query relation and temporal context.  

Third, **TGN-inspired memory modules** [3] will be integrated into the framework. Each node will maintain a memory vector updated at every interaction using a recurrent function (e.g., GRU or LSTM). The message function will incorporate relational embeddings, temporal encodings, and historical memory states, enabling the model to retain long-term temporal information beyond fixed windows. This design allows TNBFNet to capture both short-term temporal evolution and long-term historical dependencies while preserving interpretability through path-based reasoning.  

Finally, training will use time-aware negative sampling to avoid false negatives from future interactions. Evaluation will follow the filtered ranking protocol [2], [19], measuring mean reciprocal rank (MRR), mean rank (MR), and Hits@K on standard temporal knowledge graph datasets.  

---

## 6. Expected Outcomes  

The expected outcomes of this research are threefold. First, the development of a novel **Temporal NBFNet** that unifies temporal modeling and path-based interpretability. Second, empirical results demonstrating improved performance over static NBFNet and temporal baselines such as TGAT, TGN, and Know-Evolve. Third, the release of an open-source, reproducible framework to support future research. These contributions are expected to advance the state of the art in temporal knowledge graph reasoning, offering both accuracy and interpretability.  

---

## 7. Timeline  

The research will proceed in four main phases. Weeks 1–3 will be dedicated to literature review, focusing on NBFNet, temporal GNNs, and TKGC models. Weeks 4–8 will involve model design and prototype implementation, including the integration of temporal recurrence and memory modules. Weeks 9–12 will focus on experiments with ICEWS14, ICEWS18, and GDELT, while Weeks 13–14 will be dedicated to ablation studies analyzing the effect of temporal embeddings and memory. Weeks 15–16 will focus on analysis, paper writing, and final submission.  

---

## 8. Resources Required  

- **Datasets**: ICEWS14, ICEWS18, GDELT.  
- **Software**: PyTorch, PyTorch Geometric, TorchDrug, DGL, Neo4j, NumPy.  
- **Hardware**: Access to GPU resources, ideally NVIDIA T4 or A100 with at least 16GB VRAM.  

---

## References  

[1] Z. Zhu et al., “Neural Bellman–Ford Networks: A General Graph Neural Network Framework for Link Prediction,” *Advances in Neural Information Processing Systems*, vol. 34, pp. 29476–29490, 2021.  

[2] D. Xu et al., “Inductive Representation Learning on Temporal Graphs,” in *International Conference on Learning Representations (ICLR)*, 2020.  

[3] E. Rossi et al., “Temporal Graph Networks for Deep Learning on Dynamic Graphs,” in *International Conference on Learning Representations (ICLR)*, 2020.  

[4] R. Trivedi et al., “DyRep: Learning Representations Over Dynamic Graphs,” in *International Conference on Learning Representations (ICLR)*, 2019.  

[5] J. Zhou et al., “Graph Neural Networks: A Review of Methods and Applications,” *AI Open*, 2020.  

[6] R. Das et al., “Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases Using Reinforcement Learning,” in *International Conference on Learning Representations (ICLR)*, 2018.  

[7] T. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” in *International Conference on Learning Representations (ICLR)*, 2017.  

[8] P. Veličković et al., “Graph Attention Networks,” in *International Conference on Learning Representations (ICLR)*, 2018.  

[9] W. Hamilton, R. Ying, and J. Leskovec, “Inductive Representation Learning on Large Graphs,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.  

[10] K. Xu et al., “How Powerful are Graph Neural Networks?” in *International Conference on Learning Representations (ICLR)*, 2019.  

[11] L. Adamic and E. Adar, “Friends and Neighbors on the Web,” *Social Networks*, vol. 25, no. 3, pp. 211–230, 2003.  

[12] A. Bordes et al., “Translating Embeddings for Modeling Multi-Relational Data,” in *NeurIPS*, 2013.  

[13] B. Yang et al., “Embedding Entities and Relations for Learning and Inference in Knowledge Bases,” in *ICLR*, 2015.  

[14] T. Trouillon et al., “Complex Embeddings for Simple Link Prediction,” in *ICML*, 2016.  

[15] M. Schlichtkrull et al., “Modeling Relational Data with Graph Convolutional Networks,” in *European Semantic Web Conference (ESWC)*, 2018.  

[16] F. Yang, Z. Yang, and W. Cohen, “Differentiable Learning of Logical Rules for Knowledge Base Reasoning,” in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.  

[17] R. Trivedi, H. Dai, Y. Wang, and L. Song, “Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs,” in *International Conference on Machine Learning (ICML)*, 2017.  

[18] S. S. Dasgupta, S. N. Ray, and P. Talukdar, “HyTE: Hyperplane-Based Temporally Aware Knowledge Graph Embeddings,” in *EMNLP*, 2018.  

[19] W. Jin, M. Qu, and X. Ren, “Recurrent Event Network: Global Structure Inference Over Temporal Knowledge Graph,” in *NeurIPS*, 2020.  

[20] C. Zhu et al., “Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks,” *arXiv preprint arXiv:2012.08492*, 2020.  
