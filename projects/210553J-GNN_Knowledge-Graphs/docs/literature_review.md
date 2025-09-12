# Literature Review: Graph Neural Networks for Knowledge Graph Reasoning  

**Student:** 210553J  
**Research Area:** GNNs for Knowledge Graphs  
**Date:** 2025-09-12  

---

## Abstract  

Graph Neural Networks (GNNs) have become a powerful tool for learning representations on structured data, including knowledge graphs (KGs). This literature review examines developments in static and temporal knowledge graph completion (TKGC), with particular emphasis on path-based reasoning methods and temporal extensions. The review is organized into the foundations of GNNs, static link prediction models, temporal knowledge graph completion approaches, and path-based reasoning frameworks. Findings highlight that static models such as NBFNet offer interpretability through path-based reasoning but lack temporal modeling capabilities, while temporal approaches such as Temporal Graph Networks (TGN) and Temporal Graph Attention Networks (TGAT) capture dynamic evolution but seldom provide interpretable paths. The review identifies research gaps—especially the integration of interpretability, temporal awareness, and memory modules—and suggests that hybrid frameworks may represent a promising direction for advancing TKGC.  

---

## 1. Introduction  

Graphs are a natural representation of entities and their relations, widely applied in domains such as knowledge base completion, recommender systems, and event forecasting. Knowledge graphs are inherently multi-relational, encoding entities as nodes and relations as labeled edges. Classical knowledge graph completion has been addressed using embedding-based methods such as TransE [5] and ComplEx [6]. However, these methods struggle to capture multi-hop relational patterns explicitly, motivating the development of graph neural networks (GNNs) for relational reasoning [1][2].  

Despite their success, most early approaches assume static graph structures. In practice, knowledge graphs evolve continuously, with new entities and relations emerging over time. Temporal knowledge graphs (TKGs) model this evolution, and learning over TKGs has emerged as an important research area. A further challenge is interpretability: many temporal models achieve strong predictive performance but provide little insight into the reasoning process. Path-based reasoning models such as NBFNet [8] improve interpretability but are limited to static settings. This review explores literature bridging these directions, motivating future research into interpretable, temporal, and memory-augmented graph reasoning systems.  

---

## 2. Search Methodology  

### Search Terms Used  
- “Neural Bellman–Ford Networks”, “NBFNet”, “path-based reasoning”  
- “knowledge graph completion”, “link prediction”, “KGC”  
- “temporal knowledge graph”, “temporal KGC”, “dynamic graphs”  
- “TGN”, “TGAT”, “DyRep”, “Know-Evolve”, “RE-Net”, “HyTE”, “CyGNet”  

### Databases Searched  
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  
- [ ] Other: Conference proceedings (NeurIPS, ICLR, ICML, EMNLP, AAAI)  

### Time Period  
2017–2025, focusing on recent developments; seminal earlier work included for foundations.  

---

## 3. Key Areas of Research  

### 3.1 Foundations of Graph Neural Networks  

The foundations of GNNs rest on message-passing neural networks, which aggregate neighborhood information iteratively to update node representations [1]. Early work established theoretical connections between GNN expressivity and the Weisfeiler–Lehman isomorphism test [2], showing limitations of simple aggregation schemes. Extensions introduced attention mechanisms, such as Graph Attention Networks (GAT), which assign learned weights to neighboring nodes [3]. Hamilton et al. [4] introduced GraphSAGE, an inductive framework enabling large-scale learning by sampling neighborhoods. Collectively, these developments laid the groundwork for applying GNNs to relational reasoning on knowledge graphs.  

**Key Papers:**  
- Zhou et al., 2020 – Survey of GNN methods and applications [1].  
- Xu et al., 2019 – Expressivity and WL-test connection [2].  
- Veličković et al., 2018 – Graph Attention Networks [3].  

---

### 3.2 Static Link Prediction  

Static link prediction in KGs has traditionally relied on embedding-based approaches. TransE [5] modeled relations as translations in vector space, while ComplEx [6] used complex-valued embeddings to capture asymmetric relations. R-GCN [7] extended GCNs with relation-specific transformations for multi-relational graphs. More recent work introduced path-based reasoning approaches, such as Neural LP, MINERVA, and NBFNet [8], which formulates multi-hop reasoning as a differentiable generalization of the Bellman–Ford algorithm. While these methods demonstrate strong performance, they are designed for static graphs and do not capture temporal dynamics.  

**Key Papers:**  
- Bordes et al., 2013 – TransE embeddings [5].  
- Trouillon et al., 2016 – ComplEx embeddings [6].  
- Schlichtkrull et al., 2018 – R-GCN [7].  
- Yin et al., 2022 – NBFNet [8].  

---

### 3.3 Dynamic Graph Representation Learning  

Dynamic or temporal graph learning extends GNNs to evolving structures. Temporal Graph Networks (TGN) [9] introduced a memory-based framework, updating node states at event times using message functions and attention-based aggregation. Temporal Graph Attention Networks (TGAT) [10] proposed temporal attention kernels that integrate time encodings into neighborhood aggregation. DyRep [11] modeled temporal graphs as point processes, capturing both communication and association dynamics. For knowledge graphs, Know-Evolve [12] introduced deep temporal reasoning, while HyTE [13] projected embeddings into hyperplanes parameterized by time. These methods capture evolution effectively, but most lack explicit path interpretability, limiting their transparency.  

---

### 3.4 Path-based Reasoning in Temporal KGs  

Path-based reasoning is critical for interpretable KG completion. NBFNet [8] demonstrated that dynamic-programming-style recurrence can capture multi-hop relations effectively in static graphs. Extensions to temporal KGs require incorporating time encodings, causal masking to ensure chronological validity, and decay functions to down-weight distant events. RE-Net [14] and CyGNet [15] attempted sequence-based reasoning for TKGs, while models such as TeMP combined recurrent and attention-based components. These works highlight the potential for path-based reasoning in temporal settings but leave open challenges in balancing interpretability with scalability and temporal fidelity.  

---

### 3.5 Evaluation Datasets and Metrics  

Evaluation of temporal KGC models typically uses datasets such as ICEWS14, ICEWS18 [16], and GDELT [17], which record event-based triples with timestamps. Metrics include Mean Reciprocal Rank (MRR) and Hits@K (e.g., Hits@1, Hits@10), often reported under filtered evaluation to avoid penalizing plausible predictions. Time-aware negative sampling has emerged as a necessary protocol to prevent models from leveraging future knowledge. Benchmarks such as FB15k-237 and WN18RR remain important for static settings, providing continuity with earlier work.  

---

## 4. Research Gaps and Opportunities  

### Gap 1: Lack of interpretability in temporal models  
**Why it matters:** Temporal models such as TGN [9] and TGAT [10] achieve strong performance but are black-box.  
**How your project addresses it:** Extend path-based recurrence with temporal encodings and causal masking.  

### Gap 2: Limited integration of memory  
**Why it matters:** Long-term dependencies require memory for node state retention.  
**How your project addresses it:** Incorporate TGN-style memory into path-based reasoning.  

### Gap 3: Evaluation inconsistencies  
**Why it matters:** Without time-aware negative sampling, models risk temporal leakage.  
**How your project addresses it:** Adopt time-aware sampling and filtered ranking metrics.  

---

## 5. Theoretical Framework  

The theoretical basis for path-based reasoning builds on the Bellman–Ford recurrence, which computes shortest paths through iterative relaxation. NBFNet extended this to differentiable reasoning over paths [8]. Temporal extensions involve augmenting edge functions with time encodings [10], applying decay weighting to model temporal influence, and enforcing causal masks to restrict reasoning to past events. Memory modules such as those in TGN [9] can be integrated to store evolving node states, allowing hybrid reasoning that combines explicit paths with implicit temporal histories.  

---

## 6. Methodology Insights  

Several methodologies recur across the literature. Temporal encodings may be sinusoidal (TGAT [10]) or learned embeddings. Memory modules (TGN [9]) maintain state updates via message passing. Aggregators range from mean pooling to advanced schemes such as Principal Neighbourhood Aggregation (PNA). Loss functions typically involve binary cross-entropy or ranking objectives, with negative sampling adapted for temporal validity. For proposed research, a promising approach is to first extend path-based recurrence with temporal encodings and decay, then integrate memory modules for richer dynamics, enabling ablation studies to assess the contributions of each component.  

---

## 7. Conclusion  

This review highlights the evolution of graph reasoning from static embedding methods to interpretable path-based models and temporal extensions. Static models such as TransE [5] and ComplEx [6] laid the foundation, while GNNs [1]–[4] and R-GCN [7] extended representation learning. Temporal models including TGN [9], TGAT [10], DyRep [11], and Know-Evolve [12] advanced dynamic reasoning but remain largely uninterpretable. Path-based models such as NBFNet [8] excel in interpretability but are static. Future work must bridge these directions by developing temporal, interpretable, and memory-augmented frameworks capable of capturing evolving knowledge while providing transparent reasoning paths.  

---

## References  

[1] J. Zhou et al., “Graph Neural Networks: A Review of Methods and Applications,” *AI Open*, vol. 1, pp. 57–81, 2020.  
[2] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How Powerful are Graph Neural Networks?,” in *ICLR*, 2019.  
[3] P. Veličković et al., “Graph Attention Networks,” in *ICLR*, 2018.  
[4] W. Hamilton, R. Ying, and J. Leskovec, “Inductive Representation Learning on Large Graphs,” in *NeurIPS*, 2017.  
[5] A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko, “Translating Embeddings for Modeling Multi-relational Data,” in *NeurIPS*, 2013.  
[6] T. Trouillon, J. Welbl, S. Riedel, E. Gaussier, and G. Bouchard, “Complex Embeddings for Simple Link Prediction,” in *ICML*, 2016.  
[7] M. Schlichtkrull et al., “Modeling Relational Data with Graph Convolutional Networks,” in *ESWC*, 2018.  
[8] X. Z. Yin, H. Wang, S. Shah, et al., “Neural Bellman–Ford Networks: A General Graph Neural Network Framework for Link Prediction,” in *NeurIPS*, 2022.  
[9] E. Rossi et al., “Temporal Graph Networks for Deep Learning on Dynamic Graphs,” in *ICML Workshop*, 2020.  
[10] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein, “Temporal Graph Attention Networks,” in *ICML Workshop*, 2020.  
[11] R. Trivedi, M. Farajtabar, P. Biswal, and H. Zha, “DyRep: Learning Representations over Dynamic Graphs,” in *ICLR*, 2019.  
[12] R. Trivedi, H. Dai, Y. Wang, and L. Song, “Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs,” in *ICML*, 2017.  
[13] S. S. Dasgupta, S. Ray, and P. Talukdar, “HyTE: Hyperplane-based Temporally Aware Embeddings,” in *EMNLP*, 2018.  
[14] W. Jin, M. Qu, X. Pan, and X. Ren, “Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs,” in *NeurIPS*, 2020.  
[15] C. Zhu, G. Cheng, M. Xu, Z. Zhang, and S. Gao, “Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-generation Networks,” *arXiv:2012.08492*, 2020.  
[16] E. Boschee, J. Lautenschlager, S. O’Brien, and S. Shellman, “ICEWS Event Data,” *Harvard Dataverse*, 2015.  
[17] K. Leetaru and P. Schrodt, “GDELT: Global Database of Events, Language, and Tone,” in *ISA*, 2013.  
