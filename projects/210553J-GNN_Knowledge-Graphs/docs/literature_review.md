# Literature Review: Graph Neural Networks for Knowledge Graph Reasoning  

**Student:** 210553J  
**Research Area:** GNNs for Knowledge Graphs  
**Date:** 2025-09-12  

---

## Abstract  

Graph Neural Networks (GNNs) have become a powerful tool for learning representations on structured data, including knowledge graphs (KGs). This literature review examines developments in static and temporal knowledge graph completion (TKGC), with particular emphasis on path-based reasoning methods and temporal extensions. The review is organized into the foundations of GNNs, static link prediction models, temporal knowledge graph completion approaches, and path-based reasoning frameworks. Findings highlight that static models such as NBFNet offer interpretability through path-based reasoning but lack temporal modeling capabilities, while temporal approaches such as Temporal Graph Networks (TGN) and Temporal Graph Attention Networks (TGAT) capture dynamic evolution but seldom provide interpretable paths. Challenges include scalability, memory retention for long-term dependencies, and evaluation inconsistencies due to temporal leakage. This review identifies research gaps—especially the integration of interpretability, temporal awareness, and memory modules—and suggests that hybrid frameworks combining explicit path reasoning with temporal memory mechanisms may represent a promising direction for advancing TKGC.  

---

## 1. Introduction  

Graphs are a natural representation of entities and their relations, widely applied in domains such as knowledge base completion, recommender systems, event forecasting, and biological networks. Knowledge graphs encode entities as nodes and multi-relational edges as typed relationships, offering a flexible and semantically rich structure. Classical knowledge graph completion approaches aim to infer missing edges (or links) by learning latent representations of nodes and relations. Embedding-based methods such as TransE [5] and ComplEx [6] project nodes and relations into vector spaces, modeling relationships as translations or complex-valued transformations. While effective for many scenarios, these approaches are limited in capturing multi-hop relational patterns, temporal dynamics, and interpretability.  

Graph Neural Networks (GNNs) have emerged to address these limitations by performing iterative message passing across the graph structure, allowing nodes to aggregate information from their local neighborhoods and beyond. By leveraging GNNs, researchers can model high-order interactions, encode relational patterns, and integrate additional features such as temporal context. The development of temporal knowledge graphs (TKGs) introduces further complexity, as entities and relations evolve over time. TKGs incorporate timestamps into triples, allowing for dynamic reasoning but introducing challenges in long-term dependency modeling, interpretability, and scalability.  

This literature review explores key developments in static and temporal GNN-based knowledge graph reasoning, focusing on path-based models that offer interpretability and temporal models that capture evolution. By analyzing foundational concepts, current methods, and evaluation frameworks, the review identifies gaps and opportunities for hybrid approaches that can simultaneously provide temporal awareness, interpretable reasoning, and efficient memory mechanisms.  

---

## 2. Search Methodology  

A systematic search of the literature was conducted using combinations of keywords relevant to static and temporal knowledge graph reasoning. The search was performed across multiple scholarly databases to ensure coverage of both seminal and recent publications.  

### Search Terms Used  
- “Neural Bellman–Ford Networks”, “NBFNet”, “path-based reasoning”  
- “knowledge graph completion”, “link prediction”, “KGC”  
- “temporal knowledge graph”, “temporal KGC”, “dynamic graphs”  
- “TGN”, “TGAT”, “DyRep”, “Know-Evolve”, “RE-Net”, “HyTE”, “CyGNet”  
- “memory-augmented graph neural networks”  
- “causal graph reasoning”, “multi-hop relational inference”  

### Databases Searched  
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  
- [x] Conference proceedings: NeurIPS, ICLR, ICML, AAAI, EMNLP  

### Time Period  
The review primarily covers publications from 2017–2025, with earlier seminal works included to establish foundations. Emphasis was placed on recent temporal reasoning methods, hybrid frameworks, and path-based interpretability.  

---

## 3. Key Areas of Research  

### 3.1 Foundations of Graph Neural Networks  

Graph Neural Networks (GNNs) form the core of modern graph reasoning approaches. GNNs operate under the principle of neighborhood aggregation, where each node updates its representation by aggregating messages from its neighbors through learnable transformations [1]. The iterative aggregation process allows nodes to capture increasingly global structural information, enabling relational reasoning over multi-hop paths.  

The expressivity of GNNs has been theoretically analyzed through the lens of the Weisfeiler–Lehman (WL) isomorphism test [2], demonstrating that simple aggregation schemes may fail to distinguish certain non-isomorphic graphs. To enhance expressivity, Graph Attention Networks (GATs) [3] introduced attention mechanisms that assign learnable importance weights to neighboring nodes, allowing the network to focus on more relevant connections. GraphSAGE [4] introduced an inductive framework enabling learning on previously unseen nodes, crucial for large-scale and evolving graphs.  

These foundational architectures established the basis for applying GNNs to knowledge graphs, where multi-relational edges, heterogeneity, and temporal evolution present additional challenges. Extensions to these basic GNNs have included relation-aware aggregators, multi-hop path reasoning, and temporal encodings, all of which are crucial for knowledge graph completion.  

**Key Papers:**  
- Zhou et al., 2020 – Comprehensive survey of GNN methods and applications [1].  
- Xu et al., 2019 – Expressivity and WL-test connection [2].  
- Veličković et al., 2018 – Graph Attention Networks [3].  
- Hamilton et al., 2017 – GraphSAGE for inductive representation learning [4].  

---

### 3.2 Static Link Prediction  

Static link prediction aims to infer missing edges in knowledge graphs by learning node and relation embeddings. Early methods include TransE [5], which models relationships as translations in a low-dimensional vector space, and ComplEx [6], which uses complex-valued embeddings to capture asymmetric relations and multiplicative interactions. These methods effectively model pairwise relations but lack mechanisms for multi-hop relational reasoning.  

To address this limitation, GNN-based approaches such as R-GCN [7] extended convolutional networks to multi-relational graphs by introducing relation-specific weight matrices during aggregation. Path-based reasoning models, such as Neural LP, MINERVA, and NBFNet [8], explicitly learn multi-hop reasoning patterns through differentiable logic or dynamic programming-style recurrences. NBFNet, for instance, formulates a differentiable generalization of the Bellman–Ford algorithm to aggregate evidence along multiple paths, offering both performance and interpretability.  

While static models achieve strong predictive accuracy, they cannot handle temporal dynamics where relations and entities evolve over time. Moreover, path-based models for static graphs often face scalability challenges on large-scale knowledge graphs due to the combinatorial growth of possible paths.  

**Key Papers:**  
- Bordes et al., 2013 – TransE embeddings for multi-relational data [5].  
- Trouillon et al., 2016 – ComplEx embeddings [6].  
- Schlichtkrull et al., 2018 – R-GCN for relational graphs [7].  
- Yin et al., 2022 – NBFNet for path-based reasoning [8].  

---

### 3.3 Dynamic Graph Representation Learning  

Temporal knowledge graphs (TKGs) extend static KGs by adding timestamps to triples, representing the evolution of entities and relations. Dynamic graph representation learning has focused on developing models that capture the temporal evolution of nodes and edges while retaining multi-hop relational reasoning capabilities.  

Temporal Graph Networks (TGN) [9] introduced a memory-based framework that maintains evolving node states updated upon the occurrence of events. Each node possesses a memory vector updated via message functions and attention-based aggregators, enabling long-term dependency modeling. Temporal Graph Attention Networks (TGAT) [10] employ temporal attention kernels, incorporating time encodings directly into neighborhood aggregation, thereby allowing the model to weigh neighbors according to temporal proximity. DyRep [11] models the evolution of graphs as a temporal point process, capturing both association dynamics (formation of edges) and communication dynamics (node interactions).  

For knowledge graph reasoning, models like Know-Evolve [12] extend point-process-based modeling to multi-relational temporal graphs, allowing prediction of future events conditioned on past occurrences. HyTE [13] projects embeddings into temporally-aware hyperplanes, capturing temporal trends in entity-relation interactions. These models are effective at modeling dynamics but often operate as black-boxes, lacking interpretability of their reasoning process.  

**Key Insights:**  
- Memory modules are essential for capturing long-term dependencies.  
- Temporal encodings can be fixed (sinusoidal) or learned.  
- Dynamic reasoning requires causal masking to prevent leakage from future information.  

---

### 3.4 Path-based Reasoning in Temporal KGs  

Interpretability in KGC is increasingly valued, especially in domains such as finance, healthcare, and scientific discovery. Path-based reasoning provides transparency by explicitly modeling the sequences of edges connecting entities. NBFNet [8] demonstrates the power of differentiable path-based reasoning in static graphs but cannot handle temporal evolution.  

Temporal path-based reasoning requires augmenting static methods with temporal information. Key strategies include:  
1. **Time encoding:** Incorporating timestamps in edge representations to model temporal influence.  
2. **Causal masking:** Ensuring predictions only consider past events, respecting chronological order.  
3. **Decay functions:** Gradually reducing the influence of older events on current reasoning.  

Recent works, such as RE-Net [14] and CyGNet [15], attempt sequence-based reasoning on TKGs, integrating recurrent networks with attention mechanisms. TeMP extends this by combining temporal embeddings with multi-hop aggregation. Despite these advances, challenges remain in achieving scalability, maintaining interpretability, and integrating memory modules for long-term dependencies.  

---

### 3.5 Evaluation Datasets and Metrics  

Temporal KGC models are evaluated using benchmark datasets that record event-based triples:  
- **ICEWS14, ICEWS18 [16]:** Event datasets from political and social news sources.  
- **GDELT [17]:** Global database of events, language, and tone.  

Static KG models often use FB15k-237 and WN18RR. Metrics include:  
- **Mean Reciprocal Rank (MRR)**: Measures the rank of the true entity.  
- **Hits@K**: Proportion of true entities ranked in the top K predictions.  
- **Filtered evaluation:** Excludes other valid triples during ranking.  
- **Time-aware negative sampling:** Ensures that negative samples do not include future events, preventing temporal leakage.  

Evaluation consistency is critical, as models trained without proper temporal awareness may inadvertently exploit future knowledge, inflating performance metrics.  

---

### 3.6 Comparative Overview of Key Models

| Model | Year | Type | Multi-hop Reasoning | Temporal Awareness | Memory Module | Interpretability | Notes |
|-------|------|------|-------------------|-----------------|---------------|-----------------|-------|
| TransE [5] | 2013 | Static | No | No | No | Low | Simple translation-based embeddings; cannot capture multi-hop patterns. |
| ComplEx [6] | 2016 | Static | No | No | No | Low | Uses complex-valued embeddings; models asymmetric relations. |
| R-GCN [7] | 2018 | Static | Limited | No | No | Medium | Relation-specific aggregation; handles multi-relational graphs. |
| NBFNet [8] | 2022 | Static | Yes | No | No | High | Differentiable Bellman–Ford recurrence; interpretable multi-hop reasoning. |
| TGN [9] | 2020 | Temporal | Limited | Yes | Yes | Low | Memory-based node state updates; effective for evolving graphs. |
| TGAT [10] | 2020 | Temporal | Limited | Yes | No | Low | Temporal attention kernels with time encoding; good temporal modeling. |
| DyRep [11] | 2019 | Temporal | Limited | Yes | No | Low | Point process-based dynamic graph modeling; captures associations and communications. |
| Know-Evolve [12] | 2017 | Temporal | Limited | Yes | No | Low | Temporal relational modeling via continuous-time point processes. |
| HyTE [13] | 2018 | Temporal | No | Yes | No | Low | Embeddings projected into temporally-aware hyperplanes; interpretable time trends. |
| RE-Net [14] | 2020 | Temporal | Yes | Yes | No | Medium | Sequence-based recurrent reasoning over TKGs; partial interpretability. |
| CyGNet [15] | 2020 | Temporal | Yes | Yes | No | Medium | Copy-generation sequence modeling; interpretable multi-hop temporal paths. |
| TeMP | 2020 | Temporal | Yes | Yes | Yes | Medium | Combines recurrent and attention-based components with temporal encoding. |

**Legend:**  
- Multi-hop Reasoning: Ability to model paths longer than one edge.  
- Temporal Awareness: Can handle evolving graphs with timestamps.  
- Memory Module: Retains historical node states for long-term dependencies.  
- Interpretability: Provides human-understandable reasoning paths or mechanisms.  

---

## 4. Research Gaps and Opportunities  

### Gap 1: Lack of interpretability in temporal models  
Temporal models such as TGN and TGAT are powerful but opaque. Future work should explore integrating path-based reasoning into temporal frameworks to enhance interpretability without sacrificing performance.  

### Gap 2: Limited integration of memory  
Long-term dependencies are crucial for TKGs. While TGN provides a memory mechanism, its integration with path-based reasoning is underexplored. Memory-augmented path reasoning could enable both temporal awareness and transparent multi-hop inference.  

### Gap 3: Evaluation inconsistencies  
Time-aware negative sampling and filtered ranking metrics are essential to avoid temporal leakage. Standardization of evaluation protocols across datasets remains a challenge.  

### Gap 4: Scalability and efficiency  
Multi-hop path-based reasoning scales combinatorially with path length. Efficient approximations or sampling strategies are needed for large-scale TKGs.  

### Gap 5: Hybrid frameworks  
There is a need for models that jointly optimize for temporal reasoning, interpretability, and memory retention. Hybrid architectures combining recurrence, attention, path reasoning, and memory modules represent a promising research direction.  

---

## 5. Theoretical Framework  

The theoretical foundation for path-based reasoning relies on the Bellman–Ford recurrence, which iteratively computes shortest paths over graphs. NBFNet generalizes this concept into a differentiable framework for reasoning over multiple paths [8]. Temporal extensions involve:  
- **Time-aware edge functions:** Incorporating timestamps into edge weights or relation embeddings.  
- **Decay weighting:** Modeling the diminishing influence of older events.  
- **Causal masks:** Preventing leakage from future events during reasoning.  
- **Memory modules:** Retaining evolving node states to model long-term dependencies.  

Such a framework allows hybrid reasoning, combining explicit multi-hop paths with implicit temporal memory, yielding models that are both interpretable and temporally aware.  

---

## 6. Methodology Insights  

Methodologies across the literature converge on several core components:  
1. **Temporal encoding:** Sinusoidal or learned embeddings representing time.  
2. **Memory modules:** TGN-style memory maintains node histories.  
3. **Neighborhood aggregation:** Mean pooling, sum, attention, or advanced schemes like PNA.  
4. **Loss functions:** Ranking losses, binary cross-entropy, and negative sampling adapted for temporal graphs.  
5. **Path-based recurrence:** Differentiable dynamic programming for interpretable multi-hop reasoning.  

A promising hybrid approach involves extending NBFNet with temporal encodings and decay, then integrating memory modules for richer temporal dynamics. Ablation studies can quantify the contributions of each component and guide model design for scalable, interpretable temporal KGC.  

---

## 7. Conclusion  

Graph Neural Networks have transformed knowledge graph reasoning, evolving from static embedding-based models to interpretable path-based networks and temporally-aware frameworks. Static models such as TransE [5], ComplEx [6], and R-GCN [7] laid foundational representations, while path-based reasoning (NBFNet [8]) provides interpretability. Temporal models such as TGN [9], TGAT [10], DyRep [11], and Know-Evolve [12] capture dynamic evolution but remain largely opaque.  

The literature highlights several research opportunities:  
- Integrating path-based interpretability with temporal reasoning.  
- Incorporating memory mechanisms to retain long-term dependencies.  
- Standardizing evaluation with time-aware negative sampling.  
- Developing scalable, hybrid architectures that balance accuracy, interpretability, and efficiency.  

By bridging these directions, future models can achieve interpretable, temporally-aware, and memory-augmented reasoning over dynamic knowledge graphs, advancing the state of knowledge graph completion research.  

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
