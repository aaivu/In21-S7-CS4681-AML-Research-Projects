# Literature Review: GNN:Molecular Property

**Student:** 210665E
**Research Area:** GNN:Molecular Property
**Date:** 2025-09-01

## Abstract

This literature review surveys recent work at the intersection of graph neural networks and molecular property prediction, with a focus on how models handle 3D rotational symmetry. I cover three main families of models: invariant GNNs (e.g., SchNet, CGCNN), directional message-passing networks (DimeNet, DimeNet++ and GemNet), and fully equivariant E(3)-models (e.g., NequIP, PaiNN, tensor field networks). Key findings: directional message-passing (DimeNet++) improves accuracy by leveraging angular information but still uses scalar internal features; E(3)-equivariant models provide guarantees on symmetry and show substantial data efficiency; no published model (to date of this review) fully combines DimeNet++'s directional messaging with full E(3)-equivariance — this gap motivates the present project.

## 1. Introduction

Graph neural networks (GNNs) have rapidly become the method of choice for predicting molecular properties from atomic structures. A central challenge in molecular modeling is respecting physical symmetries — specifically translation and rotation invariance/equivariance — while extracting rich geometric information such as distances and angles. This review focuses on how modern GNNs incorporate 3D geometry and rotational structure, contrasting invariant approaches, directional message-passing schemes, and fully equivariant tensor methods. The goal is to identify strengths, weaknesses, and open questions that inform a hybrid DimeNet++ → E(3)-equivariant architecture.

## 2. Search Methodology

### Search Terms Used
- "graph neural network" AND "molecular" AND "equivariant"
- "DimeNet" OR "DimeNet++" OR "directional message passing"
- "E(3)-equivariant" OR "equivariant neural networks" OR "NequIP" OR "PaiNN"
- "spherical harmonics" AND "bessel basis" AND "molecular"

### Databases Searched
- Google Scholar
- arXiv
- Web of Science / Scopus (when available via institutional access)
- Conference proceedings (NeurIPS, ICLR, ICML)

### Time Period
Primary focus on work from 2018–2024, with inclusion of seminal earlier works (e.g., Tensor Field Networks, 2018) and more recent 2020–2022 advances (DimeNet++, GemNet, NequIP).

## 3. Key Areas of Research

### 3.1 Invariant GNNs
Invariant models use scalar geometric features (distances, sometimes angles) which are invariant under rotations. Examples include SchNet and CGCNN. These methods are straightforward to implement and guarantee rotational invariance of scalar outputs like energy. However, they can be data-inefficient because scalar features cannot represent directional or tensorial physical quantities directly.

Key papers:
- SchNet (Schütt et al., 2017) — continuous-filter convolutional network for molecular properties.
- CGCNN (Xie & Grossman, 2018) — crystal graph convolutions for materials.

### 3.2 Directional Message-Passing Networks
DimeNet and DimeNet++ introduced message embeddings that carry directional information associated with bonds and angles. They use spherical Bessel functions and spherical harmonics to encode distances and angles and demonstrated state-of-the-art accuracy on several molecular benchmarks. DimeNet++ in particular optimized the architecture for speed and uncertainty-aware predictions.

Key papers:
- Directional Message Passing (Klicpera et al., 2020) — DimeNet.
- Fast and Uncertainty-aware Directional Message Passing (Gasteiger et al., 2020) — DimeNet++.
- GemNet (Gasteiger et al., 2020) — further improvements with universal directional graph neural networks.

### 3.3 Equivariant GNNs (E(3)/O(3))
Equivariant models carry features that transform under rotations as vectors or higher-order tensors. These include Tensor Field Networks, PaiNN, NequIP and related approaches. By construction they propagate information in a way that preserves equivariance, which can improve data efficiency and allow direct prediction of tensorial quantities like forces.

Key papers:
- Tensor Field Networks (Thomas et al., 2018) — introduce irreducible representation based convolutions.
- NequIP (Batzner et al., 2022) — E(3)-equivariant GNN with strong data efficiency and accuracy for interatomic potentials.
- PaiNN — equivariant message passing with vector features.

## 4. Research Gaps and Opportunities

Gap 1: No published model combines DimeNet++'s directional triple interactions with full E(3)-equivariance.
Why it matters: DimeNet++ efficiently encodes angular information that is useful for non-equilibrium structures, while equivariant models deliver data efficiency and correct transformation behavior. Combining both could improve accuracy and sample efficiency.
How this project addresses it: Extend DimeNet++ features to carry irreducible-tensor representations (vectors/tensors) and replace scalar interaction blocks with equivariant tensor products and steerable filters (radial functions × spherical harmonics).

Gap 2: Efficient implementation and training of equivariant directional message passing that preserves DimeNet++'s computational advantages.
Why it matters: Equivariant models are often more expensive; the trade-off between runtime and symmetry guarantees affects usability on larger datasets or longer simulations.
How this project addresses it: Reuse DimeNet++'s radial Bessel basis and efficient angular encodings, while implementing compact equivariant interaction blocks inspired by NequIP and tensor algebra (Clebsch–Gordan products) to control computational cost.

## 5. Theoretical Framework

The theoretical basis combines directional message passing with representation theory of the rotation group. Filters are constructed as R(r) Y_l^m(\hat{r}) (radial learnable functions times spherical harmonics) and act on type-l features (irreducible representations). Tensor products (Clebsch–Gordan decompositions) combine features of different types to produce higher- or lower-order features while preserving equivariance. Energy is produced as a scalar invariant from equivariant atomic features; forces follow by analytic differentiation of the energy with respect to atomic positions, guaranteeing correct equivariant behavior.

## 6. Methodology Insights

Common methodologies:
- Radial basis expansions (spherical Bessel) and spherical harmonics to encode distances and angles.
- Directional message passing with explicit angular triple interactions (DimeNet family).
- Equivariant convolutions using steerable filters and tensor products (NequIP, Tensor Field Networks).

Most promising approach for this project: retain DimeNet++'s efficient radial and angular bases, but lift internal features to irreducible representations (vectors/tensors) and replace scalar interaction blocks with compact equivariant tensor algebra (Clebsch–Gordan or learnable steerable MLPs). Leverage e3nn or similar libraries for tested equivariant building blocks to reduce implementation burden.

## 7. Conclusion

Directional message-passing networks and E(3)-equivariant models provide complementary strengths: the former delivers efficient angular reasoning while the latter enforces physical symmetry and data efficiency. The literature shows strong motivation for combining these ideas. This project will adapt DimeNet++ to an E(3)-equivariant framework by lifting features to tensor representations and using steerable filters and tensor products to preserve equivariance while retaining computational efficiency.

## References

1. Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., ... & Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature Communications, 13(1), 2453.
2. Gasteiger, J., Giri, S., Margraf, J. T., & Günnemann, S. (2020). Fast and uncertainty-aware directional message passing for non-equilibrium molecules (DimeNet++). arXiv preprint arXiv:2011.14115.
3. Klicpera, J., Groß, J., & Günnemann, S. (2020). Directional message passing for molecular graphs (DimeNet). ICLR.
4. Gasteiger, J., Becker, F., & Günnemann, S. (2020). GemNet: Universal directional graph neural networks for molecules. NeurIPS.
5. Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. arXiv preprint arXiv:1802.08219.
6. Schütt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., Tkatchenko, A., & Müller, K.-R. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. NeurIPS.
7. Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical Review Letters, 120(14), 145301.
8. Weiler, M., Geiger, M., Welling, M., Boomsma, W., & Cohen, T. S. (2018). 3D steerable CNNs: Learning rotationally equivariant features in volumetric data. NeurIPS.
9. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A., Leswing, K., & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical Science, 9(2), 513–530.

---

**Notes:**
- Aim to expand the reference list to 15–20 high-quality papers as the project continues.
- Include a mix of conference, journal, and arXiv technical reports.