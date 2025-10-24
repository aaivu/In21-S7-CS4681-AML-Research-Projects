# Research Proposal: GNN:Molecular Property

**Student:** 210665E
**Research Area:** GNN:Molecular Property
**Date:** 2025-09-01

## Abstract

This project proposes an E(3)-equivariant adaptation of DimeNet++, a state-of-the-art directional message-passing graph neural network for molecular property prediction. The aim is to combine DimeNet++'s efficient angular triple interactions and radial basis encodings with steerable, equivariant convolutions so that internal features transform consistently under rotations. By lifting node and edge representations to irreducible-tensor types and using radial learnable functions multiplied by spherical harmonics as filters, the model will preserve physical symmetries while retaining directional information that improves predictions on non-equilibrium molecular configurations. We will implement equivariant interaction blocks, leverage existing equivariant libraries (e3nn) where appropriate, and evaluate the model on QM9 and relevant force/energy benchmarks. Success is measured by improved data efficiency and accuracy versus scalar DimeNet++, consistent energy/force predictions, and efficient training behavior.

## 1. Introduction

Predicting molecular energies and forces accurately and efficiently is central to computational chemistry and materials science. Graph neural networks that incorporate geometric information are especially effective for these tasks. However, ensuring correct behavior under rotations (equivariance/invariance) is crucial for physically meaningful models. DimeNet++ uses directional message passing to capture angles and local geometry but retains scalar internal features; E(3)-equivariant models like NequIP maintain symmetry directly and show powerful data efficiency. This project aims to combine these advantages by adapting DimeNet++ into an E(3)-equivariant framework.

## 2. Problem Statement

Existing directional GNNs (DimeNet family) encode angular information efficiently but do not maintain full E(3)-equivariance in their internal representations. Conversely, equivariant GNNs maintain symmetry but often do not exploit DimeNet++'s directional triple interactions optimized for non-equilibrium molecules. The research problem is: how to design and implement an architecture that combines DimeNet++'s angular message passing with full E(3)-equivariant feature representations and equivariant convolutions, achieving improved data efficiency and accuracy while remaining computationally practical.

## 3. Literature Review Summary

Key literature covers invariant GNNs (SchNet, CGCNN), directional message-passing models (DimeNet, DimeNet++, GemNet) and equivariant tensor-based models (Tensor Field Networks, NequIP, PaiNN). The notable gap is the lack of a published model that implements DimeNet++-style angular triple interactions within a fully E(3)-equivariant internal feature space. This project targets that gap.

## 4. Research Objectives

Primary Objective
- Develop and implement an E(3)-equivariant adaptation of DimeNet++ that preserves directional angular messaging while maintaining equivariant internal representations.

Secondary Objectives
- Demonstrate improved data efficiency and accuracy on QM9 and other benchmarks compared to scalar DimeNet++.
- Implement energy-consistent forces via analytic differentiation of the predicted energy.
- Provide reproducible code, documentation and evaluation scripts in the project repository.

## 5. Methodology

Approach summary:

- Feature design: lift scalar atom and edge features to irreducible representations (l=0 scalars and at least l=1 vectors), attach learned vector features to nodes and directed edges so they rotate appropriately.
- Equivariant filters: use steerable filters S_l^m(r̂) = R(r) Y_l^m(r̂) where R(r) is a learnable radial function and Y_l^m are spherical harmonics.
- Equivariant interactions: replace scalar triple interactions with tensor products (Clebsch–Gordan decompositions) or steerable MLPs that combine vectorial messages to produce equivariant updates.
- Output and forces: aggregate invariant scalar energies per-atom and compute forces by differentiating the total energy w.r.t. positions to ensure force equivariance.
- Libraries and implementation: prefer `e3nn` for tested equivariant operations; reuse DimeNet++ radial and angular preprocessing code in `src/dimenet` where applicable.

## 6. Expected Outcomes

- An implemented E(3)-equivariant DimeNet++ variant in the repository under `src/dimenet/model`.
- Empirical results showing improved data efficiency (learning curves) and comparable or better MAE on QM9 vs baseline DimeNet++.
- Verified energy/force consistency and equivariance tests (rotational checks).
- Documentation and scripts for reproducing experiments.

## 7. Timeline

| Week | Task |
|------|------|
| 1-2  | Literature review and architecture finalization |
| 3-4  | Implement equivariant feature lifts and interaction blocks |
| 5-8  | Prototype training, small-scale experiments and debugging |
| 9-12 | Full-scale evaluation on QM9 and COLL/MD17 variants |
| 13-15| Analysis, write-up and preparing code release |
| 16   | Final submission and supervisor review |


## 8. Resources Required

- Data: `data/qm9_eV.npz` (available in repository), MD17 / COLL datasets if required for force/trajectory experiments.
- Software: Python 3.8+, PyTorch (matching CUDA), `e3nn` (recommended), repository `requirements.txt` and `setup.py` provide base dependencies.
- Hardware: one or more NVIDIA GPUs for training; CPU for preprocessing.

## References

1. Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., ... & Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature Communications, 13(1), 2453.
2. Gasteiger, J., Giri, S., Margraf, J. T., & Günnemann, S. (2020). Fast and uncertainty-aware directional message passing for non-equilibrium molecules (DimeNet++). arXiv preprint arXiv:2011.14115.
3. Klicpera, J., Groß, J., & Günnemann, S. (2020). Directional message passing for molecular graphs (DimeNet). ICLR.
4. Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. arXiv preprint arXiv:1802.08219.

---

**Submission Instructions:**
1. Complete all sections above (done).
2. Commit your changes to the repository.
3. Create an issue with the label "milestone" and "research-proposal" for supervisor review.
4. Attach relevant plots and model checkpoints when available.