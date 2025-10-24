# Methodology: GNN:Molecular Property

**Student:** 210665E
**Research Area:** GNN:Molecular Property
**Date:** 2025-09-01

## 1. Overview

This project develops an E(3)-equivariant adaptation of DimeNet++ for molecular property prediction. The methodology combines DimeNet++'s efficient directional message passing (radial Bessel bases and spherical harmonics) with steerable, equivariant convolutions and tensor algebra (Clebsch–Gordan products) so internal features transform correctly under rotations.

## 2. Research Design

We follow an iterative research-design cycle: (1) survey existing equivariant and directional models, (2) design equivariant feature representations and interaction blocks that retain DimeNet++'s angular efficiency, (3) implement these building blocks using existing equivariant libraries (e.g., e3nn) where possible, (4) run controlled experiments on benchmark datasets (QM9, MD17, COLL), and (5) evaluate performance, data efficiency and equivariance properties (energy and force consistency).

## 3. Data Collection

### 3.1 Data Sources
- QM9 (molecular property dataset)
- MD17 (molecular dynamics trajectories for force learning) — if available
- COLL / datasets referenced in DimeNet++ paper for non-equilibrium configurations

### 3.2 Data Description
- QM9: equilibrium small organic molecules with multiple computed properties (energies, dipoles, HOMO/LUMO etc.). Provided in project as `data/qm9_eV.npz`.
- MD17 (optional): time-series trajectories with forces and energies for small molecules.

### 3.3 Data Preprocessing
- Standardize units (eV/Å where appropriate). The repository notes QM9 in eV (`data/qm9_eV.npz`).
- Normalize target properties (mean/std) per-property for stable training.
- Compute neighbor lists with a sensible cutoff (based on dataset statistics).
- Precompute radial Bessel basis and spherical harmonic components used for angular triple interactions, reusing DimeNet++ preprocessing code where beneficial.

## 4. Model Architecture

High-level design:

- Input: atom types, atomic positions.
- Feature lift: node and directed-edge features that carry irreducible-tensor types (at minimum l=0 scalars and l=1 vectors). Edge features include radial basis expansions R(r) and spherical harmonic directions Y_l^m(\hat{r}).
- Equivariant interaction blocks: steerable filters of form R(r) Y_l^m(\hat{r}) and equivariant tensor products (Clebsch–Gordan) to combine incoming features and produce updated node/edge features.
- Directional triple interactions: implement DimeNet++ style angle-based interactions using vector/tensor inputs and tensor algebra to produce equivariant angular messages.
- Output heads: scalar atomic energy predictors (invariant), summed to total energy. Compute forces by differentiating energy w.r.t. positions to ensure physical consistency.

Implementation notes:
- Reuse or reference `src/dimenet/model/*` code for radial bases and angular preprocessing.
- Use `e3nn` or a lightweight internal implementation for irreducible-tensor handling if adding external dependencies is acceptable.

## 5. Experimental Setup

### 5.1 Evaluation Metrics
- Mean Absolute Error (MAE) for scalar properties (energies, HOMO/LUMO, dipole magnitude where applicable).
- Energy-conserving force RMSE (for force prediction tasks).
- Data efficiency: learning curves vs number of training samples.
- Wall-clock training time and parameter count (compute/efficiency tradeoffs).

### 5.2 Baseline Models
- DimeNet++ (original, scalar-feature implementation).
- NequIP or other equivariant baselines (if pre-trained weights or implementations available).

### 5.3 Hardware/Software Requirements
- GPU(s) with CUDA (NVIDIA) for training; start with a single GPU for prototyping.
- Python 3.8+ (repository uses a `requirements.txt` — follow/extend it). Consider installing `e3nn` and `torch` matching CUDA version.
- Reproducible environment via `requirements.txt` and `setup.py` in `src/`.

## 6. Implementation Plan

| Phase | Tasks | Duration | Deliverables |
|-------|-------|----------|--------------|
| Phase 1 | Literature review and final architecture design | 2 weeks | Design spec and tests for equivariance |
| Phase 2 | Implement equivariant feature lifts and interaction blocks | 3 weeks | Prototype model in `src/dimenet/model` |
| Phase 3 | Small-scale training and debugging (QM9 subset / MD17) | 2 weeks | Learning curves, debugged code |
| Phase 4 | Full-scale evaluation on benchmarks (including COLL) | 3 weeks | Benchmark results and plots |
| Phase 5 | Analysis, write-up and release | 2 weeks | Final report, code release, README updates |
## 7. Risk Analysis

Risks and mitigations:
- Risk: Equivariant implementation is computationally heavy and slows training.
	Mitigation: Profile blocks, reuse efficient radial/angle encodings from DimeNet++, limit higher-order representations where unnecessary, experiment with mixed representations (scalars + vectors only).
- Risk: Implementation complexity (Clebsch–Gordan algebra, steerable filters).
	Mitigation: Use tested libraries (`e3nn`) for tensor algebra when possible; write unit tests for equivariance (rotate inputs and verify transformed outputs).
- Risk: Data availability for force tasks.
	Mitigation: Start with QM9 energy-only tasks and small MD17 subsets; if necessary reduce scope to energies first.

## 8. Expected Outcomes

Expected technical outcomes:
- An adapted DimeNet++ architecture where internal features are E(3)-equivariant (vectors/tensors) and directional triple interactions are implemented with tensor products.
- Demonstration of improved data efficiency compared to scalar-only DimeNet++ on standard benchmarks (QM9, MD17/COLL), quantified by learning curves and MAE/RMSE metrics.
- Implementation of energy and force consistent outputs with analytic gradient-based forces.

Deliverables:
- Code implementing the equivariant DimeNet++ adaptation inside `src/dimenet/model` and training scripts (e.g., `train_equivariant.py`).
- Reproducible experiments, plots, and updated documentation (README and `docs/`).

---

**Note:** Update this document as the implementation and experiments progress.