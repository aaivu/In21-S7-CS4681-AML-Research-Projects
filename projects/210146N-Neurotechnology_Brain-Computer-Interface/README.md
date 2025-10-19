# Assessing EEG Classification Performance: Classical vs Deep Learning Approaches

This repository contains the codebase, experiments, and analysis conducted for the research project exploring EEG-based brain–computer interface (BCI) decoding using the **EEG-ExPy benchmark** from [NeuroTechX](https://github.com/NeuroTechX/eeg-expy).  
The study evaluates **classical signal processing pipelines** and **deep learning approaches** across three EEG paradigms, **N170**, **P300**, and **SSVEP**, with a focus on **cross-subject generalization** and **performance consistency**.

---

## Overview

EEG-ExPy provides a standardized benchmark for evaluating EEG decoding methods under realistic conditions.  
This project builds upon the provided baseline notebooks to:

- Implement and extend **classical pipelines** using methods such as **Common Spatial Patterns (CSP)**, **Minimum Distance to Mean (MDM)**, and **Tangent Space Mapping**.  
- Experiment with **deep learning models**, including **autoencoders** and simple CNN-based architectures, to test their performance and generalization across subjects.  
- Compare results under **single-subject** and **all-subject** evaluation setups.  

Despite deep models showing promising **training accuracies (up to ~80%)**, their **test accuracies remained near-chance (≈0.5)**, indicating poor generalization due to the nature of the data and lack of.

---

## Repository Structure

```
│
├── README.md # This file
│
├── data/ # Data provided with the EEG-ExPy package
│ ├── visual-N170
│ ├── visual-P300
│ └── visual-SSVEP
│
├── docs/ # Documentation
│ ├── literature_review.md
│ ├── methodology.md
│ ├── research_proposal.md
│ └── usage_instructions.md
│
├── experiments/ # Experiment notebooks (modified EEG-ExPy baseline)
│ ├── N170_experiment.ipynb
│ ├── P300_experiment.ipynb
│ └── SSVEP_experiment.ipynb
│
├── results/ # Experimental outputs
│ └── results.md
│
├── src/ # EEG-ExPy package, the baseline
│
└── requirements.txt
```

