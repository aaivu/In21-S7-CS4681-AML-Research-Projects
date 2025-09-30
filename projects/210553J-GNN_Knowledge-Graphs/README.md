# Temporal NBFNet

Implementation of **Temporal NBFNet (T-NBFNet)**, an extended version of [NBFNet](https://github.com/DeepGraphLearning/NBFNet) originally designed for static graphs.

---

## 📌 Overview

While Neural Bellman–Ford Networks (NBFNet) [Zhu et al., 2022] provide interpretable and efficient path-based reasoning for static link prediction, they cannot capture temporal dependencies.  

To address this gap, we propose **Temporal Neural Bellman–Ford Network (T-NBFNet)**, which integrates:  

1. ⏳ **Sinusoidal time encodings**  
2. ⚖️ **Time-aware decay weighting**  
3. 🚦 **Query-time masking** (to enforce causality)  
4. 🧠 **Memory modules** (inspired by Temporal Graph Networks (TGN) [Rossi et al., 2020]) for long-term state tracking  

**Key Insight:** T-NBFNet maintains the interpretability of path reasoning while improving predictive performance in dynamic graph settings.  

This codebase is implemented in **PyTorch** and [TorchDrug], and supports both **multi-GPU** and **multi-machine** training/inference.  

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug  

---

## ⚙️ Installation

T-NBFNet requires **Python 3.7/3.8** and **PyTorch ≥ 1.8.0**. Dependencies can be installed with either **Conda** or **pip**.

### Using Conda

```bash
conda install torchdrug pytorch=1.8.2 cudatoolkit=11.1 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install ogb easydict pyyaml -c conda-forge
```

### Using pip

```bash
pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchdrug
pip install ogb easydict pyyaml
```

---

## 🚀 Reproduction

To reproduce results with T-NBFNet, run the training and evaluation scripts provided in the `experiments/` directory.

### Training

```bash
# ICEWS14
python -m experiments.train_test_icews14 --mode train --batch-size 4

# ICEWS18
python -m experiments.train_test_icews18 --mode train --batch-size 4

# WIKI
python -m experiments.train_test_WIKI --mode train --batch-size 4

# YAGO
python -m experiments.train_test_YAGO --mode train --batch-size 4
```

### Evaluation

```bash
# ICEWS14
python -m experiments.train_test_icews14 --mode test --batch-size 4

# ICEWS18
python -m experiments.train_test_icews18 --mode test --batch-size 4

# WIKI
python -m experiments.train_test_WIKI --mode test --batch-size 4

# YAGO
python -m experiments.train_test_YAGO --mode test --batch-size 4
```

---

## 📖 References

- **NBFNet:** Zhu, Z., et al. (2022). *Neural Bellman–Ford Networks: A General Graph Neural Network Framework for Link Prediction.*  
- **TGN:** Rossi, E., et al. (2020). *Temporal Graph Networks for Deep Learning on Dynamic Graphs.*  

---