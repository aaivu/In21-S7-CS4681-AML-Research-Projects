# TBT-Former Usage Instructions

This document provides detailed instructions for setting up, training, and evaluating TBT-Former on various datasets.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Setup](#dataset-setup)
   - [THUMOS14](#thumos14)
   - [ActivityNet 1.3](#activitynet-13)
   - [EPIC-Kitchens 100](#epic-kitchens-100)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Pre-trained Models](#pre-trained-models)
6. [Monitoring Training](#monitoring-training)

---

## Installation

### Prerequisites

**System Requirements:**

- A machine with CUDA support (GPU with at least 12GB memory is recommended)
- Python 3.x

**Python Dependencies:**

All necessary Python packages are listed in [`requirements.txt`](../requirements.txt). Install them using:

```shell
pip install -r requirements.txt
```

### Compiling NMS Module

Part of NMS is implemented in C++. The code must be compiled before use:

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

**Important:** The code should be recompiled every time you update PyTorch.

---

## Dataset Setup

### THUMOS14

#### Download Features and Annotations

- Download `thumos.tar.gz` (`md5sum 375f76ffbf7447af1035e694971ec9b2`) from:
  - [Box Link](https://uwmadison.box.com/s/glpuxadymf3gd01m1cj6g5c3bn39qbgr)
  - [Google Drive Link](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing)
- The file includes I3D features, action annotations, and external classification scores.

**Feature Details:**

- Extracted from two-stream I3D models pretrained on Kinetics
- Clips of `16 frames` at video frame rate (`~30 fps`)
- Stride of `4 frames`
- Results in one feature vector per `4/30 ~= 0.1333` seconds

For more information about THUMOS14, see the [THUMOS14 dataset documentation](../data/THUMOS14.md).

#### Unpack Features and Annotations

Unpack the file under `./data` (or create a symlink to `./data`). The folder structure should look like:

```
TBT-Former/
│   README.md
│   ...
│
└───data/
    └───thumos/
        └───annotations/
        └───i3d_features/
```

---
- For EPIC-Kitchens 100  & ActivityNet dataset setup, follow the same procedure as THUMOS14. Refer to the [EPIC-Kitchens 100 dataset documentation](../data/EPIC_Kitchens_100.md) for more details about the EPIC-Kitchens 100 dataset.
For more information about ActivityNet, see the [ActivityNet dataset documentation](../data/ActivityNet.md).
---

## Training

### Training on THUMOS14

Train TBT-Former with I3D features. This will create an experiment folder under `./ckpt` that stores the config, logs, and checkpoints.

```shell
python ./train.py ./configs/tbtformer_thumos_i3d.yaml --output reproduce_tbtformer
```

**Hardware Requirements:**

- Training requires ~5GB GPU memory
- Inference may require over 10GB GPU memory
- A GPU with at least 12GB memory is recommended

---

## Monitoring Training

You can monitor the training process in real-time using TensorBoard:

```shell
tensorboard --logdir=./ckpt/tbtformer_thumos_i3d_reproduce_tbtformer/logs
```

Replace the path with the appropriate checkpoint directory for your experiment.

---

## Evaluation

### Evaluating on THUMOS14

Evaluate the trained model:

```shell
python ./eval.py ./configs/tbtformer_thumos_i3d.yaml ./ckpt/tbtformer_thumos_i3d_reproduce_tbtformer
```

**Expected Results:**

- Average mAP: **68.0%**

Detailed results (mAP at different tIoU thresholds):

| Method         | 0.3  | 0.4  | 0.5  | 0.6  | 0.7  | **Avg**  |
| -------------- | ---- | ---- | ---- | ---- | ---- | -------- |
| **TBT-Former** | 82.5 | 79.0 | 72.4 | 60.6 | 45.3 | **68.0** |

---


## Additional Resources

- **Code Structure:** See [Code Overview](../src/README.md#code-overview) for details on the repository structure
- **Dataset Information:**
  - [THUMOS14 Documentation](../data/THUMOS14.md)
  - [ActivityNet Documentation](../data/ActivityNet.md)
  - [EPIC-Kitchens 100 Documentation](../data/EPIC_Kitchens_100.md)
- **Experiments:** See the [experiments folder](../experiments/README.md) for details on ablation studies and preliminary experiments

---

## Troubleshooting

### Common Issues

1. **NMS Compilation Errors:**

   - Ensure you have a compatible C++ compiler installed
   - Recompile the NMS module after any PyTorch update

2. **Out of Memory Errors:**

   - Reduce batch size in the configuration file
   - Ensure you have at least 12GB GPU memory
   - Close other GPU-intensive applications

3. **Dataset Loading Errors:**
   - Verify the folder structure matches the expected layout
   - Check that all files were extracted correctly
   - Verify file permissions

---

## Contact

For questions or issues, please contact:

- Thisara Rathnayaka (thisara.21@cse.mrt.ac.lk)
