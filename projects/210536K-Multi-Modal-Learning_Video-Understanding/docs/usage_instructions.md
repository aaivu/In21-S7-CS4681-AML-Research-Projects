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

### ActivityNet 1.3

#### Download Features and Annotations

- Download `anet_1.3.tar.gz` (`md5sum c415f50120b9425ee1ede9ac3ce11203`) from:
  - [Box Link](https://uwmadison.box.com/s/aisdoymowukc99zoc7gpqegxbb4whikx)
  - [Google Drive Link](https://drive.google.com/file/d/1VW8px1Nz9A17i0wMVUfxh6YsPCLVqL-S/view?usp=sharing)
- The file includes TSP features and annotations.

**Feature Details:**

- Extracted from R(2+1)D-34 model pretrained with TSP on ActivityNet
- Non-overlapping clips of `16 frames` at `15 fps`
- Results in one feature vector per `16/15 ~= 1.067` seconds

For more information about ActivityNet, see the [ActivityNet dataset documentation](../data/ActivityNet.md).

#### Unpack Features and Annotations

Unpack the file under `./data`. The folder structure should be:

```
TBT-Former/
│   ...
└───data/
    └───anet_1.3/
        └───annotations/
        └───tsp_features/
```

---

### EPIC-Kitchens 100

For EPIC-Kitchens 100 dataset setup, follow the same procedure as THUMOS14. Refer to the [EPIC-Kitchens 100 dataset documentation](../data/EPIC_Kitchens_100.md) for more details about the dataset.

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

### Training on ActivityNet 1.3

Train TBT-Former with TSP features:

```shell
python ./train.py ./configs/tbtformer_anet_tsp.yaml --output reproduce_tbtformer
```

---

### Training on EPIC-Kitchens 100

For EPIC-Kitchens 100, follow the same training procedure as THUMOS14 with the appropriate configuration file.

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

### Evaluating on ActivityNet 1.3

Evaluate the trained model:

```shell
python ./eval.py ./configs/tbtformer_anet_tsp.yaml ./ckpt/tbtformer_anet_tsp_reproduce_tbtformer
```

**Expected Results:**

- Average mAP: **36.8%**

Detailed results:

| Method         | mAP@0.5 | mAP@0.75 | mAP@0.95 | **Avg mAP** |
| -------------- | ------- | -------- | -------- | ----------- |
| **TBT-Former** | 53.9    | 38.2     | 8.5      | **36.8**    |

---

### Evaluating on EPIC-Kitchens 100

For EPIC-Kitchens 100, follow the same evaluation procedure as THUMOS14 with the appropriate configuration file.

---

## Pre-trained Models

### THUMOS14 Pre-trained Model

1. Download the pre-trained TBT-Former model for THUMOS14 (link to be provided)
2. Create a folder `./pretrained` and unpack the file there
3. Evaluate the pre-trained model:

```shell
python ./eval.py ./configs/tbtformer_thumos_i3d.yaml ./pretrained/tbtformer_thumos_i3d_reproduce/
```

---

### ActivityNet 1.3 Pre-trained Model

1. Download the pre-trained TBT-Former model for ActivityNet (link to be provided)
2. Create `./pretrained` and unpack the file there
3. Evaluate the pre-trained model:

```shell
python ./eval.py ./configs/tbtformer_anet_tsp.yaml ./pretrained/tbtformer_anet_tsp_reproduce/
```

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
