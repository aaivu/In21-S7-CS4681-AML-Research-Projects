# Dataset Integration (Temporal & Sentinel)

**Purpose.** This document describes how to add and prepare the two fMoW-derived datasets used by this project:

- **Temporal** (`fMoW-rgb` / `fMoW-full`) - multi-temporal imagery (large).
- **Sentinel** (`fmow-sentinel`) - Sentinel-series multi-band tiles prepared for fMoW use.

This README gives precise commands, recommended workflows, and the expected final directory layout so you can register these datasets in the training/inference pipeline immediately.

> For Temporal data, if you do **not** require multispectral bands for your experiments, use **`fmow-rgb`**. It is ~200 GB vs ~3.5 TB for the full multispectral collection — the time and cost saved are substantial for most development and prototyping workflows.

---

## Table of Contents

1. [Overview & choices](#overview--choices)  
2. [Temporal dataset (fMoW)](#temporal-dataset-fmow)  
   - [Where to get it](#where-to-get-it)  
   - [Download commands (examples)](#download-commands-examples)  
   - [Extract ground truth](#extract-ground-truth)  
   - [Rename to `temporal`](#rename-to-temporal)  
   - [Metadata files & placement](#metadata-files--placement)  
   - [Subsampling (recommended for small compute)](#subsampling-recommended-for-small-compute)  
   - [Expected directory layout](#expected-directory-layout-temporal)  
3. [Sentinel dataset (`fmow-sentinel`)](#sentinel-dataset-fmow-sentinel)  
   - [Where to get it](#where-to-get-it-1)  
   - [Download & extract (example)](#download--extract-example)  
   - [Preprocess metadata](#preprocess-metadata)  
   - [Subsampling (optional)](#subsampling-optional)  
   - [Expected directory layout](#expected-directory-layout-sentinel)  
4. [Notes, best-practices and warnings](#notes-best-practices-and-warnings)  
5. [Acknowledgements & references](#acknowledgements--references)

---

# Overview & Choices

Two fMoW variants are commonly used:

- **fMoW-full** - TIFFs, multi-band (4-band and 8-band multispectral). Very large (~**3.5 TB**). Use only when you need multispectral information for model design or ablation.
- **fMoW-rgb** - JPEGs where multispectral imagery has been converted to RGB. Much smaller (~**200 GB**), perfect for fast iteration.

Both are compatible with the pipeline as delivered. Choose `fmow-rgb` for most work; choose `fmow-full` when multispectral fidelity is required.

---

# Temporal Dataset (`fMoW`)

## Where to Get

Public S3 listing:
```bash
s3://spacenet-dataset/Hosted-Datasets/fmow/
```

Inside that prefix you will find:
- `fmow-rgb/`  (≈ 200 GB, JPEG)
- `fmow-full/` (≈ 3.5 TB, TIFF multispectral)

## Download Commands

Download **fmow-rgb** (recommended for most users):
```bash
# preferred: sync (resumable, safe)
aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb ./fmow-rgb
# or: recursive copy
aws s3 cp --recursive s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb ./fmow-rgb
```

Download fmow-full (only if you need multispectral bands):
```bash
aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-full ./fmow-full
```

Strong recommendation: use `aws s3 sync` instead of `cp --recursive` for large datasets. `sync` is resumable and will not repeatedly re-download files if interrupted.

## Extract Ground Truth

In the dataset root there is a `groundtruth.tar.bz2`. This file contains ground-truth metadata and will be merged with the image metadata.
```bash
cd fmow-rgb              # or fmow-full
# verify the file exists
ls -lh groundtruth.tar.bz2

# extract (bzip2 compressed tar)
tar -xvjf groundtruth.tar.bz2
```

After extraction the ground-truth files will be present and can be merged with the image metadata (the supplied files already perform that merge in this repo's helper scripts).

## Rename Dataset Folder

Standardize the dataset directory name so the pipeline expects the same path:
```bash
cd ..
mv fmow-rgb temporal     # or: mv fmow-full temporal
```

## Metadata Files & Placement

Download the CSV metadata files (provided on the Google Drive link you were given) and place them under the `temporal` directory:
- `train_62classes.csv`
- `val_62classes.csv`

Options to obtain them:
1. Manual (recommended for most users)
   - Open the Drive folder: [https://drive.google.com/drive/folders/1-xSXNpq0xJ4z3F7BPzEcZ04eZ7LqPbYD](https://drive.google.com/drive/folders/1-xSXNpq0xJ4z3F7BPzEcZ04eZ7LqPbYD)
   - Download `train_62classes.csv` and `val_62classes.csv` via browser, then copy them to `temporal/`.

2. Automated (if you have file IDs) using `gdown`:
```bash
# Example (replace FILE_ID with actual file id from Drive)
gdown --id <FILE_ID> -O temporal/train_62classes.csv
gdown --id <FILE_ID> -O temporal/val_62classes.csv
```

Once placed, your temporal directory should include those CSVs.

## Subsampling (recommended for limited compute)

The full temporal dataset contains ~1M images and 62 classes. If you lack large compute, subsample by class using the repository's helper:
```bash
python subsample.py \
  --input_csv temporal/train_62classes.csv \
  --output_csv temporal/train_62classes_sample.csv \
  --remove_ratio 0.80 \
  --seed 42
```

- `--remove_ratio` = fraction of examples to remove (0.8 removes 80% of samples); set to desired reduction.
- Repeat for validation CSV similarly.

## Expected Directory Layout (Temporal)

After following the above steps your `/data` directory should look like:
```bash
/data
└── temporal
    ├── train/
    │   ├── airport/
    │   ├── ...
    ├── val/
    │   ├── airport/
    │   ├── ...
    ├── train_62classes.csv
    ├── train_62classes_sample.csv    # if you created a subsample
    ├── val_62classes.csv
    └── val_62classes_sample.csv      # if you created a subsample
```

Once this structure is in place you can register `temporal` as the dataset in the pipeline and start training/finetuning.

---

# Sentinel Dataset (`fMoW-sentinel`)

## Where to Get

Public persistent URL:
```bash
https://purl.stanford.edu/vg497cb6002
```
- This is a multispectral image dataset containing `.TIF` files worth ~78 GB.

## Download & Extract

Create the directory and download the archive. Next, download the files using `wget` and then extract:
```bash
mkdir -p data/sentinel
cd data/sentinel

# Attempt direct download:
wget https://stacks.stanford.edu/file/vg497cb6002/fmow-sentinel.tar.gz
wget https://stacks.stanford.edu/file/vg497cb6002/test_gt.csv
wget https://stacks.stanford.edu/file/vg497cb6002/train.csv
wget https://stacks.stanford.edu/file/vg497cb6002/val.csv

# Extract:
tar -xzvf fmow-sentinel.tar.gz
```

## Preprocess Metadata

The Sentinel metadata CSVs in the archive do **not** contain individual per-image file paths. Use the provided `preprocess.py` to add the `image_path` column (the script concatenates `category`, `folder`, and `image id` to create a full relative path).
```bash
python preprocess.py
```

The script will create records with paths in the format:
```bash
fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif
```

## Subsampling (optional)

Sentinel also contains a large number of images. Follow the same `subsample.py` usage pattern if you want a smaller dataset:
```bash
python subsample.py \
  --input_csv sentinel/train.csv \
  --output_csv sentinel/train_sample.csv \
  --remove_ratio 0.80 \
  --seed 42
```
- `--remove_ratio` = fraction of examples to remove (0.8 removes 80% of samples); set to desired reduction.
- Repeat for validation CSV similarly.

## Expected Directory Layout (Sentinel)

```bash
/data
└── sentinel
    ├── train/
    │   ├── airport/
    │   ├── ...
    ├── val/
    │   ├── airport/
    │   ├── ...
    ├── train.csv
    ├── train_sample.csv
    ├── val.csv
    └── val_sample.csv
```

---

# Notes, Best-practices and Warnings

- **Disk & I/O costs**: `fmow-full` is ~3.5 TB. This consumes large disk, time and egress fees if moving across regions. Don't underestimate this.
- **Use `sync`**: `aws s3 sync` is preferable to `aws s3 cp --recursive` for large, interruptible transfers.
- **Subsample aggressively for quick iteration**. Even a small subset retained across all classes gives you realistic training dynamics while saving lots of time.
- **No preprocessing required for fMoW imagery** (the images themselves are already prepared). Only the Sentinel metadata required preprocessing to add explicit file paths.

---

# Acknowledgements & References

- Code & ideas inspired by: **SatMAE** - [https://github.com/sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE)
- Datasets: Functional Map of the World (fMoW) - [https://github.com/fMoW/dataset](https://github.com/fMoW/dataset)
- fMoW hosted S3: `s3://spacenet-dataset/Hosted-Datasets/fmow/`
- Sentinel PURL: [https://purl.stanford.edu/vg497cb6002](https://purl.stanford.edu/vg497cb6002)
- Metadata Google Drive: [https://drive.google.com/drive/folders/1-xSXNpq0xJ4z3F7BPzEcZ04eZ7LqPbYD](https://drive.google.com/drive/folders/1-xSXNpq0xJ4z3F7BPzEcZ04eZ7LqPbYD)


