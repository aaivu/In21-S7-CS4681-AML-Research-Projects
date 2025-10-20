# Dataset Creation for Low-Resource Speech Recognition

## Overview

This document explains the **dataset creation process** used to prepare low-resource subsets of the LibriSpeech dataset (or a custom `.parquet` dataset) for speech recognition research.  
The goal of this step is to build **smaller, manageable datasets** (e.g., 10 minutes, 1 hour, and 10 hours of audio) that can be used for **pretraining and fine-tuning** Wav2Vec2 models efficiently in limited-resource environments.

---

## Objectives

1. **Simulate low-resource conditions:** Create small, nested subsets of speech data to test how model performance changes with data availability.  
2. **Ensure reproducibility:** Use fixed seeds and deterministic dataset sampling.  
3. **Reduce computational cost:** Allow training on smaller, representative subsets instead of the full dataset.  
4. **Prepare standardized data:** Make sure each audio sample has a computed duration and consistent format.

---

## Tools and Dependencies

The process uses open-source Python libraries commonly available in machine learning environments:

- **Transformers** — for later fine-tuning Wav2Vec2 models.  
- **Datasets** — for efficient dataset handling and storage.  
- **Torchaudio** and **Torchcodec** — for audio loading, processing, and potential feature extraction.  
- **Google Drive (Colab)** — to store the processed datasets persistently.  

All dependencies can be installed with:
```bash
pip install transformers datasets torchaudio torchcodec
Steps Performed
1. Mounting Google Drive
The first step connects the Colab environment to Google Drive so that all processed data can be saved permanently.
This ensures datasets won’t be lost when the Colab runtime resets.

2. Loading the Dataset
Two dataset options are supported:

Option 1: LibriSpeech (librispeech_asr) — a large, clean English speech dataset.

Option 2: Custom .parquet dataset — user-provided dataset (in this case, /content/drive/MyDrive/SP/0000.parquet).

The dataset is loaded using the Hugging Face datasets library, which allows efficient streaming, filtering, and saving.

3. Adding Duration Information
Each audio sample has an “array” (the waveform) and a “sampling_rate”.
A new field called duration is added to each entry by calculating:

Duration = (length of audio array) ÷ (sampling rate)

This helps the script track how much total speech time each subset covers.

4. Creating Nested Subsets
The dataset is then divided into smaller subsets based on total audio duration — for example:

600 seconds (10 minutes)

3600 seconds (1 hour)

36,000 seconds (10 hours)

These subsets are nested, meaning:

The 10-minute dataset is fully contained inside the 1-hour dataset,
and the 1-hour dataset is fully contained inside the 10-hour dataset.

This design is important for experiments where models are trained progressively with increasing amounts of data.

5. Shuffling and Sampling Logic
The dataset is shuffled randomly (using a fixed random seed = 42) to ensure:

There is no bias in which samples get selected first.

The selection process is reproducible.

The script then iterates through the shuffled dataset, continuously adding samples until each target duration (e.g., 10 minutes) is reached.

Once one duration target is met, it continues collecting data for the next (larger) subset.

6. Saving the Datasets
After the subsets are created, they are saved to disk in Hugging Face Arrow format using save_to_disk().
This format is fast to load, supports large files, and works directly with the datasets library during training.

Example directories:

swift
Copy code
/content/drive/MyDrive/SP/librispeech_datasets/dataset_10min
/content/drive/MyDrive/SP/librispeech_datasets/dataset_1h
/content/drive/MyDrive/SP/librispeech_datasets/dataset_10h
In addition to the training subsets, the script also saves:

Validation set — from LibriSpeech “validation-clean”

Test set — from LibriSpeech “test-clean”

These are used to evaluate model performance and ensure consistent testing conditions.

What You Have Achieved
By running this pipeline, you now have:

Dataset	Duration	Purpose
10-minute subset	600 seconds	Fast testing and debugging
1-hour subset	3600 seconds	Low-resource pretraining
10-hour subset	36,000 seconds	Standard small-scale experiment
Validation set	-	Evaluation during training
Test set	-	Final model performance check

These datasets are compact, reproducible, and compatible with any Wav2Vec2-based speech recognition model.

Next Steps
After creating the datasets, the next phases in your research pipeline should be:

Pretraining Stage
Use the 10-hour dataset to pretrain the Wav2Vec2 model with your Inter-Codebook Similarity Loss (ICSL) to improve representation learning efficiency.

Fine-tuning Stage
Fine-tune the pretrained model using the same 10-hour or smaller subsets, applying Residual Vector Quantization (RVQ) for better low-resource adaptation.

Evaluation
Measure model performance using metrics such as CTC Loss and Word Error Rate (WER) on the validation and test sets.

Comparison
Compare results between:

Baseline Wav2Vec2 model

Wav2Vec2 + ICSL (pretraining improvement)

Wav2Vec2 + RVQ (fine-tuning improvement)

Recommendations and Improvements
Data diversity: If possible, include different speakers, accents, or domains to enhance generalization.

Filtering: Avoid extremely short utterances (e.g., less than 2 seconds) as they often cause issues in contrastive training.

Metadata tracking: Store dataset statistics (e.g., number of samples, total duration) for documentation and reproducibility.

Augmentation (optional): Consider applying noise, pitch shift, or speed perturbation to simulate real-world variability.

Summary
This dataset creation process is a crucial foundation for your low-resource speech recognition project.
It enables scalable experiments by providing consistent, high-quality subsets of speech data — ready for pretraining, fine-tuning, and evaluation of advanced models like Wav2Vec2.

Author: Rasara Thathsarana
Institution: Department of Computer Science and Engineering, University of Moratuwa
Purpose: Preparing low-resource speech datasets for Wav2Vec2 experiments involving Inter-Codebook Similarity Loss (ICSL) and Residual Vector Quantization (RVQ).
