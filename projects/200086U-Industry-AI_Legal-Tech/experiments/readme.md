## Experiments

This section details the dataset, configurations, and methodology used to evaluate multi-caption bootstrapping for BLIP.

### Data and Setup

* [cite_start]**Dataset:** The study utilizes the **COCO 2017 dataset**[cite: 118].
* [cite_start]**Data Split:** All experiments follow the standard **Karpathy split** [cite: 118][cite_start], which consists of 113k training images, 5k validation images, and 5k test images[cite: 118].
* [cite_start]**Baseline Data:** The baseline model is trained on the standard Karpathy training split, where each image is paired with a **single unique human-annotated caption** to ensure uniformity across experiments[cite: 119].

### Synthetic Caption Generation

To construct the augmented datasets, a parallel set of synthetic captions was generated.

* [cite_start]**Model:** A pre-trained BLIP caption generator (using the `checkpoint_best.pth` from COCO fine-tuning) was employed in decoding mode[cite: 124, 125].
* [cite_start]**Method:** One caption was generated for each of the 113k training images[cite: 126, 135].
* [cite_start]**Parameters:** Captions were generated using **nucleus sampling** ($p=0.9$) [cite: 126] [cite_start]with a maximum token length of 20 [cite: 126] [cite_start]and an input resolution of $384\times384$ pixels[cite: 126].

### Experimental Configurations

[cite_start]Four distinct fine-tuning configurations were systematically compared to evaluate the influence of synthetic captions[cite: 9, 31, 137]:

1.  [cite_start]**Human-only (Baseline):** The BLIP model was fine-tuned on the standard 113k human-annotated captions from the Karpathy split[cite: 9, 31].
2.  [cite_start]**Synthetic-only:** The model was fine-tuned exclusively on the 113k BLIP-generated synthetic captions[cite: 9, 31, 138].
3.  [cite_start]**Hybrid (1:1):** A mixed dataset was formed by sampling 50% human-annotated captions and 50% synthetic captions [cite: 9, 139][cite_start], preserving the total dataset size at ~113k pairs[cite: 139]. [cite_start]Each image in this set was paired with only one caption, either human or synthetic[cite: 140].
4.  [cite_start]**Multi-caption:** This dataset paired *every* image with *both* its human-annotated caption and its synthetic caption [cite: 9, 31, 141][cite_start], effectively doubling the training data to ~226k captions[cite: 141].

### Training and Evaluation

* [cite_start]**Model:** The experiments use the BLIP model with a **ViT-B/16** vision backbone[cite: 120].
* [cite_start]**Baseline Training:** The baseline was fine-tuned for **two epochs** [cite: 121] [cite_start]using the AdamW optimizer [cite: 121][cite_start], a learning rate of $8\times10^{-5}$ [cite: 121][cite_start], and a batch size of 50 [cite: 121][cite_start], run on 2×NVIDIA A100 GPUs[cite: 121].
* [cite_start]**Experiment Training:** All other configurations (Synthetic-only, Hybrid, Multi-caption) were fine-tuned using identical optimizer parameters, batch size, and learning rate as the baseline[cite: 143].
* [cite_start]**Evaluation Metrics:** All models were evaluated on the COCO validation and test splits using the standard suite of captioning metrics: **BLEU** [cite: 144][cite_start], **METEOR** [cite: 144][cite_start], **ROUGE-L** [cite: 144][cite_start], **CIDEr** [cite: 144][cite_start], and **SPICE**[cite: 144].