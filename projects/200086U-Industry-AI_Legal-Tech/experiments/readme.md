## Experiments

This section details the dataset, configurations, and methodology used to evaluate multi-caption bootstrapping for BLIP.

### Data and Setup

* **Dataset:** The study utilizes the **COCO 2017 dataset**.
* **Data Split:** All experiments follow the standard **Karpathy split**, which consists of 113k training images, 5k validation images, and 5k test images.
* **Baseline Data:** The baseline model is trained on the standard Karpathy training split, where each image is paired with a **single unique human-annotated caption** to ensure uniformity across experiments.

### Synthetic Caption Generation

To construct the augmented datasets, a parallel set of synthetic captions was generated.

* **Model:** A pre-trained BLIP caption generator (using the `checkpoint_best.pth` from COCO fine-tuning) was employed in decoding mode.
* **Method:** One caption was generated for each of the 113k training images.
* **Parameters:** Captions were generated using **nucleus sampling** ($p=0.9$) with a maximum token length of 20 and an input resolution of $384\times384$ pixels.

### Experimental Configurations

Four distinct fine-tuning configurations were systematically compared to evaluate the influence of synthetic captions:

1.  **Human-only (Baseline):** The BLIP model was fine-tuned on the standard 113k human-annotated captions from the Karpathy split.
2.  **Synthetic-only:** The model was fine-tuned exclusively on the 113k BLIP-generated synthetic captions.
3.  **Hybrid (1:1):** A mixed dataset was formed by sampling 50% human-annotated captions and 50% synthetic captions, preserving the total dataset size at ~113k pairs. Each image in this set was paired with only one caption, either human or synthetic.
4.  **Multi-caption:** This dataset paired *every* image with *both* its human-annotated caption and its synthetic caption, effectively doubling the training data to ~226k captions.

### Training and Evaluation

* **Model:** The experiments use the BLIP model with a **ViT-B/16** vision backbone.
* **Experiment Training:** All other configurations (Synthetic-only, Hybrid, Multi-caption) were fine-tuned using identical optimizer parameters, batch size, and learning rate as the baseline.
* **Evaluation Metrics:** All models were evaluated on the COCO validation and test splits using the standard suite of captioning metrics: **BLEU**, **METEOR**, **ROUGE-L**, **CIDEr**, and **SPICE**.
