# Experimental Setup and Procedures

This document outlines the experimental setup and procedures followed to evaluate the NCF-SSL model against the Neural Collaborative Filtering (NCF) baseline.

## 1. Objective

The primary objective of these experiments was to,
1.  Quantitatively compare the recommendation performance of NCF-SSL against the NCF baseline using standard top-K ranking metrics.
2.  Analyze the characteristics of the learned user and item embeddings to understand the regularization effects of the self-supervised contrastive learning task.
3.  Study the impact of key hyperparameters specific to the self-supervised component ($p_{drop}$ and $\lambda$).

## 2. Datasets

Two publicly available benchmark datasets were used:

* **MovieLens 1M:** Contains 1,000,209 anonymous ratings from 6,040 users on 3,952 movies.
* **Pinterest-20:** A larger dataset, also representing implicit feedback, which typically poses a greater challenge for recommendation models due to higher sparsity. (Note: Specific details like number of users/items/interactions for Pinterest-20 would be good to add if available).

### Data Preprocessing:
For both datasets, explicit ratings were converted to implicit feedback (any interaction is positive). A **leave-one-out** evaluation strategy was adopted, where for each user, their last interacted item was held out for testing, and the remaining interactions formed the training set. For each positive interaction in the training set, **49 negative (uninteracted) items** were randomly sampled to balance the training data, following common practice in NCF research.

## 3. Models Evaluated

1.  **NCF (Baseline):** The Neural Collaborative Filtering model, specifically its NeuMF variant, which combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) components. This serves as the direct baseline for comparison.
2.  **NCF-SSL (Proposed):** Our enhanced NCF model, which integrates a self-supervised contrastive learning task. It utilizes embedding dropout to create augmented views and an InfoNCE loss to regularize the user and item embeddings.

## 4. Training Details

* **Optimizer:** Adam optimizer was used for training both models.
* **Learning Rate:** (Specify learning rate, e.g., 0.001)
* **Batch Size:** (Specify batch size, e.g., 256 or 512, particularly relevant for contrastive learning)
* **Epochs:** Models were trained for a fixed number of epochs (e.g., 100 epochs) with early stopping based on validation set performance (NDCG@10).
* **Negative Sampling in Test:** For evaluation, for each user's test item, 99 randomly sampled negative items were combined with the positive test item to form a list of 100 items for ranking.

## 5. Hyperparameters

### Shared Hyperparameters:
* **Predictive Factors (Embedding Dimension):** Evaluated at 8, 16, and 32 to study the impact of model capacity.
* **MLP Layer Sizes:** (e.g., [64, 32, 16] for a model with 16 factors, corresponding to the NCF paper's structure).

### NCF-SSL Specific Hyperparameters:
* **Dropout Rate for Augmentation ($p_{drop}$):** Varied (e.g., 0.1, 0.2, 0.3, 0.4, 0.5) to find the optimal level of perturbation for creating augmented views.
* **Contrastive Loss Weight ($\lambda$):** Varied (e.g., 0.2, 0.4, 0.6, 0.8, 1.0) to control the balance between the primary recommendation task and the auxiliary self-supervised task.
* **Temperature Parameter ($\tau$):** (Specify value, e.g., 0.5) for the InfoNCE loss.

## 6. Evaluation Metrics

Top-K ranking metrics were used to assess model performance:

* **Hit Ratio at 10 (HR@10):** Measures the recall of the top 10 recommended items.
* **Normalized Discounted Cumulative Gain at 10 (NDCG@10):** A position-aware metric that weights hits at higher ranks more heavily.

## 7. Analysis Procedures

Beyond quantitative metrics, the following analyses were performed,

* **Embedding Space Regularization:**
    * Calculated the standard deviation of user and item embedding norms for both NCF and NCF-SSL.
    * Measured the average cosine similarity between augmented positive pairs (same user/item, different views) in NCF-SSL.
* **Qualitative Visualization:**
    * Used t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce high-dimensional user and item embeddings to 2D for visual inspection of clustering and separation.
* **Mitigation of Data Sparsity:**
    * Analyzed the average cosine similarity to nearest neighbors for cold-start users (e.g., those with <10 interactions) in both models to assess embedding robustness.

## 8. Computational Environment

* **Hardware:** Experiments were conducted on Google Colab using the available Nvidia Tesla T4 GPU
* **Software:** Python 3.8+, PyTorch (version 2.8.0), NumPy

This setup ensures that the experiments are reproducible and provides a clear basis for interpreting the results.