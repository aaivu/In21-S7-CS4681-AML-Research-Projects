# Neural Collaborative Filtering

This is a pytorch implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling. 

For more details on the original implementation, refer to the [official GitHub repository](https://github.com/hexiangnan/neural_collaborative_filtering).

**Please cite the original WWW'17 paper if you use the codes. Thanks!** 

Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

## Updates

### PyTorch Implementation
This repository now includes a **PyTorch implementation** of the Neural Collaborative Filtering models, providing:
- Modern deep learning framework support
- GPU acceleration capabilities
- Improved code readability and maintainability
- Compatible evaluation metrics (Hit Ratio and NDCG)

### Multi-Task Learning Extension (NeuMF_MTL)
We have extended the original NeuMF model with a **Multi-Task Learning (MTL)** variant that jointly optimizes for:
1. **Ranking task**: Binary classification for implicit feedback (original task)
2. **Rating prediction task**: Explicit rating prediction as an auxiliary task

The MTL approach can potentially improve model generalization by learning richer user-item representations through joint optimization of related tasks.

## Environment Settings

### PyTorch Implementation
- PyTorch version: '1.0+' (recommended: latest stable version)
- Python version: '3.6+' (recommended: 3.8 or higher)
- CUDA support: Optional (for GPU acceleration)

### PyTorch Implementation Examples

Run NeuMF with PyTorch:
```
python NeuMF_pytorch.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 16 --layers [32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1
```

Run NeuMF_MTL (Multi-Task Learning):
```
python NeuMF_MTL.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 16 --layers [32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 
```

### Analyse Results

After training the models, you can analyze and visualize the learned embeddings using the provided analysis tool. The analysis script generates comprehensive visualizations and statistics for both MF (Matrix Factorization) and MLP (Multi-Layer Perceptron) embeddings.

#### Basic Usage

Analyze the latest trained NeuMF_MTL model:
```
python analyse.py --dataset ml-1m --model_type NeuMF_MTL
```

Analyze a specific saved model:
```
python analyse.py --dataset ml-1m --model_path Pretrain/ml-1m_NeuMF_MTL_20241215.pt --model_type NeuMF_MTL
```

Analyze regular NeuMF model:
```
python analyse.py --dataset ml-1m --model_type NeuMF --model_path Pretrain/ml-1m_NeuMF_20241215.pt
```

#### Available Parameters

- `--path`: Input data path (default: `Data/`)
- `--dataset`: Dataset name (default: `ml-1m`)
- `--model_path`: Path to saved model (if empty, uses the latest model in `Pretrain/`)
- `--num_factors`: Embedding size of MF model (default: 8)
- `--layers`: MLP layers configuration (default: `[64,32,16,8]`)
- `--dropout`: Dropout rate (default: 0.2)
- `--model_type`: Type of model - `NeuMF_MTL` or `NeuMF` (default: `NeuMF_MTL`)
- `--sample_size`: Number of samples to visualize (default: 1000)
- `--output_dir`: Directory to save visualizations (default: `visualizations/`)

#### Generated Visualizations

The analysis script generates the following visualizations in the output directory:

1. **t-SNE Embeddings** (`tsne_embeddings.png`)
   - 2D t-SNE projection of MF and MLP embeddings
   - Visualizes clustering patterns for users and items
   - Helps identify if similar users/items are grouped together

2. **PCA Embeddings** (`pca_embeddings.png`)
   - 2D PCA projection with explained variance ratios
   - Shows principal components and data spread
   - Indicates dimensionality and information retention

3. **Embedding Norms** (`embedding_norms.png`)
   - Distribution of L2 norms for each embedding type
   - Includes mean and median statistics
   - Helps detect embedding magnitude patterns

4. **Norm Comparison** (`norm_comparison.png`)
   - Box plots and violin plots comparing norms across embedding types
   - Identifies differences between MF and MLP embeddings
   - Shows variability in embedding magnitudes

5. **Embedding Heatmaps** (`embedding_heatmaps.png`)
   - Heatmap visualization of embedding values
   - Shows activation patterns across dimensions
   - Useful for detecting sparse or dense representations

6. **Nearest Neighbor Cosine Similarity** (`nn_cosine_similarity.png`)
   - Distribution of average cosine similarity to k-nearest neighbors
   - Indicates embedding space density and separation
   - Higher values suggest more clustered embeddings

7. **Nearest Neighbor Euclidean Distance** (`nn_euclidean_distance.png`)
   - Distribution of average Euclidean distance to k-nearest neighbors
   - Complements cosine similarity analysis
   - Shows absolute distance patterns in embedding space

8. **NN Comparison** (`nn_comparison.png`)
   - Side-by-side comparison of similarity metrics
   - Box plots for both cosine similarity and Euclidean distance
   - Facilitates cross-embedding type comparison

9. **Embedding Statistics** (`embedding_statistics.txt`)
   - Comprehensive text report with numerical statistics
   - Includes mean, std, min, max, median for all metrics
   - Contains nearest neighbor statistics (k=10 by default)

#### Example Analysis Workflow

1. Train your model:
   ```
   python NeuMF_MTL.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 16 --layers [32,16,8] --lr 0.001
   ```

2. Run analysis on the trained model:
   ```
   python analyse.py --dataset ml-1m --model_type NeuMF_MTL --sample_size 2000
   ```

3. Compare with baseline NeuMF:
   ```
   python analyse.py --dataset ml-1m --model_type NeuMF --model_path Pretrain/ml-1m_NeuMF_baseline.pt --sample_size 2000
   ```

4. Review visualizations in the `visualizations/` directory and statistics in `embedding_statistics.txt`

#### Advanced Analysis

For custom sample sizes (useful for large datasets):
```
python analyse.py --dataset ml-1m --sample_size 5000 --output_dir visualizations/large_sample/
```

For specific model configurations:
```
python analyse.py --dataset pinterest-20 --num_factors 32 --layers [128,64,32,16] --model_path Pretrain/custom_model.pt
```

## Dataset
Two processed datasets in the orignal repo are included here: MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20). 

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

Last Update Date: October 15, 2025