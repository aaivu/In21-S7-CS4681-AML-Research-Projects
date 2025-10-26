# Results

## Final Results Description

The final results from this study evaluate four activation functions on the PEGASUS-X model's abstractive summarization performance across six benchmark datasets. Summarization quality was measured using ROUGE-1, ROUGE-2, and ROUGE-Lsum metrics.

Overall, GELU consistently achieved the highest ROUGE scores across most datasets, particularly excelling on longer, complex texts like GovReport and SummScreen. Its smooth probabilistic gating enhances gradient flow and semantic representation, leading to more effective modeling of non-linear relationships in abstractive summarization tasks. GELU New showed marginal improvements over GELU on shorter-text datasets (CNN/DailyMail and XSum), suggesting better sensitivity to local linguistic variations in concise summarization tasks. ReLU performed stably but lower than GELU, offering computational efficiency while suffering from limitations like inactive neurons in negative input regions. SiLU yielded the lowest scores overall, with reduced gradient responsiveness potentially hindering contextual signal propagation in deep transformer layers.

These results highlight that activation function choice significantly influences model performance, with GELU variants providing the best balance of stability and expressiveness for abstractive summarization, especially in handling diverse text lengths and complexities.

## ROUGE-1 Scores Across Different Activation Functions

| **Dataset**      | **ReLU** | **GELU** | **GELU New** | **SiLU** |
|------------------|----------|----------|--------------|----------|
| XSum            | 37.4    | 45.8    | **46.2**    | 22.9    |
| CNN/DailyMail   | 34.8    | 43.4    | **43.9**    | 25.1    |
| QMSum           | 26.5    | **32.9** | 32.4        | 17.8    |
| SummScreen      | 27.2    | **35.0** | 34.5        | 18.3    |
| GovReport       | 41.5    | **59.3** | 58.7        | 31.8    |
| BIGPATENT       | 48.4    | **61.3** | 60.6        | 30.1    |

## ROUGE-2 Scores Across Different Activation Functions

| **Dataset**      | **ReLU** | **GELU** | **GELU New** | **SiLU** |
|------------------|----------|----------|--------------|----------|
| XSum            | 17.2    | 22.8    | **23.1**    | 10.4    |
| CNN/DailyMail   | 18.8    | 21.2    | **21.5**    | 10.2    |
| QMSum           | 7.4     | **9.8**  | 9.5         | 2.9     |
| SummScreen      | 4.8     | **8.9**  | 8.5         | 2.3     |
| GovReport       | 20.1    | **29.3** | 28.7        | 13.2    |
| BIGPATENT       | 37.4    | **42.6** | 41.9        | 22.9    |

## ROUGE-Lsum Scores Across Different Activation Functions

| **Dataset**      | **ReLU** | **GELU** | **GELU New** | **SiLU** |
|------------------|----------|----------|--------------|----------|
| XSum            | 29.6    | 37.6    | **38.0**    | 20.1    |
| CNN/DailyMail   | 30.5    | 40.6    | **41.2**    | 21.9    |
| QMSum           | 17.4    | **21.4** | 21.0        | 12.8    |
| SummScreen      | 13.2    | **20.4** | 19.9        | 8.9     |
| GovReport       | 21.8    | **30.9** | 30.1        | 14.6    |
| BIGPATENT       | 37.4    | **50.1** | 49.4        | 23.7    |
