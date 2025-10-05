## Datasets

There are two datasets used for FinBERT. The language model further training is done on a subset of Reuters TRC2
dataset. This dataset is not public, but researchers can apply for access
[here](https://trec.nist.gov/data/reuters/reuters.html).

For the sentiment analysis, we used Financial PhraseBank from [Malo et al. (2014)](https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts).
The dataset can be downloaded from this [link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list).
If you want to train the model on the same dataset, after downloading it, you should create three files under the
`data/sentiment_data` folder as `train.csv`, `validation.csv`, `test.csv`.
To create these files, do the following steps:

- Download the Financial PhraseBank from the above link.
- Get the path of `Sentences_50Agree.txt` file in the `FinancialPhraseBank-v1.0` zip.
- Run the [datasets script](scripts/datasets.py):
