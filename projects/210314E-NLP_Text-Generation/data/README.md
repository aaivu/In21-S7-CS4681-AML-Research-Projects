# Datasets

This project utilizes six benchmark summarization datasets to evaluate the performance of the PEGASUS-X model across various domains, lengths, and styles. These datasets enable comprehensive assessment of abstractive summarization capabilities, ranging from short news articles to long-form documents.

## Dataset Descriptions

### GovReport
- **Description**: Contains long-form government reports and policy documents summarized into concise executive summaries.
- **Size**: 17,000 training samples, 1,000 validation samples, 1,000 test samples.
- **Characteristics**: Documents average around 9,000 tokens, summaries approximately 500 tokens. Evaluates model scalability and gradient dynamics in long-context settings.

### CNN/DailyMail
- **Description**: Consists of online news articles paired with multi-sentence highlights written by journalists.
- **Size**: Approximately 287,000 training samples, 13,000 validation samples, 11,000 test samples.
- **Characteristics**: Articles average about 760 tokens, summaries around 60 tokens. Measures ability to produce coherent, factual, and moderately abstractive summaries.
- **Note**: Due to computational constraints, only a subset was used for fine-tuning.

### XSum
- **Description**: Contains BBC news articles with single-sentence abstractive summaries designed to capture the core message.
- **Size**: Roughly 204,000 training samples, 11,000 validation samples, 11,000 test samples.
- **Characteristics**: Documents average 430 tokens, requiring concise, information-dense summaries.

### SummScreen
- **Description**: Dialogue-centric dataset from television and movie transcripts paired with human-written recaps.
- **Size**: Approximately 26,000 examples.
- **Characteristics**: Transcripts often exceed 6,000 tokens per episode. Tests handling of extended contexts and conversational structures.

### QMSum
- **Description**: Human-annotated benchmark for long transcripts of meetings across academic, product, and committee domains.
- **Size**: 1,808 query-summary pairs derived from 232 meetings.
- **Characteristics**: Average input length around 9,100 words, summaries approximately 70 words. Evaluates long-context dialogues and focused summarization.

### BIGPATENT
- **Description**: Large-scale collection of U.S. patent documents for abstractive summarization.
- **Size**: Approximately 1.3 million records.
- **Characteristics**: Input is detailed patent descriptions, summaries are human-written abstracts. Challenges models with specialized, long-form technical text and complex discourse.
- **Note**: Due to computational constraints, only a subset was used for fine-tuning.

## Data Preprocessing
All datasets underwent standardized preprocessing:
- Cleaning: Removal of HTML tags, escape characters, and extra whitespace.
- Tokenization: Using PEGASUS tokenizer with 96,000 subword units.
- Lowercasing for consistency.
- Truncation/Padding: Dataset-specific max input lengths (e.g., GovReport: 12,288 tokens, XSum: 1,024 tokens).
- Official train/validation/test splits are preserved.

## Download Instructions
To download the datasets, use
```
git lfs pull
```
