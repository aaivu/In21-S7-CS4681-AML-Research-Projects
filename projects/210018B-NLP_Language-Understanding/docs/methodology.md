# Methodology: NLP: Language Understanding

**Student:** 210018B (Abisherk Sivakumar)  
**Research Area:** NLP: Language Understanding - One-Shot Learning and Example Selection  
**Date:** 2025-01-20

## 1. Overview

This research investigates the optimization of one-shot learning for the LAMBADA dataset through intelligent example selection strategies. Unlike traditional random selection approaches, we propose a hybrid methodology that combines semantic similarity (measured via sentence embeddings) and syntactic compatibility (assessed through Part-of-Speech matching) to select optimal demonstration examples.

**Core Research Question:**

> Can strategic example selection combining semantic and syntactic features significantly improve one-shot learning performance on discourse-level word prediction tasks?

**Methodological Approach:**

We adopt a quantitative experimental methodology with controlled comparisons between:
* Random baseline selection (standard approach)
* Semantic-only selection (embedding-based retrieval)
* Syntactic-only selection (POS-matching)
* Hybrid selection (weighted combination of semantic and syntactic factors)

The methodology involves four main components:

1.  **Data preparation:** Cleaning and preprocessing LAMBADA validation set for candidate examples
2.  **Feature engineering:** Precomputing semantic embeddings and syntactic POS tags
3.  **Selection algorithm:** Hybrid scoring mechanism with tunable weighting parameter
4.  **Evaluation protocol:** Systematic accuracy measurement across full test set with comparative analysis

## 2. Research Design

### 2.1 Research Type

Experimental Research Design with quantitative evaluation and systematic ablation studies.

### 2.2 Research Paradigm

Empirical NLP Research: Testing hypotheses about example selection effectiveness through controlled experiments on a standardized benchmark.

### 2.3 Experimental Framework

```
Input: Test context C_test
       ↓
[Example Selection Module]
  ├→ Semantic Similarity (SBERT embeddings)
  ├→ Syntactic Compatibility (POS matching)
  └→ Hybrid Score = α·Semantic + (1-α)·Syntactic
       ↓
Selected Example: C_demo, w_demo
       ↓
[Prompt Construction]
  → Cloze-style format with instructions
       ↓
[LLM Query]
  → GPT-3.5-Turbo-Instruct (temperature=0)
       ↓
Prediction: w_pred
       ↓
[Evaluation]
  → Accuracy = ⊮(w_pred == w_gold)
```

### 2.4 Variables

**Independent Variables:**
* **Selection strategy:** {Random, Semantic, Syntactic, Hybrid}
* **Weighting parameter α:** [0.0, 1.0]
* **Prompt format:** {Cloze, Default}
* **Number of shots k:** {0, 1}

**Dependent Variable:**
* **Accuracy** (percentage of exact word matches)

**Control Variables:**
* **Model:** GPT-3.5-Turbo-Instruct (fixed)
* **Temperature:** 0.0 (deterministic)
* **Max tokens:** 3 (single-word predictions)
* **Dataset:** LAMBADA test set (5,153 examples)
* **Evaluation metric:** Case-insensitive exact match

### 2.5 Hypotheses

* **H1 (Primary):** Hybrid selection (semantic + syntactic) achieves higher accuracy than random selection.
* **H2:** Syntactic compatibility is more important than semantic similarity for word prediction tasks.
* **H3:** There exists an optimal weighting α ≠ 0.5 that outperforms equal weighting.
* **H4:** Strategic one-shot learning can match or exceed originally reported GPT-3 one-shot performance (72.5%).

## 3. Data Collection

### 3.1 Data Sources

**Primary Dataset: LAMBADA**
* **Source:** HuggingFace `datasets` library (`lambada` dataset)
* **Original paper:** Paperno et al., 2016
* **License:** Public domain (extracted from BookCorpus)
* **Access method:** `datasets.load_dataset("lambada", split="test"/"validation")`

**Validation Set (Candidate Pool):**
* **Size:** 4,869 examples (before cleaning)
* **Purpose:** Source of demonstration examples for one-shot learning
* **Preprocessing:** Filtered to 4,217 clean examples

**Test Set (Evaluation):**
* **Size:** 5,153 examples
* **Purpose:** Final evaluation of selection strategies
* **Preprocessing:** Minimal (used as-is for fair comparison)

### 3.2 Data Description

**LAMBADA Characteristics:**
* **Task:** Predict the final word of a passage requiring discourse understanding
* **Passage length:** Average 4.6 sentences (75.4 words)
* **Target words:** Single words (nouns, verbs, adjectives, names)
* **Difficulty:** Target unpredictable from final sentence alone but guessable with full context
* **Domain:** Fiction (novels from BookCorpus)

**Example:**
> **Context:** "Yes, I thought to myself as I stared at his blood-stained shirt. He was definitely going to need a new..."
>
> **Target:** shirt

**POS Distribution in Validation Set (post-cleaning):**
* **NOUN:** 2,847 examples (67.5%)
* **VERB:** 892 examples (21.2%)
* **ADJ:** 287 examples (6.8%)
* **PROPN:** 191 examples (4.5%)

### 3.3 Data Preprocessing

#### 3.3.1 Validation Set Cleaning

**Objective:** Create high-quality candidate pool for example selection

**Filtering Criteria:**
```python
def is_valid_example(text):
    if " " not in text:
        return False  # Skip empty or single-word passages
    
    context, target = text.rsplit(" ", 1)
    
    # Filter multi-word targets
    if len(target.split()) > 1:
        return False
    
    # Filter punctuation-heavy targets
    if not re.match(r"^[A-Za-z'-]+$", target):
        return False
    
    return True
```

**Cleaning Results:**
* **Original validation examples:** 4,869
* **After filtering:** 4,217 (86.6% retained)
* **Removed:** 652 examples (13.4%)

**Rationale:**
* Multi-word targets don't fit single-word prediction format
* Punctuation causes inconsistent tokenization
* Clean examples ensure fair selection comparisons

#### 3.3.2 Text Normalization

**Applied to all examples:**
* Whitespace normalization (strip leading/trailing spaces)
* Consistent lowercasing for evaluation (predictions and gold labels)
* Preserve case in prompts (maintain natural text)

**Not applied:**
* Stemming/lemmatization (preserve word forms)
* Stop word removal (context matters for discourse)
* Special character removal from contexts (maintain original text)

#### 3.3.3 Feature Extraction

**Semantic Features:**
```python
# Precompute sentence embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
dev_embeddings = embedder.encode(
    dev_texts,
    convert_to_tensor=True,
    show_progress_bar=True,
    batch_size=32
)
# Dimensions: (4217, 384)
```

**Syntactic Features:**
```python
# Extract POS tags for target words
nlp = spacy.load("en_core_web_sm")
dev_pos = []
for text in dev_texts:
    doc = nlp(text)
    pos_tag = doc[-1].pos_ if len(doc) > 0 else "X"
    dev_pos.append(pos_tag)
```

**Storage:**
* **Embeddings:** Torch tensor on GPU (if available) or CPU
* **POS tags:** Python list (memory-efficient)
* **Total preprocessing time:** ~8 minutes (one-time cost)

## 4. Model Architecture

### 4.1 System Components

Our methodology doesn't train a new model but rather optimizes example selection for an existing LLM. The architecture consists of four modules:

#### 4.1.1 Example Selection Module

* **Input:** Test context `C_test`
* **Output:** Selected demonstration example (`C_demo`, `w_demo`)
* **Algorithm:**
    ```python
    def select_example(C_test, alpha=0.2):
        # Step 1: Compute semantic similarity
        test_embedding = embedder.encode(C_test)
        semantic_scores = cosine_similarity(
            test_embedding, 
            dev_embeddings
        )
        
        # Step 2: Compute syntactic compatibility
        test_pos = extract_pos(C_test)
        syntactic_scores = [
            1.0 if pos == test_pos else 0.0 
            for pos in dev_pos
        ]
        
        # Step 3: Combine with weighting
        hybrid_scores = (
            alpha * semantic_scores + 
            (1 - alpha) * syntactic_scores
        )
        
        # Step 4: Select best example
        best_idx = argmax(hybrid_scores)
        return validation_set[best_idx]
    ```

**Key Design Decisions:**
* **Additive combination:** Simple, interpretable, computationally efficient
* **Binary syntactic score:** Clear match/no-match criterion
* **Cosine similarity for semantics:** Standard for embedding comparison
* **Single best example:** One-shot learning constraint

#### 4.1.2 Prompt Construction Module

* **Input:** Selected example, test context, prompt mode
* **Output:** Formatted prompt string

**Cloze-Style Format (Primary):**
```
Below are examples where you must predict the final missing word.
Each passage ends with a blank (_____), and the correct word 
follows after '→'.
Your task: predict the final word for the last passage.
Rules:
- Output exactly ONE meaningful English word (noun, verb, or name).
- No punctuation or explanations.

[Context_demo] _____ → [Word_demo]

[Context_test] _____ →
```

**Design Principles:**
* Clear task framing (reduces ambiguity)
* Explicit output format specification
* Visual cue (`→`) for input-output mapping
* Single demonstration (one-shot constraint)

#### 4.1.3 LLM Query Module

* **Model:** `gpt-3.5-turbo-instruct` via OpenAI API
* **Configuration:**
    ```python
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=3,
        temperature=0.0,
        top_p=1.0,
    )
    ```

**Rationale for Parameters:**
* `max_tokens=3`: Accommodates single word + potential whitespace/punctuation
* `temperature=0.0`: Deterministic predictions for reproducibility
* `top_p=1.0`: No nucleus sampling (consider full distribution)

#### 4.1.4 Post-Processing Module

* **Input:** Raw model output
* **Output:** Cleaned prediction
    ```python
    def clean_prediction(text):
        # Remove whitespace
        text = text.strip()
        
        # Remove non-alphabetic characters
        text = re.sub(r"[^A-Za-z'-]+", "", text)
        
        # Lowercase for comparison
        text = text.lower()
        
        return text
    ```

### 4.2 Baseline Models

1.  **Baseline 1: Random Selection**
    * Uniform random sampling from validation set
    * No semantic or syntactic consideration
    * Standard approach in few-shot learning literature
2.  **Baseline 2: Zero-Shot**
    * No demonstration example provided
    * Prompt contains only instructions and test context
    * Tests whether examples help at all
3.  **Baseline 3: Semantic-Only Selection (α=1.0)**
    * Pure semantic similarity via embeddings
    * Ignores syntactic factors
    * Tests semantic-only hypothesis
4.  **Baseline 4: Syntactic-Only Selection (α=0.0)**
    * Pure POS matching
    * Ignores semantic content
    * Tests syntactic-only hypothesis
5.  **Baseline 5: Published GPT-3 Results**
    * Brown et al. 2020: 72.5% one-shot, 76.2% zero-shot
    * Historical comparison (different model)

## 5. Experimental Setup

### 5.1 Evaluation Metrics

#### 5.1.1 Primary Metric: Accuracy

**Definition:**
> Accuracy = (Number of exact matches) / (Total predictions) × 100%

**Matching Criteria:**
* Case-insensitive comparison
* Exact string match after normalization
* No partial credit for similar words

**Example:**
* **Gold:** "happy"
* **Prediction:** "happy" → Correct (1.0)
* **Prediction:** "Happy" → Correct (1.0, case-insensitive)
* **Prediction:** "pleased" → Incorrect (0.0, no partial credit)
* **Prediction:** "happ" → Incorrect (0.0, must be complete word)

**Justification:**
* Standard metric for LAMBADA (enables comparison with prior work)
* Clear, interpretable, no hyperparameters
* Aligned with task definition (predict the exact word)

### 5.3 Hardware/Software Requirements

#### 5.3.1 Hardware

* **Minimum Requirements:**
    * **CPU:** Modern x86-64 processor (e.g., Intel i5 or equivalent)
    * **RAM:** 8 GB (16 GB recommended for embedding computation)
    * **Storage:** 5 GB free space (for datasets, models, and results)
    * **GPU:** Optional (CUDA-compatible GPU accelerates embedding computation)
* **Recommended Configuration:**
    * **CPU:** Intel i7 or AMD Ryzen 7
    * **RAM:** 16 GB
    * **GPU:** NVIDIA RTX 3060 or better (for faster preprocessing)
    * **Storage:** SSD with 10 GB free space
* **Our Setup:**
    * **CPU:** Intel Ultra 9 185H
    * **RAM:** 16GB
    * **GPU:** CUDA 11.8 compatible (for torch)
    * **Storage:** SSD

#### 5.3.2 Software

* **Operating System:**
    * Linux (Ubuntu 20.04+) - Primary
    * macOS 11+ - Compatible
    * Windows 10+ - Compatible
* **Programming Language:**
    * Python 3.8 or higher (3.10 recommended)
* **Core Libraries:**
    * `openai==1.109.1`
    * `datasets==2.14.0`
    * `sentence-transformers==5.1.1`
    * `spacy==3.8.7`
    * `torch==2.0.1`
    * `python-dotenv==1.0.0`
    * `numpy==1.24.3`
    * `scipy==1.10.1`
* **Additional Models:**
    * spaCy language model: `python -m spacy download en_core_web_sm`
    * Sentence-BERT model: `all-MiniLM-L6-v2` (auto-downloaded)
* **Development Tools:**
    * Git (version control)
    * Jupyter Notebook (exploratory analysis)
    * Visual Studio Code or PyCharm (IDE)

#### 5.3.3 API Requirements

* **OpenAI API:**
    * Account with API access
    * API key (stored in `.env` file)
    * Usage limits: Free tier sufficient for development; paid tier recommended for full experiments
    * Estimated cost: $15-30 for complete experiments
* **HuggingFace:**
    * Optional token for dataset access (usually not required)
    * Free account sufficient

#### 5.3.4 Computational Costs

* **Preprocessing (one-time):** <10 minutes
    * Embedding computation: ~5 minutes (CPU) or ~1 minute (GPU)
    * POS tagging: ~3 minutes (CPU)
* **Evaluation (per strategy):**
    * 1,000 samples: ~45 minutes
    * Full test set (5,153 samples): ~4 hours
    * All experiments: ~20 hours
* **Storage:** <100 MB

## 6. Implementation Plan

| Phase                          | Tasks                                             | Duration | Deliverables                                 | Status        |
| ------------------------------ | ------------------------------------------------- | -------- | -------------------------------------------- | ------------- |
| **Phase 1: Setup & Preprocessing** |                                                   | **2 weeks** |                                              |               |
|                                | 1.1 Environment setup                           | 2 days   | Working Python environment                   | ✅ Complete   |
|                                | 1.2 Dataset loading and exploration             | 2 days   | Data statistics report                       | ✅ Complete   |
|                                | 1.3 Validation set cleaning                     | 2 days   | Clean candidate pool (4,217 examples)        | ✅ Complete   |
|                                | 1.4 Feature precomputation (embeddings, POS)    | 3 days   | Cached embeddings and POS tags               | ✅ Complete   |
|                                | 1.5 Unit testing for data pipeline              | 2 days   | Test suite with 95%+ coverage                | ✅ Complete   |
| **Phase 2: Core Implementation** |                                                   | **3 weeks** |                                              |               |
|                                | 2.1 Example selection module                    | 4 days   | Hybrid selection algorithm                   | ✅ Complete   |
|                                | 2.2 Prompt construction module                  | 3 days   | Cloze and default formatters                 | ✅ Complete   |
|                                | 2.3 LLM query interface                         | 2 days   | OpenAI API wrapper with error handling       | ✅ Complete   |
|                                | 2.4 Evaluation pipeline                         | 3 days   | Accuracy computation and logging             | ✅ Complete   |
|                                | 2.5 Integration testing                         | 3 days   | End-to-end test on small subset              | ✅ Complete   |
|                                | 2.6 Code refactoring and documentation          | 3 days   | Clean, documented codebase                   | ✅ Complete   |
| **Phase 3: Experimentation** |                                                   | **3 weeks** |                                              |               |
|                                | 3.1 Baseline experiments (random, zero-shot)    | 3 days   | Baseline accuracy results                    | ⏳ In Progress |
|                                | 3.2 Single-factor experiments (semantic, syntactic) | 3 days   | Individual strategy results                  | ⏳ Planned     |
|                                | 3.3 Hybrid experiments (α sweep)                | 4 days   | Optimal α identification                     | ⏳ Planned     |
|                                | 3.4 Full test set evaluation                    | 5 days   | Complete accuracy results (all strategies)   | ⏳ Planned     |
|                                | 3.5 Prompt format comparison                    | 2 days   | Cloze vs. default results                    | ⏳ Planned     |
|                                | 3.6 Error analysis                              | 3 days   | Per-POS accuracy, error categorization       | ⏳ Planned     |
| **Phase 4: Analysis & Documentation** |                                                   | **2 weeks** |                                              |               |
|                                | 4.1 Statistical significance testing            | 2 days   | p-values for comparisons                     | ⏳ Planned     |
|                                | 4.2 Visualization and charts                    | 3 days   | Performance graphs, POS distribution         | ⏳ Planned     |
|                                | 4.3 Literature comparison                       | 2 days   | Comparison with GPT-3 and related work       | ⏳ Planned     |
|                                | 4.4 Results interpretation                      | 3 days   | Discussion of findings                       | ⏳ Planned     |
|                                | 4.5 Paper writing                               | 4 days   | Complete research paper draft                | ⏳ Planned     |
|                                | 4.6 Code release preparation                    | 2 days   | GitHub repository with README and examples   | ⏳ Planned     |
| **Total Duration** |                                                   | **10 weeks**|                                              |               |

**Current Status:** Phase 2 complete, Phase 3 in progress

### 6.1 Milestones

* **Milestone 1 (Week 2):** Clean dataset and precomputed features ready
* **Milestone 2 (Week 5):** Working implementation with unit tests
* **Milestone 3 (Week 8):** All experiments completed
* **Milestone 4 (Week 10):** Paper submitted and code released

### 6.2 Dependencies

* Phase 3 depends on Phase 2 completion
* Phase 4 depends on Phase 3 completion
* Statistical testing requires full results from Phase 3
* Paper writing requires analysis from Phase 4.1-4.4

## 7. Risk Analysis

### 7.1 Technical Risks

| Risk                      | Likelihood | Impact | Mitigation Strategy                                                                                                           |
| ------------------------- | ---------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| API rate limits or outages| Medium     | High   | • Implement exponential backoff<br>• Cache intermediate results<br>• Use multiple API keys<br>• Have backup timeframe       |
| Insufficient API budget   | Medium     | Medium | • Start with subset experiments<br>• Optimize prompt length<br>• Consider alternative models if necessary                   |
| Memory constraints        | Low        | Medium | • Process embeddings in batches<br>• Use CPU fallback if GPU memory insufficient<br>• Clear cache between operations        |
| Reproducibility issues    | Medium     | High   | • Fix all random seeds<br>• Use temperature=0<br>• Document exact library versions<br>• Save intermediate results            |
| Model API changes         | Low        | High   | • Pin specific model version<br>• Archive responses for replication<br>• Document API version and date                       |

### 7.2 Methodological Risks

| Risk                                   | Likelihood | Impact | Mitigation Strategy                                                                                                           |
| -------------------------------------- | ---------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Strategy doesn't improve over random   | Low        | High   | • We have preliminary evidence of improvement<br>• Fall back to analyzing failure modes<br>• Explore alternative features |
| Results don't generalize to other tasks| Medium     | Medium | • Acknowledge LAMBADA-specific findings<br>• Discuss task characteristics<br>• Suggest future cross-task validation      |
| Overfitting to validation set          | Low        | Medium | • Use separate test set for final evaluation<br>• No hyperparameter tuning on test set<br>• Report val/test results        |
| Unfair comparison with GPT-3           | Medium     | Low    | • Acknowledge model differences<br>• Focus on relative improvements<br>• Clearly state model used                        |
| Example selection adds minimal value   | Low        | Medium | • We have evidence of 2.37% improvement<br>• Even modest gains validate research<br>• Analyze when/why selection helps   |

### 7.3 Timeline Risks

| Risk                               | Likelihood | Impact | Mitigation Strategy                                                                                                        |
| ---------------------------------- | ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| Experiments take longer than expected | Medium     | Medium | • Run experiments in parallel<br>• Prioritize most important comparisons<br>• Have buffer time in schedule              |
| API latency causes delays          | Medium     | Low    | • Run experiments overnight<br>• Batch requests efficiently<br>• Start experiments early                                  |
| Writing and analysis take longer   | Medium     | Low    | • Start writing early<br>• Have outline ready<br>• Focus on core contributions first                                    |

### 7.4 Data Risks

| Risk                           | Likelihood | Impact | Mitigation Strategy                                                                                                     |
| ------------------------------ | ---------- | ------ | ----------------------------------------------------------------------------------------------------------------------- |
| Dataset licensing issues       | Very Low   | Medium | • LAMBADA is public domain<br>• Document data sources clearly<br>• Follow HuggingFace terms of service                  |
| Data quality issues            | Low        | Low    | • We've already cleaned validation set<br>• Manual inspection of edge cases<br>• Document filtering decisions            |
| Insufficient validation examples | Very Low   | High   | • 4,217 clean examples is sufficient<br>• Can relax filtering if needed<br>• Multiple examples per POS category available |

### 7.5 Contingency Plans

* **If API costs exceed budget:**
    * Reduce test set size to 2,000 examples
    * Focus on most promising strategies
    * Seek additional funding or API credits
* **If results are negative:**
    * Analyze failure modes (when does selection hurt?)
    * Compare with other selection features
    * Reframe as an exploratory study
* **If timeline slips:**
    * Prioritize core experiments (hybrid vs random)
    * Defer secondary analyses
    * Submit abbreviated results

## 8. Expected Outcomes

### 8.1 Quantitative Outcomes

* **Primary Outcome:**
    * **Hypothesis:** Hybrid selection (α=0.2) achieves >72% accuracy.
    * **Target:** 73-74% accuracy (2-3% improvement over random ~71%).
    * **Stretch goal:** Exceed 73.5%.
* **Secondary Outcomes:**
    * Demonstrate syntactic selection (α=0.0) > semantic selection (α=1.0).
    * Identify optimal α in range [0.1, 0.3].
    * Show statistical significance (p < 0.001) for hybrid vs random.

### 8.2 Qualitative Outcomes

* **Theoretical Contributions:**
    * Understanding the one-shot anomaly on LAMBADA.
    * Establishing the importance of syntactic compatibility for word prediction.
    * Demonstrating the complementary value of semantic and syntactic signals.
* **Methodological Contributions:**
    * A reusable hybrid selection framework.
    * Implementation best practices.
    * A systematic evaluation protocol.
* **Practical Contributions:**
    * A strategy to reduce prompt length while maintaining accuracy.
    * A method for cost reduction and latency improvement.
    * Guidance on when to use syntactic vs. semantic selection.

### 8.3 Deliverables

* **Academic Deliverables:**
    * Research paper (6-8 pages, conference format) for a top NLP venue (ACL, EMNLP).
    * Supplementary materials with extended results.
* **Technical Deliverables:**
    * Open-source GitHub repository with documented code.
    * Replication package with preprocessed data and scripts.
* **Educational Deliverables:**
    * Tutorial/blog post explaining the approach.
    * Presentation materials (slides, poster).

### 8.4 Success Criteria

* **Minimum Success:** Hybrid selection shows a statistically significant improvement over random; code is released and reproducible.
* **Target Success:** Achieve 73%+ accuracy; demonstrate syntactic > semantic; publish a research paper.
* **Exceptional Success:** Exceed 73.5% accuracy; generalize findings to other tasks; gain community adoption.

### 8.5 Long-Term Impact

* **Research Community:** Establish example selection as a key factor in one-shot learning; motivate further investigation into linguistic features for prompting.
* **NLP Practitioners:** Offer practical guidance for optimizing one-shot prompts, reducing deployment costs.
* **Theoretical Understanding:** Contribute to understanding in-context learning mechanisms and the role of linguistic information in LLMs.

---

**Document Status:** Version 1.0 - Initial methodology  
**Last Updated:** 2025-10-20  
**Next Review:** After Phase 3 completion (experimental results)

**Change Log:**
* 2025-10-20: Initial document creation
