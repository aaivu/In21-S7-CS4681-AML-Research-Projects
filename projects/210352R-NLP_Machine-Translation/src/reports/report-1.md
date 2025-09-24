# Denoising Pretraining Strategies for mT5 on OPUS-100

## 1. Introduction

Pretraining large language models such as **mT5** requires a denoising objective that enables the model to learn meaningful representations of language. In this project, two denoising strategies were implemented using the **OPUS-100 bilingual dataset (English–Italian pair)** with limited compute resources:

1. **Naïve Noise Injection** – introducing random word/character deletions and word swaps.
2. **T5-style Span Corruption** – replacing text spans with sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, …) as proposed in the original T5 framework.

This report explains both approaches, their mechanisms, strengths, limitations, and justifies the choice of the best method.

---

## 2. Approach 1: Naïve Noise Injection

### 2.1 Method

Input sentences are corrupted by:

- Deleting words with 10% probability.
- Deleting a random character in a word with 10% probability.
- Swapping adjacent words with 5% probability.

The corrupted sentence is used as input, while the original sentence is used as the target.

### 2.2 Intuition

This noise mimics typos, dropped words, or reordering errors often seen in real-world text. The model is trained to reconstruct the clean text from the noisy input, encouraging robustness against surface-level noise.

### 2.3 Strengths

- Simple to implement.
- Mimics real-world text corruption scenarios (typos, dropped words).
- Helps the model handle noisy user-generated content.

### 2.4 Weaknesses

- The corruption is local and shallow (only small perturbations).
- The task is not as challenging as true span prediction, so the model may learn less about deep contextual dependencies.
- Does not align with the original T5 pretraining objective, limiting transferability of results.

---

## 3. Approach 2: T5-Style Span Corruption

### 3.1 Method

- A fixed fraction (e.g., 15%) of tokens is selected for masking.
- Instead of masking individual tokens, **contiguous spans** of tokens are replaced by **sentinel tokens** (`<extra_id_0>`, `<extra_id_1>`, …).

**Example:**

**Original:**  
The quick brown fox jumps over the lazy dog

**Corrupted (input):**  
The <extra_id_0> fox jumps <extra_id_1> dog

**Target (output):**

<extra_id_0> quick brown <extra_id_1> over the lazy

The model must reconstruct the missing spans autoregressively.

### 3.2 Intuition

This objective forces the model to reason over longer contexts to recover missing chunks of text. It is more challenging than character/word noise and closely follows the original pretraining setup of T5.

### 3.3 Strengths

- Faithful to T5’s original design, ensuring architectural compatibility.
- Encourages the model to learn semantic and syntactic structure across longer contexts.
- Supports generalization to a wide variety of downstream tasks (translation, summarization, QA).

### 3.4 Weaknesses

- More complex implementation.
- Slightly higher computational overhead compared to simple noise injection.
- Does not directly mimic “typo noise” but rather a mask-and-recover task.

---

## 4. Comparative Analysis

| Aspect                            | Naïve Noise Injection          | T5-Style Span Corruption                 |
| --------------------------------- | ------------------------------ | ---------------------------------------- |
| Complexity                        | Simple, lightweight            | More complex, requires sentinel handling |
| Faithfulness to T5                | Not aligned                    | Fully aligned                            |
| Task Difficulty                   | Low (surface-level corruption) | High (semantic reconstruction)           |
| Contextual Learning               | Limited                        | Strong (long-range dependencies)         |
| Real-world robustness             | Captures typos & noise         | Captures semantic understanding          |
| Suitability for Domain Adaptation | Moderate                       | Strong (better transfer to new tasks)    |

---

## 5. Conclusion

Both approaches contribute to denoising pretraining, but they serve different purposes:

- The **Naïve Noise Injection** approach is useful as a lightweight robustness training technique. It helps the model tolerate noisy input, but does not push it to deeply understand context.

- The **T5-Style Span Corruption** approach is superior for domain adaptation and generalization. It aligns with the original T5 objective, enforces semantic reasoning, and ensures that the fine-tuned model retains compatibility with broader downstream tasks.

### ✅ Chosen Approach

For academic rigor and faithful replication of the T5 methodology, the **T5-style span corruption with sentinel tokens** was chosen as the final method.

This ensures that the project not only achieves denoising pretraining on OPUS-100 but also follows best practices in modern sequence-to-sequence pretraining, making it the most effective and generalizable approach.

Fine-tunes mT5-small on your noisy translation dataset.

The model learns to denoise and translate in both directions ---------- Special Point- Novelty (\*\*\*)

Sure! Let’s break it down carefully.

1️⃣ What Happens During EN_noisy → IT / IT_noisy → EN

Your model is trained in a bilingual denoising translation setup:

EN_noisy → IT

Input: An English sentence with noise, e.g., missing words, swapped words, or deleted characters.

Target: The correct Italian translation of the original clean English sentence.

Model Task: Learn to both:

Correct the noisy English input (denoising).

Translate it into Italian.

IT_noisy → EN

Input: An Italian sentence with noise.

Target: The correct English translation of the original clean Italian sentence.

Model Task: Learn to denoise Italian input and translate it into English.

2️⃣ How This Works in Practice

The encoder of mT5 reads the noisy input.

The decoder generates the clean translation in the other language.

By training on both directions, the model becomes bidirectional:

It can translate both English → Italian and Italian → English.

It can handle imperfect input text, like typos or OCR errors.

3️⃣ Importance of This Approach

Robust Translation:

Real-world text is rarely perfect. Users make typos, OCR outputs are noisy, etc.

Training with noisy input ensures the model still produces correct translations.

Bidirectional Learning:

Instead of training two separate models, one model handles both directions.

Saves computation and allows shared multilingual understanding.

Data Efficiency:

Adding noise artificially increases the variety of input without needing extra data.

The model learns to generalize better.

Real-World Applications:

Useful for translation apps, chatbots, or OCR pipelines where input text is often messy.

Improves the user experience by handling imperfect text gracefully.
