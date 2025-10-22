## Results and Discussion

This section presents the quantitative results from the four experimental configurations, evaluated on the COCO validation and test sets.

### Quantitative Results

Performance was measured using BLEU (B1-B4), METEOR (M), ROUGE-L (R-L), CIDEr (C), and SPICE (S).

**TABLE I: Validation Set Captioning Performance**

| Setup | B-4 | M | R-L | C | S |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Human-only (Baseline)** | 0.347 | 0.283 | 0.566 | 1.170 | 0.215 |
| Synthetic-only | 0.316 | 0.294 | 0.559 | 1.072 | 0.229 |
| Hybrid (50%+50%) | 0.354 | 0.293 | 0.576 | 1.193 | 0.225 |
| Multi-caption (2x) | 0.351 | 0.295 | 0.576 | 1.192 | 0.228 |
*(Table data sourced from)*

**TABLE II: Test Set Captioning Performance**

| Setup | B-4 | M | R-L | C | S |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Human-only (Baseline)** | 0.349 | 0.285 | 0.569 | 1.192 | 0.218 |
| Synthetic-only | 0.312 | 0.293 | 0.556 | 1.069 | 0.231 |
| Hybrid (50%+50%) | 0.348 | 0.292 | 0.575 | 1.191 | 0.226 |
| Multi-caption (2x) | 0.347 | 0.294 | 0.574 | 1.192 | 0.229 |
*(Table data sourced from)*

### Analysis and Discussion

* **Baseline:** The human-only configuration achieved strong results on the test set, establishing a CIDEr score of 1.192 and BLEU-4 of 0.349.
* **Synthetic-only:** This model exhibited reduced lexical overlap metrics ($BLEU-4=0.312$, $CIDEr=1.069$) compared to the baseline, highlighting the linguistic limitations of machine-generated captions. However, METEOR and SPICE scores remained comparable, suggesting semantic adequacy was largely preserved.
* **Hybrid (50/50):** This setup achieved the highest overall scores on the validation set ($CIDEr=1.193$, $BLEU-4=0.354$), marginally surpassing the baseline. This indicates that combining synthetic captions with human data introduces useful lexical and stylistic diversity that enhances generalization.
* **Multi-caption (2x):** This configuration yielded performance nearly identical to the hybrid model, maintaining a $CIDEr=1.192$ and slightly improving METEOR and SPICE. The minimal difference suggests diminishing returns once dataset diversity is sufficiently saturated.
* **Conclusion:** While fully synthetic captions alone are insufficient, their strategic integration (either as a hybrid mix or as multi-caption augmentation) can match or slightly improve performance, offering a scalable and cost-efficient complement to human annotations.