
#  Dataset

###  Introduction
This folder contains the dataset used for our research on biological reasoning with DNA-based models.  This containes 


This dataset was instroduced in the following paper:

> Fallahpour, A., Magnuson, A., Gupta, P., Ma, S., Naimer, J., Shah, A., Duan, H., Ibrahim, O., Goodarzi, H., Maddison, C. J., & Wang, B. (2025). *BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model.* [arXiv:2505.23579](https://arxiv.org/abs/2505.23579)

---


###  Dataset Description

- **Total samples:** ~1,450  
  - Train: ~1,160  
  - Validation: ~144  
  - Test: ~146

- **Modality:** Text + DNA sequence  
- **License:** Apache-2.0

### ✨ Columns

| Column                | Type   | Description                                                                 |
|------------------------|--------|------------------------------------------------------------------------------|
| `question`             | str    | Biological reasoning question, often includes pathway and variant context.  |
| `answer`               | str    | Expected answer (typically the disease or biological effect).               |
| `reference_sequence`   | str    | DNA reference sequence.                                                    |
| `variant_sequence`     | str    | DNA variant sequence (mutation).                                           |


---

