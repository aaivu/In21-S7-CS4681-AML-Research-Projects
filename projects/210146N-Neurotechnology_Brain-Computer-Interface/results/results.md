# EEG-ExPy Benchmark Results

This repo contains performance results for classical EEG pipelines evaluated on the three paradigms from the **EEG-ExPy** benchmark, N170, P300, and SSVEP.  
Metrics used: **AUC (Area Under Curve)**.  
All models tested under both **single-subject single-session** and **all-subject all-session** setups.

---

## N170 Paradigm

| Method | Mean | Std | Max |
|:--|:--:|:--:|:--:|
| **Single-subject, single-session** ||||
| ERPCov + MDM | 0.689 | 0.018 | 0.707 |
| ERPCov + TS | 0.704 | 0.030 | 0.734 |
| ERPCov + TS + Ridge | 0.687 | 0.033 | 0.720 |
| Vect + LR | 0.645 | 0.023 | 0.668 |
| Vect + RegLDA | 0.665 | 0.023 | 0.688 |
| XdawnCov + MDM | 0.661 | 0.023 | 0.684 |
| XdawnCov + TS | 0.708 | 0.029 | 0.737 |
| ERPCov + TS finetuned | 0.721 | 0.029 | 0.750 |
| XdawnCov + TS finetuned | 0.723–0.724 | 0.027–0.028 | 0.750–0.752 |
|||||
| **All-subject, all-session** ||||
| ERPCov + MDM | 0.544 | 0.029 | 0.573 |
| ERPCov + TS | 0.587 | 0.018 | 0.605 |
| ERPCov + TS + Ridge | 0.584 | 0.017 | 0.601 |
| Vect + LR | 0.562 | 0.020 | 0.582 |
| Vect + RegLDA | 0.573 | 0.018 | 0.591 |
| XdawnCov + MDM | 0.532 | 0.029 | 0.561 |
| XdawnCov + TS | 0.587 | 0.017 | 0.604 |
| ERPCov + TS finetuned | 0.590 | 0.015 | 0.605 |
| XdawnCov + TS finetuned | 0.583–0.589 | 0.012–0.015 | 0.597–0.602 |

---

## P300 Paradigm

| Method | Mean | Std | Max |
|:--|:--:|:--:|:--:|
| **Single-subject, single-session** ||||
| ERPCov + MDM | 0.777 | 0.041 | 0.818 |
| ERPCov + TS | 0.784 | 0.039 | 0.823 |
| ERPCov + TS + Ridge | 0.781 | 0.041 | 0.822 |
| Vect + LR | 0.668 | 0.039 | 0.707 |
| Vect + RegLDA | 0.757 | 0.046 | 0.803 |
| XdawnCov + MDM | 0.762 | 0.044 | 0.806 |
| XdawnCov + TS | 0.784 | 0.038 | 0.822 |
| ERPCov + TS finetuned | 0.786 | 0.040 | 0.826 |
| XdawnCov + TS finetuned | 0.783–0.784 | 0.039–0.041 | 0.823–0.825 |
|||||
| **All-subject, all-session** ||||
| ERPCov + MDM | 0.555 | 0.029 | 0.584 |
| ERPCov + TS | 0.636 | 0.012 | 0.648 |
| ERPCov + TS + Ridge | 0.636 | 0.012 | 0.648 |
| Vect + LR | 0.579 | 0.016 | 0.595 |
| Vect + RegLDA | 0.593 | 0.017 | 0.610 |
| XdawnCov + MDM | 0.542 | 0.028 | 0.570 |
| XdawnCov + TS | 0.636 | 0.012 | 0.648 |
| ERPCov + TS finetuned | 0.634 | 0.012 | 0.646 |
| XdawnCov + TS finetuned | 0.628–0.633 | 0.012 | 0.640–0.645 |

---

## SSVEP Paradigm

| Method | Mean | Std | Max |
|:--|:--:|:--:|:--:|
| **Single-subject, single-session** ||||
| CSP + Cov + TS | 0.949 | 0.030 | 0.979 |
| CSP + RegLDA | 0.941 | 0.030 | 0.971 |
| Cov + MDM | 0.896 | 0.048 | 0.944 |
| Cov + TS | 0.952 | 0.026 | 0.978 |
| CSP + Cov + TS finetuned | 0.948 | 0.029 | 0.977 |
| CSP + RegLDA finetuned | 0.940 | 0.034 | 0.974 |
| Cov + TS finetuned | 0.951 | 0.025 | 0.976 |
| Cov + TS + SVM | 0.910 | 0.030 | 0.940 |
| Xdawn + TS + LR | 0.935 | 0.033 | 0.968 |
|||||
| **All-subject, all-session** ||||
| CSP + Cov + TS | 0.864 | 0.045 | 0.909 |
| CSP + RegLDA | 0.850 | 0.042 | 0.892 |
| Cov + MDM | 0.758 | 0.057 | 0.815 |
| Cov + TS | 0.869 | 0.030 | 0.899 |
| CSP + Cov + TS finetuned | 0.861 | 0.047 | 0.908 |
| CSP + RegLDA finetuned | 0.858 | 0.040 | 0.898 |
| Cov + TS finetuned | 0.867 | 0.034 | 0.901 |
| Cov + TS + SVM | 0.814 | 0.040 | 0.854 |
| Xdawn + TS + LR | 0.815 | 0.048 | 0.863 |

---

## Notes
- Metrics: **AUC**, computed per trial averaged across folds.  
- “finetuned” refers to minor parameter optimizations beyond the default EEG-ExPy baseline.
- SSVEP shows the strongest generalization; N170 and P300 drop significantly in cross-subject setups.

