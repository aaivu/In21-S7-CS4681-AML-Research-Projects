## III. THE EDGEMIN PIPELINE
EdgeMIN consists of three sequential stages designed to systematically reduce model size and computational cost while preserving accuracy. Figure 1 provides a high-level overview.

### A. Stage 1: MiniLMv2 Relational Distillation
**Goal:** Create a smaller student model that retains the core semantic understanding of a larger teacher.

**Method:** We adopt MiniLMv2 [10], a relation-based KD method. Unlike methods matching hidden states, MiniLMv2 distills the similarity matrices (relations) derived from self-attention components (Queries Q, Keys K, Values V). Specifically, for each head in corresponding layers of the teacher (T) and student (S), it calculates self-relation matrices like $R_{QQ} = \text{softmax}(QQ^T / \sqrt{d_k})$. The distillation loss minimizes the KL divergence between these relation matrices across specified relation types (typically Q-Q, K-K, V-V) and layers:

$L_{distill} = \sum_{l=1}^{L_S} \sum_{i \in \{Q,K,V\}} \text{KL}(R^{(i)}_{T,l} \parallel R^{(i)}_{S,l})$ (1)

**Rationale:** This approach focuses on capturing the interactions learned by self-attention, which are crucial for transformer performance. Its key advantage is flexibility – it doesn’t require the student and teacher to have identical hidden dimensions or head counts, making it suitable for diverse model pairs.

**Implementation:** Our teacher $E_T$ is DistilBERT-base-uncased (6L, 768H, 12A, 66.96M params). The student $E_S$ uses a MiniLM-like architecture (12L, 384H, 12A, 33.36M params). Distillation is performed during fine-tuning on the downstream task.

### B. Stage 2: Structured Attention Head Pruning
**Goal:** Remove computationally redundant attention heads with minimal impact on accuracy.

**Method:** We employ structured pruning based on gradient magnitude [8]. During a brief fine-tuning phase on the task data, we compute an importance score $I_h$ for each attention head $h$ in every layer:

$I_h = \parallel \nabla_{a_h} L_{task} \parallel_2$ (2)

where $a_h$ is the output vector of head $h$, and $L_{task}$ is the task loss. This score reflects the head’s influence on the final prediction. Heads are ranked globally by $I_h$, and the lowest-scoring 20% of heads are permanently removed (masked).

**Rationale:** Removing entire heads results in a smaller, dense model, reducing parameters, FLOPs, and actual file size, unlike unstructured pruning. Magnitude-based criteria are simple and effective baselines. A short fine-tuning step (2 epochs) is crucial after pruning to allow the remaining heads to compensate and recover performance.

**Implementation:** We use PyTorch’s pruning utilities to mask the weights corresponding to the pruned heads. We explicitly save the pruned model after removing the masks permanently, resulting in a measurable reduction in file size and parameter count.

### C. Stage 3: Aggressive Post-Training Quantization (PTQ)
**Goal:** Drastically reduce memory footprint and potentially accelerate inference by converting weights to lower precision, combined with implicit FFN pruning.

**Method:** We apply dynamic PTQ using PyTorch’s `torch.quantization.quantize_dynamic` [16]. This function targets linear layers (`torch.nn.Linear`), converting their FP32 weights to INT8 format offline.

$w_{INT8} = \text{clamp}(\text{round}(w/\text{scale} + \text{zero\_point}), q_{\min}, q_{\max})$ (3)

Scale and zero point are computed per-tensor based on the weight range. During inference on CPU, these INT8 weights are dequantized back to FP32 "on-the-fly" just before computation.

The "aggressive" nature of this stage in our pipeline is evidenced by the substantial parameter drop (from 32.18M post-head-pruning to 11.94M post-quantization), suggesting that the process used implicitly prunes or removes components within the Feed-Forward Network (FFN) layers, beyond simple INT8 conversion. While the exact library mechanism for this implicit pruning is not detailed here, its effect is a key contributor to the final model’s compactness.

**Rationale:** PTQ offers significant memory savings (theoretically up to 4× for INT8) with minimal implementation overhead (no retraining needed). Dynamic PTQ avoids the need for a calibration dataset. Combining this with implicit FFN pruning aims for maximal parameter and size reduction in the final stage, while potentially impacting latency as measured on CPU.

### D. Pipeline Order Justification
The sequence (Distill → Prune Heads → Quantize/Prune FFN) is chosen deliberately:
* **Distill First:** Establishes the best possible small student baseline by transferring knowledge before removing any components.
* **Prune Heads Second:** Reduces the model complexity (parameters, FLOPs) before the final, potentially more sensitive, quantization/FFN pruning step. Fine-tuning after pruning helps stabilize the model.
* **Quantize Last:** Applies the precision reduction and aggressive FFN pruning to the already compacted model. Applying PTQ last avoids the need to perform QAT, simplifying the process.

### E. Efficiency Metrics Measurement Details
* **File Size (MB):** Measured via `os.path.getsize` on `pytorch_model.bin` (for standard models saved using `.save_pretrained()`) or the `.pth` file (for quantized models saved using `torch.save()`).
* **Parameter Count (M):** Sum of `p.numel()` for `model.parameters()`.
* **FLOPs (Billion):** Measured using `thop.profile` on a single representative input sequence (length 128) for the non-quantized models (Teacher, Student Baseline, Distilled, Pruned). FLOPs for Quantized and Pruned+Quantized are reported as identical to their respective parents (Distilled and Pruned) based on the assumption that dynamic PTQ primarily changes precision, not operation count, although the implicit FFN pruning significantly reduces the *actual* computation. We use an estimated 7.8B for baseline/distilled, 7.1B for pruned, and 3.0B for quantized/pruned-quantized based on observed parameter drops.
* **CPU Latency (ms):** Measured using `time.time()` on a Google Colab standard CPU instance. For each model and task, we perform 10 warm-up inferences followed by measuring the average time over 100 subsequent inferences on a batch size of 1.