
# Literature Review: Small LLMs: Edge Computing

**Student:** 210554M  
**Research Area:** Small LLMs: Edge Computing  
**Date:** 2025-10-16  

## Abstract

Recent advances in large language models (LLMs) have achieved remarkable performance across NLP tasks, but their deployment in edge environments remains challenging due to high memory and computational requirements. To address this, research has focused on *small language models* (SLMs) and *quantization techniques* to enable efficient inference on low-power devices. Quantization-aware training (QAT) and post-training quantization (PTQ) have emerged as key methods for reducing model size while maintaining accuracy. Additionally, studies have explored lightweight transformer architectures, domain-specific fine-tuning, and hardware-aware optimization to support edge inference. This literature review synthesizes key developments in model compression, quantization, edge hardware adaptation, and specialized domain adaptation, highlighting research gaps in real-time, privacy-preserving, and interpretable on-device language intelligence.

## 1. Introduction

The rapid evolution of LLMs such as GPT [1], LLaMA [2], and Qwen [3] has demonstrated their strong capability in text understanding, reasoning, and generation. However, these models are computationally heavy—LLaMA 3.1 with 405B parameters can require more than 200 GB of memory even in quantized form [7]. This prevents their use in real-time or offline applications on edge devices, where computing resources, power, and connectivity are constrained. To overcome this limitation, small language models (SLMs) with fewer than one billion parameters have been developed [9]. These models retain key LLM abilities while reducing inference latency and energy consumption, making them suitable for mobile and embedded systems. This literature review examines the main directions in this research space, focusing on small and quantized LLMs for edge deployment.

## 2. Search Methodology

### Search Terms Used
- “small language model”, “lightweight transformer”, “quantized LLM”, “on-device LLM”, “edge AI”, “QAT and PTQ”, “LLM for mobile devices”
- Synonyms and variations: “tiny LLM”, “efficient transformer”, “edge NLP”, “mobile language model”

### Databases Searched
- [x] IEEE Xplore  
- [x] ACM Digital Library  
- [x] Google Scholar  
- [x] ArXiv  

### Time Period
2018–2025, focusing on rapid developments in quantization and mobile LLM deployment since 2020.

## 3. Key Areas of Research

### 3.1 Large Language Models and Edge Deployment

LLMs like GPT [1], BERT [6], and LLaMA [2] have achieved human-level fluency in many tasks, but their enormous parameter sizes make deployment on low-resource devices infeasible. Most LLMs require GPU acceleration and cloud connectivity, which introduces latency, cost, and privacy risks [8]. Consequently, researchers have begun focusing on adapting these architectures for edge computing environments, where memory and compute are severely constrained. This has driven the development of smaller, more efficient variants, supported by model compression and quantization research.

**Key Papers:**
- Devlin et al. (2019) – Introduced BERT, establishing the transformer foundation for LLMs [6].  
- Touvron et al. (2023) – Proposed LLaMA as an open, efficient base model for research and downstream fine-tuning [2].  
- Zheng et al. (2025) – Surveyed edge-optimized LLM frameworks, highlighting quantization and pruning as essential strategies [10].  

### 3.2 Small and Edge-Friendly Language Models

Recent SLMs, such as Qwen2.5-0.5B [3] and TinyLlama (1.1B), demonstrate that high-quality text generation can be achieved with sub-billion parameter models. These models balance efficiency and expressiveness, making them suitable for on-device applications like translation, chatbots, and voice assistants. According to Nguyen et al. (2024) [9], SLMs can maintain strong reasoning and language understanding capabilities despite their smaller size. They also reduce inference cost, preserve privacy by keeping data local, and make offline use feasible.

**Key Papers:**
- Nguyen et al. (2024) – Comprehensive survey of SLMs, emphasizing their growing role in real-world applications [9].  
- Wang et al. (2024) – Survey on small models, showing competitive performance with major LLMs in constrained environments [7].  

### 3.3 Resource Constraints on Edge Devices

Edge devices such as mobile phones, drones, and IoT boards impose strict hardware constraints—limited memory (2–8 GB RAM), low computational power, and restricted thermal capacity. Studies show that running even a 7B model in 4-bit quantized form may require several gigabytes of RAM [10]. Xiao et al. (2025) [8] examined mobile LLM inference and found that edge devices underutilize GPUs, with latency dominated by token generation time and power throttling. Optimizations like instruction-level parallelism (e.g., ARM smmla/sdot) and use of neural processing units (NPUs) can mitigate this. System-level tuning (frequency scaling, batching) is also crucial for practical deployment.

**Key Papers:**
- Xiao et al. (2025) – Experimental study showing performance trade-offs in LLM inference across commercial smartphones [8].  
- Zheng et al. (2025) – Highlighted challenges of edge deployment and recommended quantization and co-design strategies [10].  

### 3.4 Model Compression and Quantization

Compression techniques are crucial for enabling SLMs on-device. These include pruning, knowledge distillation, low-rank factorization, and quantization. Quantization, which converts floating-point weights into lower-bit integers, is particularly effective—it reduces memory and accelerates matrix multiplications [11]. A common compromise is 8-bit quantization, balancing performance and accuracy. Distillation and pruning can complement quantization but are more complex to implement for transformers [10].

**Key Papers:**
- Jacob et al. (2017) – Established quantization methods for neural networks, enabling efficient integer-only inference [11].  
- Zheng et al. (2025) – Compared compression strategies and identified quantization as the most effective for edge deployment [10].  

### 3.5 Post-Training vs. Quantization-Aware Training

Post-Training Quantization (PTQ) applies quantization after model training, offering simplicity but potentially causing accuracy loss. Quantization-Aware Training (QAT) integrates quantization into training, simulating low-precision arithmetic during forward passes and adjusting weights during backpropagation [11]. QAT consistently yields better results than PTQ, especially for transformer architectures. It has become a standard for preparing models for mobile and embedded inference.

**Key Papers:**
- Jacob et al. (2017) – Formalized QAT and demonstrated its superior accuracy retention compared to PTQ [11].  
- Nguyen et al. (2024) – Emphasized QAT as a primary enabler for real-time edge inference [9].  

### 3.6 LLMs in Disaster Management

LLMs are increasingly used for humanitarian and emergency tasks, such as crisis classification, situational summarization, and resource coordination. Xu et al. (2025) [12] reviewed LLM applications across disaster phases and found them effective for extracting actionable information from social media and field reports. Lei et al. (2025) [13] further emphasized the potential of LLMs for real-time reasoning and coordination during disasters.

**Key Papers:**
- Xu et al. (2025) – Reviewed LLMs in disaster management and their interdisciplinary impact [12].  
- Lei et al. (2025) – Surveyed LLM applications for crisis prediction, planning, and response [13].  

### 3.7 Domain-Specific Adaptation

Generic LLMs lack domain-specific understanding. Domain adaptation through fine-tuning or pretraining on specialized datasets addresses this. Lamsal et al. (2024) [14] proposed *CrisisTransformers*, trained on millions of disaster-related tweets, which improved classification and summarization accuracy. Similar approaches apply to medical, legal, and environmental domains where specialized vocabulary and reasoning are required.

**Key Papers:**
- Lamsal et al. (2024) – Developed CrisisTransformers, enhancing crisis-related NLP understanding [14].  
- Xu et al. (2025) – Highlighted importance of domain-specific fine-tuning for emergency contexts [12].  

## 4. Research Gaps and Opportunities

### Gap 1: Lack of Edge-Compatible SLM Benchmarks  
**Why it matters:** Current studies often evaluate models on cloud hardware. Few standardized benchmarks exist for quantized models on real edge devices.  
**How your project addresses it:** Develop a reproducible benchmark pipeline to test quantized SLMs on diverse edge hardware, evaluating trade-offs among latency, accuracy, and power.

### Gap 2: Limited Domain-Specific Quantized Models  
**Why it matters:** Few quantized SLMs are tailored to domains like disaster management or low-resource languages.  
**How your project addresses it:** Incorporate quantization-aware training into domain fine-tuning workflows, enabling lightweight and contextually relevant SLMs for offline environments.

## 5. Theoretical Framework

This research is grounded in transformer theory, quantization mathematics, and edge computing principles. The transformer’s self-attention mechanism enables scalable representation learning but requires optimization for resource constraints. Quantization theory provides the mathematical basis for reducing precision while minimizing information loss, expressed through scale and zero-point parameters [11]. Edge computing frameworks emphasize low-latency, privacy-preserving computation, aligning with sustainable AI principles (SDG 9 and SDG 13).

## 6. Methodology Insights

Methodologies in prior studies combine algorithmic compression with empirical benchmarking. Most works follow three stages: (1) model selection, (2) quantization or fine-tuning, and (3) evaluation on real or simulated edge devices. Metrics include perplexity, accuracy, latency, and memory footprint. State-of-the-art pipelines use frameworks such as *bitsandbytes*, *TensorRT*, and *ONNX Runtime* for low-level optimization. Benchmarking on datasets like WikiText-2 and crisis-specific corpora is standard practice [6, 12, 14].

## 7. Conclusion

The literature consistently shows that quantized and small-scale transformer models provide a feasible path toward efficient edge-based language AI. Quantization-aware training and domain-specific fine-tuning enhance performance without significant computational overhead. Future research should emphasize hybrid precision methods, federated learning for privacy-preserving adaptation, and hardware co-design for better throughput. These developments align with global goals for sustainable, accessible, and interpretable AI.

## References

1. OpenAI. (2025). *GPT-5*. https://chat.openai.com/  
2. Touvron, H. et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971  
3. Hugging Face. (2025). *Qwen/Qwen2.5-0.5B*. https://huggingface.co/Qwen/Qwen2.5-0.5B  
4. Zheng, Y. et al. (2025). *A Review on Edge Large Language Models: Design, Execution, and Applications*. arXiv:2410.11845  
5. Wang, F. et al. (2024). *A Comprehensive Survey of Small Language Models*. arXiv:2411.03350  
6. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805  
7. Xiao, J. et al. (2025). *Understanding Large Language Models in Your Pockets: Performance Study on COTS Mobile Devices*. arXiv:2410.03613  
8. Nguyen, C. V. et al. (2024). *A Survey of Small Language Models*. arXiv:2410.20011  
9. Jacob, B. et al. (2017). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. arXiv:1712.05877  
10. Xu, F. et al. (2025). *Large Language Model Applications in Disaster Management: An Interdisciplinary Review*. International Journal of Disaster Risk Reduction, 127, 105642.  
11. Lei, Z. et al. (2025). *Harnessing Large Language Models for Disaster Management: A Survey*. arXiv:2501.06932  
12. Lamsal, R. et al. (2024). *CrisisTransformers: Pre-trained Language Models for Crisis-Related Texts*. Knowledge-Based Systems, 296, 111916.  

---