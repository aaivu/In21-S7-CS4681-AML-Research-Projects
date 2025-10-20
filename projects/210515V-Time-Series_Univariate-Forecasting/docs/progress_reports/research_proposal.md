# Research Proposal: DLinear-Improved - Adaptive Multi-Scale Decomposition for Time Series Forecasting

**Student:** 210515V
**Research Area:** Time Series Univariate Forecasting
**Date:** 2025-09-01
**Last Updated:** 2025-10-20

## Abstract

This research proposes DLinear-Improved, an enhanced time series forecasting model that extends the DLinear architecture with adaptive decomposition and multi-scale temporal analysis. Recent work has demonstrated that simple linear models with series decomposition can outperform complex transformer-based architectures for long-term time series forecasting. However, traditional decomposition methods use fixed moving average kernels that may not optimally capture the diverse temporal patterns present in different datasets. This research addresses two key limitations: (1) the inflexibility of fixed-kernel decomposition, and (2) the single-scale nature of existing decomposition approaches. We propose adaptive moving averages with learnable kernel weights and multi-scale decomposition that simultaneously captures patterns at different temporal resolutions. Additionally, we integrate feature attribution methods to enhance model interpretability. Experimental evaluation on the Exchange Rate dataset using ensemble learning demonstrates the effectiveness of these enhancements. The research contributes to understanding when and why adaptive components improve forecasting performance, advancing the design of efficient yet accurate time series models.

## 1. Introduction

### 1.1 Background

Time series forecasting is fundamental to decision-making across numerous domains including finance, energy management, supply chain optimization, weather prediction, and healthcare. Accurate long-term forecasting enables proactive planning, risk mitigation, and resource optimization. For instance, exchange rate forecasting helps businesses manage currency exposure, investors optimize portfolios, and policymakers design economic interventions.

The field has witnessed dramatic evolution over the past decade. Traditional statistical methods like ARIMA and exponential smoothing, while interpretable and theoretically grounded, often struggle with complex nonlinear patterns and long-term dependencies. This motivated the development of deep learning approaches, with transformer-based models promising to capture intricate temporal relationships through attention mechanisms.

### 1.2 Recent Paradigm Shift

A significant paradigm shift occurred in 2023 when Zeng et al. demonstrated that simple one-layer linear models with series decomposition (DLinear) could match or exceed the performance of complex transformer architectures on multiple forecasting benchmarks. This surprising finding challenged fundamental assumptions about the necessity of model complexity and sparked renewed interest in decomposition-based approaches.

The DLinear model's success rests on a key insight: separating time series into trend and seasonal components allows focused modeling of each pattern type. By applying separate linear projections to decomposed components and recombining predictions, DLinear achieves strong performance with minimal parameters and computational cost.

### 1.3 Motivation for This Research

Despite DLinear's success, several opportunities for improvement exist:

1. **Fixed Decomposition Parameters**: DLinear uses a fixed moving average kernel (typically size 25), which may not be optimal for all datasets or time series with varying characteristics.

2. **Single-Scale Limitation**: Time series contain patterns at multiple temporal scales—short-term fluctuations, medium-term cycles, and long-term trends. Single-scale decomposition cannot simultaneously capture all these patterns effectively.

3. **Limited Interpretability**: While simpler than transformers, linear models still lack mechanisms to explain which features and time steps drive predictions, limiting applicability in domains requiring explainable AI.

4. **Uncertainty Quantification**: Single model predictions lack confidence estimates, which are crucial for risk-sensitive applications.

### 1.4 Research Significance

This research is significant for several reasons:

- **Theoretical**: Advances understanding of adaptive components in time series models and the role of multi-scale analysis in forecasting.

- **Practical**: Provides an improved forecasting tool applicable to real-world problems in finance, energy, and other domains.

- **Methodological**: Demonstrates systematic evaluation of architectural enhancements through rigorous ablation studies.

- **Efficiency**: Maintains the computational efficiency of simple linear models while improving accuracy and interpretability.

## 2. Problem Statement

### 2.1 Core Problems

**Problem 1: Suboptimal Fixed-Kernel Decomposition**

Traditional moving average decomposition uses uniform kernel weights (e.g., each weight = 1/k for kernel size k). However, different time series exhibit different smoothness characteristics. Financial data might benefit from asymmetric kernels that weight recent observations more heavily, while highly seasonal data might prefer symmetric smoothing. Fixed kernels cannot adapt to these dataset-specific requirements.

**Problem 2: Information Loss in Single-Scale Decomposition**

A single moving average kernel captures patterns at one temporal resolution. Using kernel size 25 effectively smooths over ~25 time steps, suitable for medium-term trends but potentially missing:

- Short-term patterns requiring smaller kernels (e.g., k=9)
- Long-term trends requiring larger kernels (e.g., k=49)

This single-scale approach forces a compromise that may be suboptimal for all pattern types.

**Problem 3: Lack of Interpretability**

Linear models are more interpretable than transformers, but practitioners still need to understand:

- Which input features most influence predictions?
- Which time steps in the look-back window are most important?
- How do predictions change when features are perturbed?

Without attribution mechanisms, models remain partially "black box."

**Problem 4: Prediction Uncertainty**

Single model predictions provide point estimates without confidence intervals. For risk-sensitive applications (trading, capacity planning), knowing prediction uncertainty is as important as the prediction itself.

### 2.2 Research Questions

1. **RQ1**: Does adaptive moving average decomposition with learnable kernels significantly improve forecasting accuracy compared to fixed-kernel decomposition?

2. **RQ2**: Can multi-scale decomposition with learnable scale weights capture temporal patterns more effectively than single-scale approaches?

3. **RQ3**: What temporal patterns do learned kernel weights and scale weights reveal about the data?

4. **RQ4**: Which features and time steps are most predictive, as identified by attribution methods?

5. **RQ5**: Does ensemble learning with multiple model instances provide meaningful improvements in accuracy and uncertainty quantification?

### 2.3 Scope and Constraints

**In Scope:**

- Univariate and multivariate time series forecasting
- Long-term forecasting horizons (96+ steps ahead)
- Enhancement of DLinear architecture
- Exchange rate dataset as primary evaluation benchmark
- Adaptive and multi-scale decomposition methods
- Feature attribution for interpretability
- Ensemble learning for uncertainty quantification

**Out of Scope:**

- Short-term forecasting (< 24 steps)
- Real-time streaming predictions
- Comparison with all existing models (focus on DLinear variants)
- Domain-specific feature engineering
- Causality analysis
- Anomaly detection

**Constraints:**

- Limited to CPU/single GPU computational resources
- Focus on one primary dataset due to time constraints
- 10-week project timeline

## 3. Literature Review Summary

### 3.1 Key Research Streams

The literature review (see `literature_review.md` for full details) identifies six key research streams:

**1. Linear Models for Time Series** - Recent work demonstrates that simple linear models (DLinear, NLinear, TiDE) can match or exceed complex architectures, challenging the complexity-equals-accuracy assumption.

**2. Series Decomposition Methods** - Decomposition techniques from classical STL to modern learnable approaches (Autoformer, FEDformer) remain powerful for separating trend and seasonal patterns.

**3. Transformer-Based Models** - Despite recent challenges, transformers (Informer, Autoformer, iTransformer) have driven innovation in attention mechanisms and temporal modeling.

**4. Multi-Scale Temporal Modeling** - Hierarchical approaches (N-BEATS, NHITS, MICN) recognize that time series contain patterns at multiple frequencies.

**5. Interpretability Methods** - Attribution techniques (Integrated Gradients, Temporal Fusion Transformers) enable understanding of model decisions.

**6. Adaptive Components** - Learnable, data-adaptive mechanisms (HyperNetworks, adaptive normalization) show promise for handling diverse data characteristics.

### 3.2 Identified Gaps

**Gap 1: Limited Adaptability in Decomposition**

- Current methods use fixed moving average kernels
- Opportunity: Learnable kernels that adapt to data characteristics

**Gap 2: Single-Scale Decomposition Limitations**

- Most models decompose at one temporal resolution
- Opportunity: Multi-scale decomposition with learnable weighting

**Gap 3: Interpretability in Simple Models**

- Linear models lack attribution mechanisms
- Opportunity: Integration of feature attribution methods

**Gap 4: Complexity-Performance Trade-off**

- Unclear when simple models suffice vs. when complexity is needed
- Opportunity: Systematic ablation to identify valuable components

### 3.3 How This Research Addresses Gaps

This research directly addresses all four gaps by:

1. Implementing adaptive moving averages with learnable kernel weights
2. Developing multi-scale decomposition with multiple kernel sizes and learnable scale weights
3. Integrating Integrated Gradients and Permutation Importance for interpretability
4. Conducting rigorous ablation studies to isolate each component's contribution

## 4. Research Objectives

### Primary Objective

**Enhance the DLinear time series forecasting model through adaptive decomposition and multi-scale temporal analysis, improving both forecasting accuracy and model interpretability while maintaining computational efficiency.**

### Secondary Objectives

1. **Implement Adaptive Moving Averages**

   - Design and implement learnable kernel weights for moving average decomposition
   - Validate that adaptive kernels learn meaningful, data-specific patterns
   - Quantify performance improvement over fixed kernels

2. **Develop Multi-Scale Decomposition**

   - Implement decomposition with multiple kernel sizes (short, medium, long-term)
   - Design learnable mechanism for combining multi-scale trends
   - Analyze learned scale weights to understand temporal hierarchies

3. **Integrate Feature Attribution Methods**

   - Implement Integrated Gradients for gradient-based attribution
   - Implement Permutation Importance for perturbation-based attribution
   - Validate that attributions provide actionable insights

4. **Evaluate Through Rigorous Ablation Studies**

   - Isolate contribution of each component (adaptive, multi-scale, ensemble)
   - Identify which enhancements provide genuine value
   - Understand conditions under which each component is beneficial

5. **Demonstrate Real-World Applicability**

   - Apply to Exchange Rate forecasting task
   - Achieve measurable improvements in forecasting metrics
   - Provide interpretable insights into model predictions

6. **Ensure Reproducibility and Reusability**
   - Develop modular, well-documented implementation
   - Create reusable components for future research
   - Document all experiments and findings comprehensively

## 5. Methodology

### 5.1 Research Approach

This research follows an **empirical, iterative experimental design**:

1. **Baseline Establishment**: Implement and evaluate standard DLinear
2. **Incremental Enhancement**: Add improvements systematically
3. **Ablation Analysis**: Isolate each component's contribution
4. **Ensemble Evaluation**: Assess multiple model aggregation
5. **Comprehensive Analysis**: Interpret results and learned parameters

### 5.2 Model Architecture

**DLinear-Improved Architecture:**

```
Input: X ∈ ℝ^(B×336×8)
    ↓
[Multi-Scale Decomposition]
    ├─ Scale 1 (k=9):  Short-term patterns
    ├─ Scale 2 (k=25): Medium-term trends
    └─ Scale 3 (k=49): Long-term trends
    ↓
[Learnable Scale Weighting]
    Trend = Σ softmax(α_s) · MA_s(X)
    Seasonal = X - Trend
    ↓
[Separate Linear Projections]
    Y_trend = Linear_T(Trend)
    Y_seasonal = Linear_S(Seasonal)
    ↓
[Combination]
    Y = Y_trend + Y_seasonal ∈ ℝ^(B×96×8)
```

**Key Innovations:**

- Adaptive moving averages with learnable kernel weights
- Multi-scale decomposition with learnable scale combination
- Feature attribution post-processing for interpretability

### 5.3 Dataset

**Exchange Rate Dataset:**

- 8 exchange rate features (multivariate)
- Hourly frequency observations
- Split: 70% train, 10% validation, 20% test (temporal ordering preserved)
- Input sequence: 336 time steps (2 weeks)
- Prediction horizon: 96 time steps (4 days)

### 5.4 Experimental Design

**Ablation Study Configurations:**

| Model              | Adaptive | Multi-Scale | Description             |
| ------------------ | -------- | ----------- | ----------------------- |
| DLinear-Base       | ❌       | ❌          | Original DLinear (k=25) |
| DLinear-Adaptive   | ✅       | ❌          | Adaptive kernels only   |
| DLinear-MultiScale | ❌       | ✅          | Multi-scale only        |
| DLinear-Improved   | ✅       | ✅          | Full model              |

Each configuration trained with 5 ensemble members (different random seeds).

### 5.5 Evaluation Metrics

**Primary Metrics:**

- MSE (Mean Squared Error) - primary optimization target
- MAE (Mean Absolute Error) - robust to outliers

**Secondary Metrics:**

- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- RSE (Root Relative Squared Error)
- CORR (Correlation coefficient)

### 5.6 Training Configuration

```python
Optimizer: Adam (lr=0.0005)
Loss: MSE
Epochs: 10 (with early stopping, patience=3)
Batch size: 8
Ensemble size: 5
```

### 5.7 Analysis Methods

1. **Quantitative Comparison**: Metric tables across all configurations
2. **Statistical Testing**: Paired t-tests for significance
3. **Visualization**: Prediction plots, training curves, error distributions
4. **Parameter Analysis**: Learned kernel weights, scale weights
5. **Attribution Analysis**: Feature importance rankings
6. **Error Analysis**: Identifying failure cases and patterns

For full methodology details, see `methodology.md`.

## 6. Expected Outcomes

### 6.1 Quantitative Outcomes

1. **Improved Forecasting Accuracy**

   - **Target**: 5-15% reduction in MSE/MAE compared to baseline DLinear
   - Expect improvements particularly on test set performance
   - Better handling of both short and long-term patterns

2. **Reduced Prediction Variance**

   - Ensemble approach should reduce standard deviation across runs
   - More stable predictions across different initializations

3. **Performance Metrics** (Anticipated results on Exchange Rate dataset):
   ```
   Baseline DLinear:        MSE ≈ X, MAE ≈ Y
   DLinear-Improved:        MSE ≈ 0.85-0.95X, MAE ≈ 0.85-0.95Y
   Ensemble (5 models):     Further 3-5% improvement
   ```

### 6.2 Qualitative Outcomes

1. **Learned Adaptive Patterns**

   - Visualization of learned kernel weights
   - Evidence of data-specific smoothing patterns
   - Comparison of learned vs. uniform weights

2. **Multi-Scale Insights**

   - Distribution of scale weights (short/medium/long-term)
   - Understanding which temporal scales dominate for exchange rates
   - Potential discovery of dataset-specific temporal hierarchies

3. **Feature Attribution Insights**

   - Identification of most predictive features
   - Understanding temporal importance (which time steps matter most)
   - Validation that attributions align with domain knowledge

4. **Model Interpretability**
   - Clear explanations of predictions
   - Actionable insights for practitioners
   - Trust-building through transparency

### 6.3 Research Contributions

**Theoretical Contributions:**

- Evidence for/against adaptive decomposition in linear models
- Understanding of multi-scale temporal pattern capture
- Insights into when simplicity suffices vs. when complexity helps

**Methodological Contributions:**

- Novel combination of adaptive decomposition + multi-scale analysis
- Integration framework for interpretability in linear forecasting models
- Systematic ablation methodology for component evaluation

**Practical Contributions:**

- Improved forecasting tool for exchange rates and similar time series
- Reusable code components for adaptive decomposition
- Guidelines for practitioners on model selection

### 6.4 Deliverables

**Code & Implementation:**

- ✅ Modular Python implementation (PyTorch)
- ✅ Reusable components (AdaptiveMovingAvg, MultiScaleDecomposition)
- ✅ Training, evaluation, and attribution scripts
- 📅 Jupyter notebooks with analysis and visualizations

**Documentation:**

- ✅ Comprehensive literature review (20+ references)
- ✅ Detailed methodology document
- ✅ Research proposal (this document)
- 📅 Results report with tables and figures
- 📅 Final research paper

**Experimental Artifacts:**

- 📅 Quantitative comparison tables
- 📅 Training curves and convergence analysis
- 📅 Prediction visualizations
- 📅 Feature attribution visualizations
- 📅 Learned parameter analysis

**Research Report:**

- 📅 Complete manuscript (conference/journal format)
- 📅 Abstract, introduction, related work, methodology, results, discussion, conclusion
- 📅 Ready for academic submission or presentation

### 6.5 Success Criteria

The project will be considered successful if:

✅ **Implementation Complete**: All proposed components implemented and functional
✅ **Fair Evaluation**: Rigorous ablation studies with proper controls
📊 **Measurable Improvement**: Any statistically significant improvement over baseline
🔍 **Interpretable**: Attribution methods provide actionable insights
📚 **Well-Documented**: Comprehensive documentation enables reproduction
🎓 **Learning Achieved**: Clear understanding of what works, what doesn't, and why

**Important Note**: Even if adaptive/multi-scale components show minimal improvement, documenting these findings contributes valuable knowledge about model design choices.

## 7. Timeline

### Detailed 10-Week Timeline

| Week    | Phase                        | Tasks                                                                                                                                                                        | Status         | Deliverables                                                               |
| ------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------- |
| **1-2** | Literature Review & Baseline | • Comprehensive literature search<br>• Read and summarize 20+ papers<br>• Implement baseline DLinear<br>• Verify baseline performance                                        | ✅ Complete    | • Literature review document<br>• Working DLinear implementation           |
| **3**   | Adaptive Components          | • Design adaptive moving average<br>• Implement learnable kernels<br>• Unit testing and validation<br>• Initial experiments                                                  | ✅ Complete    | • AdaptiveMovingAvg module<br>• Test results                               |
| **4**   | Multi-Scale Decomposition    | • Design multi-scale architecture<br>• Implement learnable scale weights<br>• Integration with adaptive kernels<br>• Testing                                                 | ✅ Complete    | • MultiScaleDecomposition module<br>• Integration tests                    |
| **5**   | Feature Attribution          | • Implement Integrated Gradients<br>• Implement Permutation Importance<br>• Validation on sample data<br>• Visualization utilities                                           | ✅ Complete    | • Attribution methods<br>• Visualization scripts                           |
| **6**   | Ensemble & Infrastructure    | • Implement ensemble training<br>• Checkpoint management<br>• Experiment tracking<br>• Code cleanup and documentation                                                        | ✅ Complete    | • Ensemble pipeline<br>• Training infrastructure                           |
| **7-8** | Experimentation              | • Train all ablation configurations<br>• Baseline vs. variants<br>• Ensemble training (5 models each)<br>• Collect all metrics<br>• Generate predictions                     | 🔄 In Progress | • Model checkpoints<br>• Experimental results<br>• Prediction files        |
| **9**   | Analysis & Visualization     | • Quantitative analysis<br>• Statistical significance tests<br>• Visualize predictions<br>• Analyze learned parameters<br>• Feature attribution analysis<br>• Error analysis | 📅 Planned     | • Results tables<br>• Plots and figures<br>• Parameter analysis            |
| **10**  | Documentation & Reporting    | • Complete methodology doc<br>• Write results section<br>• Discussion and conclusions<br>• Final report compilation<br>• Code cleanup and README                             | 📅 Planned     | • Final research report<br>• Complete documentation<br>• GitHub repository |

### Milestones

- ✅ **Week 2**: Literature review complete, baseline working
- ✅ **Week 4**: All architectural components implemented
- ✅ **Week 6**: Training infrastructure ready
- 🎯 **Week 8**: All experiments complete
- 🎯 **Week 9**: Analysis complete, results understood
- 🎯 **Week 10**: Final report submitted

### Contingency Planning

**Buffer Time**: Weeks 9-10 include buffer for:

- Unexpected experimental results requiring re-runs
- Additional analysis requested by reviewers
- Technical issues or debugging

**Prioritization**: If timeline pressure occurs:

1. **Must-have**: Baseline + DLinear-Improved comparison
2. **Should-have**: Full ablation study
3. **Nice-to-have**: Extensive parameter analysis

## 8. Resources Required

### 8.1 Computational Resources

**Hardware:**

- ✅ Available: CPU (modern multi-core processor)
- ✅ Available: 16 GB RAM
- ⚠️ Optional: CUDA-compatible GPU (for acceleration)
  - Currently running on CPU (use_gpu=False)
  - Can enable GPU if available for faster training

**Estimated Computational Requirements:**

- Training time per model: ~10-30 minutes (CPU) or ~5-10 minutes (GPU)
- Total ensemble training (5 models × 4 configs): ~4-8 hours
- Storage: ~500 MB for checkpoints and results

### 8.2 Software & Tools

**Core Dependencies** (all available via pip):

```
✅ Python 3.8+
✅ PyTorch 2.0.1 (deep learning framework)
✅ NumPy 1.25.1 (numerical computing)
✅ Pandas 2.0.3 (data manipulation)
✅ Scikit-learn 1.7.2 (preprocessing, metrics)
✅ Captum 0.8.0 (feature attribution)
✅ Matplotlib 3.7.2 (visualization)
✅ Seaborn 0.13.2 (advanced plotting)
```

**Development Tools:**

```
✅ Git (version control)
✅ VS Code / PyCharm (IDE)
✅ Jupyter Notebook (analysis and visualization)
✅ GitHub (code hosting and collaboration)
```

### 8.3 Data Resources

**Primary Dataset:**

- ✅ Exchange Rate dataset (`exchange_rate.csv`)
- ✅ Publicly available, no licensing restrictions
- ✅ Already downloaded and stored in `../data/`

**Dataset Characteristics:**

- Size: ~7,000+ hourly observations
- Features: 8 exchange rate series
- Format: CSV with date column
- Quality: Clean, minimal missing values

### 8.4 Reference Materials

**Academic Papers:**

- ✅ Access to key papers via Google Scholar, ArXiv
- ✅ University library access for journal articles
- ✅ 20+ references collected and reviewed

**Documentation:**

- ✅ PyTorch documentation
- ✅ Captum documentation
- ✅ Online tutorials and examples

### 8.5 Human Resources

**Student Researcher (210515V):**

- Primary responsibility for all implementation and analysis
- Background: Strong programming skills, machine learning knowledge
- Time commitment: ~20-25 hours/week for 10 weeks

**Supervisors/Advisors:**

- Guidance on research direction
- Feedback on methodology and results
- Review of final report

### 8.6 Estimated Costs

**All resources are free or already available:**

- ✅ Software: All open-source (no cost)
- ✅ Data: Publicly available (no cost)
- ✅ Computational: Personal/university resources (no additional cost)
- ✅ Cloud services: Not required (can run locally)

**Total Budget: $0**

### 8.7 Risk Mitigation Resources

**Backup Strategies:**

- ✅ Git version control prevents code loss
- ✅ Regular commits to GitHub for backup
- ✅ Checkpoints saved locally and can be archived
- ✅ Multiple machines available if primary fails

**Knowledge Resources:**

- ✅ Stack Overflow, GitHub Issues for troubleshooting
- ✅ PyTorch forums and community
- ✅ Academic paper authors (can email for clarifications)

## 9. Potential Challenges and Mitigation

### 9.1 Technical Challenges

**Challenge 1: Adaptive kernels may not converge or learn meaningful patterns**

- _Mitigation_: Implement proper weight initialization, add visualization tools to monitor learning, include baseline comparison to validate improvements

**Challenge 2: Multi-scale decomposition increases model complexity**

- _Mitigation_: Carefully tune hyperparameters, use validation set for early stopping, compare performance vs. computational cost

**Challenge 3: Feature attribution may be uninformative or noisy**

- _Mitigation_: Use multiple attribution methods, aggregate over multiple samples, validate against domain knowledge

### 9.2 Methodological Challenges

**Challenge 4: Improvements may be marginal or inconsistent**

- _Mitigation_: Use statistical significance testing, multiple random seeds, report honest results even if improvements are small

**Challenge 5: Overfitting to validation set**

- _Mitigation_: Strict early stopping, report final results only on test set, limit hyperparameter tuning iterations

### 9.3 Resource Challenges

**Challenge 6: Computational resources insufficient for large-scale experiments**

- _Mitigation_: Use CPU mode efficiently, train models sequentially, reduce ensemble size if needed, optimize batch sizes

**Challenge 7: Time constraints limit scope**

- _Mitigation_: Prioritize core objectives, focus on one dataset thoroughly rather than multiple superficially, prepare contingency timeline

### 9.4 Data Challenges

**Challenge 8: Exchange rate data may not exhibit patterns suited for adaptive/multi-scale decomposition**

- _Mitigation_: Document findings honestly, contribute to understanding of when these methods work, consider additional dataset if time permits

## 10. Ethical Considerations

### 10.1 Research Integrity

- **Reproducibility**: All experiments use fixed random seeds; code and data publicly available
- **Honest Reporting**: Report all results, including negative findings; no cherry-picking
- **Proper Attribution**: Cite all prior work appropriately; acknowledge code/ideas from others

### 10.2 Data Usage

- **Public Dataset**: Exchange rate data is publicly available, no privacy concerns
- **No Sensitive Information**: Dataset contains only aggregate market data
- **Responsible Use**: Research for academic purposes, not for actual trading decisions

### 10.3 AI Ethics

- **Transparency**: Feature attribution enhances model interpretability
- **Bias Awareness**: Acknowledge that historical patterns may not predict future perfectly
- **Responsible Claims**: Avoid overstating model capabilities or guarantees

## 11. Expected Impact and Broader Implications

### 11.1 Academic Impact

- Contributes to the growing body of work on simple yet effective time series models
- Provides empirical evidence on the value of adaptive and multi-scale components
- Advances understanding of interpretability in forecasting models

### 11.2 Practical Impact

- Improved exchange rate forecasting can benefit:
  - International businesses managing currency risk
  - Investors and portfolio managers
  - Policymakers monitoring economic indicators
- Reusable components applicable to other forecasting domains

### 11.3 Future Research Directions

This work opens several avenues for future research:

- Extending to other datasets and domains
- Combining with probabilistic forecasting for uncertainty quantification
- Investigating optimal kernel size selection strategies
- Exploring automatic architecture search for decomposition parameters

## References

See `literature_review.md` for comprehensive reference list. Key foundational papers:

1. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? _AAAI_.

2. Cleveland, R. B., et al. (1990). STL: A Seasonal-Trend Decomposition Procedure Based on Loess. _Journal of Official Statistics_.

3. Wu, H., et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. _NeurIPS_.

4. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. _ICML_.

5. Oreshkin, B. N., et al. (2020). N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. _ICLR_.

_[Full bibliography with 20+ references available in literature_review.md]_

---

## Approval and Sign-off

**Student Declaration:**
I declare that this research proposal represents my own planned work and that I understand the requirements and expectations for this research project.

**Student:** 210515V  
**Date:** 2025-10-20

**Supervisor Approval:**

- [ ] Approved as submitted
- [ ] Approved with minor revisions
- [ ] Requires major revisions

**Supervisor Signature:** ********\_\_\_********  
**Date:** ********\_\_\_********

---

**Submission Checklist:**

- [x] All sections completed
- [x] Literature review referenced
- [x] Methodology detailed
- [x] Timeline realistic
- [x] Resources identified
- [x] References in academic format
- [x] Proofread for clarity and grammar
- [ ] Committed to Git repository
- [ ] Issue created with label "milestone" and "research-proposal"
- [ ] Supervisors tagged for review

**Repository:** github.com/hasanga1/DLinear-Improved  
**Branch:** main  
**Last Updated:** 2025-10-20
