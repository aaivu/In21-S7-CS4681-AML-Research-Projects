🧾 Progress Report – OurTD3 Research Project

Title: Agreement-Weighted Replay and Value-Improvement Regularization for Continuous Control
Author: Sajith Anuradha – Department of Computer Science & Engineering, University of Moratuwa
Period: Weeks 4 – 12

🗓 Week 4 – Initial Implementation & Baseline Setup

Main Tasks
• Implemented the standard TD3 baseline with twin critics, target-policy smoothing, and delayed actor updates.
• Verified MuJoCo v5 environment installation (Hopper, Walker2d, HalfCheetah) through Gymnasium.
• Created core training loop, evaluation routine, and replay-buffer logic.
• Established logging pipeline using TensorBoard/W&B.
Outcome
• Achieved reproducible TD3 runs (~3.3k avg return on Hopper).
• Confirmed environment determinism and evaluation metrics.

🗓 Week 5 – Agreement-Weighted Replay (AWR)

Main Tasks
• Designed and implemented agreement-weighted replay mechanism.
• Defined cosine-agreement weight
\alpha = \frac{\delta_1\delta_2}{|\delta_1||\delta_2|+\varepsilon} and integrated exponential weighting with normalization.
• Added hyper-parameters: κ (agreement strength), weight clipping (0.5 – 2.0).
• Ran comparative experiments (TD3 vs OurTD3-AWR).
Outcome
• Observed faster early learning on Hopper (+15 % sample-efficiency).
• Stable gradient behavior confirmed.

🗓 Week 6 – Value-Improvement (VI) Regularizer

Main Tasks
• Added critic value-improvement term using greedy backup target y\_{\max}.
• Tuned λ ∈ {0.005, 0.01, 0.05}.
• Conducted controlled ablations to analyze bias/variance trade-off.
Outcome
• λ ≈ 0.01 gave consistent asymptotic gain (~+300 return on HalfCheetah).
• Large λ caused mild optimism—documented limits.

🗓 Week 7 – Stability Enhancements & Ablations

Main Tasks
• Added gradient-norm clipping (‖∇‖≤10).
• Performed multi-seed (10×) runs to measure variance.
• Combined AWR + VI → final OurTD3 architecture.
• Measured runtime overhead and training variance.
Outcome
• Variance reduced ≈ 25 %; runtime overhead < 3 %.
• All critics stable across seeds.

🗓 Week 8 – Comprehensive Evaluation & Visualization

Main Tasks
• Generated complete learning-curve plots (return vs timesteps).
• Added performance tables and summary metrics.
• Prepared dataset/environment description for the paper.
• Validated results on Hopper-v5, Walker2d-v5, HalfCheetah-v5.
Outcome
• OurTD3 outperformed TD3 and TD3-BC on all benchmarks.
• Completed experiment summary and experiments.md.

🗓 Week 9 – Short Paper Preparation & Optimization

Main Tasks
• Drafted short-paper version (4 pages) for internal submission.
• Optimized code for faster rollouts and reproducibility.
• Cleaned results directory, generated average/variance plots.
Outcome
• Short paper submitted successfully.
• Code and figures reproducible on single GPU.

🗓 Week 10 – Algorithm Refinement & Hyper-tuning

Main Tasks
• Fine-tuned κ annealing schedule (0→5) and λ decay for smoother training.
• Performed sensitivity analysis for weight-normalization strategy.
• Polished figure captions and algorithm pseudocode for paper.
Outcome
• Improved stability on Walker2d (no late collapse).
• Ready for full paper integration.

🗓 Week 11 – Full Paper Writing & Final Results

Main Tasks
• Expanded full research paper to 7 pages.
• Added extended sections: Dataset and Environments, Ablation Study, Discussion, Conclusion.
• Inserted final quantitative tables and updated bibliography.
Outcome
• Camera-ready draft produced with finalized plots and metrics.
• All experiments cross-verified and consistent.

🗓 Week 12 – Finalization & Submission

Main Tasks
• Performed proofreading and supervisor revisions.
• Verified references, figure labels, and LaTeX build.
• Uploaded all artifacts: code, plots, experiments.md, and final PDF.
Outcome
• Research paper finalized and submitted.
• Repository structured and archived for reproducibility.
