# Few-Shot Learning Evaluation Commands for TNC Encoder
# ======================================================

# Quick test with default settings (recommended first run)
.venv/bin/python -m evaluations.few_shot_test --data simulation --shots 1,3,5 --trials 10

# Comprehensive evaluation (all methods, more shots, more trials)  
.venv/bin/python -m evaluations.few_shot_test --data simulation --shots 1,3,5,10,20 --trials 50

# Test only Prototypical Networks (main method)
.venv/bin/python -m evaluations.few_shot_test --data simulation --method prototypical --shots 1,3,5,10 --trials 50

# Test only k-NN baseline
.venv/bin/python -m evaluations.few_shot_test --data simulation --method knn --shots 1,3,5,10 --trials 30

# Test only Linear classifier baseline  
.venv/bin/python -m evaluations.few_shot_test --data simulation --method linear --shots 1,3,5,10 --trials 30

# Analyze feature quality with t-SNE visualization
.venv/bin/python -m evaluations.few_shot_test --data simulation --analyze_features --shots 5 --trials 10

# Custom encoder path (if you have different checkpoint)
.venv/bin/python -m evaluations.few_shot_test --data simulation --encoder_path ./ckpt/simulation/my_checkpoint.pth.tar --shots 1,5,10

# Custom data path
.venv/bin/python -m evaluations.few_shot_test --data simulation --data_path ./my_data/simulated_data/ --shots 1,3,5


# 1. Linear Prototypical Networks (best for linearly separable features)
.venv/bin/python -m evaluations.few_shot_test --data simulation --method linear_prototypical --shots 1,3,5,10 --trials 30

# 2. Metric Prototypical Networks (learnable distance metric)
.venv/bin/python -m evaluations.few_shot_test --data simulation --method metric_prototypical --shots 1,3,5,10 --trials 30

# 3. Hybrid Linear-Prototypical (combines both approaches)
.venv/bin/python -m evaluations.few_shot_test --data simulation --method hybrid --shots 1,3,5,10 --trials 30

# 4. Adaptive Prototypical Networks (adapts to shot number)
.venv/bin/python -m evaluations.few_shot_test --data simulation --method adaptive --shots 1,3,5,10,20 --trials 30

# Compare all methods (baseline + improved)
.venv/bin/python -m evaluations.few_shot_test --data simulation --method all --shots 1,3,5,10 --trials 20


# ==============================================================================
# Expected Results for Your TNC Encoder:
# ==============================================================================
# 
# 1-shot:  ~35-45% accuracy (vs 25% random)
# 3-shot:  ~45-55% accuracy  
# 5-shot:  ~50-60% accuracy
# 10-shot: ~55-65% accuracy
# 20-shot: ~60-70% accuracy
#
# Prototypical Networks should perform best, followed by k-NN, then Linear
# ==============================================================================