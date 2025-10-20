# Healthcare AI: Medical Imaging Usage Instructions

## Cloning the Project

Since this repository contains multiple projects, clone only this specific project folder:

```bash
git clone --no-checkout https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd In21-S7-CS4681-AML-Research-Projects
git sparse-checkout init --cone
git sparse-checkout set projects/210329E-Healthcare-AI_Medical-Imaging
git checkout
```

## Setup

1. Navigate to the source directory:
   ```bash
   cd projects/210329E-Healthcare-AI_Medical-Imaging/src
   ```

2. Download the model files as listed in `All models.txt` and `Final Models.txt`.

3. Install dependencies:
   ```bash
   pip install numpy pandas pillow matplotlib seaborn torch torchvision scikit-learn opencv-python tqdm
   ```

   **Required Libraries:**
   - `numpy` - Numerical computing
   - `pandas` - Data manipulation
   - `pillow` - Image processing (PIL)
   - `matplotlib` - Plotting and visualization
   - `seaborn` - Statistical visualization
   - `torch` - PyTorch deep learning framework
   - `torchvision` - Computer vision utilities for PyTorch
   - `scikit-learn` - Machine learning algorithms and metrics
   - `opencv-python` - Computer vision library
   - `tqdm` - Progress bars

4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

5. Open and run the notebooks in order:
   - `individual_model_testing_comparison.ipynb`
   - `ensemble_simple_average.ipynb`
   - `ensemble_gradcam_visualization.ipynb`
   - `ensemble_uncertainty_quantification.ipynb`