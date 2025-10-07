# Usage Instructions

## 1. Clone Repository & Open Notebook

Since the repository contains multiple projects, you can clone only this project folder using Git sparse-checkout:

```bash
git clone --no-checkout https://github.com/aaivu/In21-S7-CS4681-AML-Research-Projects.git
cd .\In21-S7-CS4681-AML-Research-Projects\

# Enable sparse-checkout
git sparse-checkout init --cone

# Set the folder you want to checkout
git sparse-checkout set projects/210471F-Healthcare-AI_Medical-Imaging

# Chekout branch
git checkout main 
```

Now only the contents of this project folder will be downloaded.

Or open directly in Kaggle or Google Colab by uploading the notebook.

## 2. Install Dependencies

To run this notebook, install the following Python libraries:

```bash
pip install numpy pandas matplotlib opencv-python nibabel scikit-image seaborn \
scipy scikit-learn tensorflow
```


| Library                   | Purpose                                                    |
| --------------------------- | ------------------------------------------------------------ |
| **numpy**                 | Numerical computations, array operations                   |
| **pandas**                | Data manipulation and handling CSV/metadata                |
| **matplotlib**            | Data visualization, plotting CT scans and results          |
| **seaborn**               | Statistical visualizations (confusion matrix, plots)       |
| **scipy**                 | Scientific computations, rotations, shifts, augmentation   |
| **opencv-python** (`cv2`) | Image preprocessing, CLAHE enhancement, cropping           |
| **nibabel**               | Reading and processing medical imaging files (`.nii`)      |
| **scikit-image**          | Image transformations, resizing, gamma correction          |
| **scikit-learn**          | Model evaluation, train-test split, class weights          |
| **tensorflow**            | Deep learning framework (ConvNeXt, EfficientNet, training) |

📌 *Standard Python libraries used (no installation needed):* `os`, `glob`, `gc`, `random`

## 3. Prepare Dataset

The notebook expects COVID-19 CT scan datasets:

- COVID-19-CT-20 dataset
- MedSeg COVID-19 CT segmentation dataset

Refer to the original sources mentioned in the `data` directory for download links of these datasets.

Download and place them under:

```
/input/covid19-ct-scans/
/input/medseg-covid-dataset-2/
```

(or update paths inside the notebook).

## 4. Run the Notebook

The notebook has three main phases:

- Data Preprocessing
  * Applies CLAHE enhancement, lung cropping, resizing, and normalization.

  * Saves processed CT scan images and labels as .npy files.

- Data Augmentation

  * Balances COVID vs Non-COVID cases using rotations, flips, noise injection, and gamma correction.

- Training Phase

  * Trains a multi-branch ConvNeXt model for COVID classification.

  * Includes fine-tuning with callbacks (early stopping, learning rate scheduling).

Simply run all cells in order (Run All). Intermediate results (plots, preprocessed images, metrics) will be displayed.
