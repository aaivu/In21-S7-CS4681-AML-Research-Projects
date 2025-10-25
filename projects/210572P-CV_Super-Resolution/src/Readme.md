# ID-CodeFormer: Enhancing Identity Preservation in Blind Face Restoration

[**Original CodeFormer Repository**](https://github.com/SanjanaChamindu/CodeFormer.git)

This repository contains the official PyTorch implementation for the paper: **"Enhancing Identity Preservation in Blind Face Restoration via Supervised Feature Embedding"**.

ID-CodeFormer is an enhanced version of the state-of-the-art Blind Face Restoration model, CodeFormer. Our work addresses the "identity drift" problem, where restored faces can lose resemblance to the original subject. By integrating an identity-preserving loss supervised by a pre-trained ArcFace network, ID-CodeFormer significantly improves identity fidelity while maintaining the high perceptual quality and robustness of the original model.

### ✨ Key Features

- **Enhanced Identity Preservation:** Directly mitigates identity loss by incorporating a supervised feature embedding loss from a pre-trained face recognition network.
- **High-Fidelity Restoration:** Achieves a superior balance between perceptual quality and identity preservation, even on severely degraded images.
- **Robust Performance:** Retains the robustness of the original CodeFormer, effectively handling various real-world degradations like noise, blur, and compression artifacts.
- **Simple and Effective:** The identity-preserving module is easy to integrate into the original training pipeline without complex architectural changes.

### 🎨 Qualitative Results Comparison

Our model demonstrates a clear improvement in preserving the subject's identity compared to the baseline CodeFormer.

**_Note:_** _To display your image, upload collage.jpg to your repository (e.g., into an assets folder) and replace the placeholder text below with the correct Markdown link: !\[Qualitative Results\](./assets/collage.jpg)_

_From left to right: Degraded Input, Baseline CodeFormer, Our ID-CodeFormer, Ground Truth. Notice how ID-CodeFormer produces a restoration that is more faithful to the ground truth identity._

### 🏛️ Architecture Overview

ID-CodeFormer builds directly upon the robust architecture of CodeFormer. The core modification is the introduction of an **Identity-Preserving Loss** during the training phase.

- A low-quality input is processed by CodeFormer's encoder and Transformer to predict a sequence of discrete codebook indices.
- These indices are used to construct a feature map that is decoded into a restored face image.
- **Our Contribution:** During training, we feed both the restored image and the ground-truth image into a frozen, pre-trained ArcFace network to extract identity embeddings.
- The cosine distance between these embeddings is calculated as an identity loss (\$L_{ids}\$), which is then backpropagated to update the weights of the CodeFormer encoder and Transformer. This new supervisory signal explicitly guides the model to predict codes that better preserve the subject's identity.

**_Note:_** _To display your diagram, upload it to your repository and replace the placeholder text below with the correct Markdown link: !\[Architecture Diagram\](./assets/architecture.png)_

_Diagram illustrating the CodeFormer pipeline with the addition of the ArcFace-supervised identity loss feedback loop._

### 🚀 Getting Started

#### 1\. Prerequisites

- Python >= 3.8
- PyTorch >= 1.9
- Other dependencies can be installed via pip:  
    pip install -r requirements.txt  

#### 2\. Clone the Repository

git clone \[<https://github.com/SanjanaChamindu/CodeFormer.git\>](<https://github.com/SanjanaChamindu/CodeFormer.git>)  
cd CodeFormer  

#### 3\. Download Pre-trained Models

You will need the pre-trained weights for our ID-CodeFormer model, the baseline CodeFormer, and other dependent models like ArcFace.

Run the following script to download the necessary models:

python scripts/download_pretrained_models.py  

This will download and place the models in the weights/ directory.

### ⚙️ Inference

To restore faces in your own images, use the inference_codeformer.py script.

**Restore a single image:**

python inference_codeformer.py --input_path path/to/your/image.jpg --output_path results/  

**Key Arguments:**

- \--input_path: Path to the input image or a folder of images.
- \--output_path: Folder to save the restored images.
- \--w: A value between 0 and 1 to control the trade-off between quality (lower w) and fidelity (higher w). Default: 0.8.
- \--face_upsample: Set this flag to upsample the restored faces.

### 📈 Quantitative Results

Our method shows a significant improvement in identity similarity (ID-Sim) with a negligible impact on standard reconstruction metrics.

| **Method** | **PSNR ↑** | **SSIM ↑** | **LPIPS ↓** | **FID ↓** | **ID-Sim ↑** |
| --- | --- | --- | --- | --- | --- |
| CodeFormer | 26.54 | 0.782 | 0.231 | 35.1 | 0.58 |
| --- | --- | --- | --- | --- | --- |
| **ID-CodeFormer** | **26.51** | **0.781** | **0.233** | **35.5** | **0.72** |
| --- | --- | --- | --- | --- | --- |
| _Quantitative results on the LFW dataset._ |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |

### 🎓 Training ID-CodeFormer

To train the model from scratch, please follow these steps:

- **Prepare Datasets:** Prepare the FFHQ dataset as described in the original CodeFormer documentation.
- **Configure:** Modify the YAML configuration file in the options/ directory to specify dataset paths and training parameters.
- **Run Training:**  
    python basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher pytorch  

### 📜 Citation

If you find our work useful for your research, please consider citing our paper:

@inproceedings{sanjana2025idcodeformer,  
title={Enhancing Identity Preservation in Blind Face Restoration via Supervised Feature Embedding},  
author={Sanjana K. Y. C. and Thayasivam, Uthayasanker},  
booktitle={To Be Determined},  
year={2025}  
}  

### 🙏 Acknowledgments

This project is built upon the excellent work of the original [**CodeFormer**](https://github.com/sczhou/CodeFormer) authors. We are grateful for their high-quality public repository.

### 📄 License

This project is licensed under the [S-Lab License](https://www.google.com/search?q=LICENSE). It is available for non-commercial research purposes only.
