# Student Usage Instructions

This document provides instructions on how to set up the environment, download necessary models, train the ID-CodeFormer model, and run inference using the CodeFormer repository.

## 1\. Setup Environment

First, clone the CodeFormer repository and install the required Python packages. This step also builds the custom CUDA extensions needed by the project.

\# Clone the CodeFormer repository from GitHub  
git clone \[<https://github.com/SanjanaChamindu/CodeFormer.git\>](<https://github.com/SanjanaChamindu/CodeFormer.git>)  
cd CodeFormer  
<br/>\# Create a new conda environment (optional but recommended)  
\# conda create -n codeformer python=3.8 -y  
\# conda activate codeformer  
<br/>\# Install the dependencies listed in requirements.txt  
pip install -r requirements.txt  
<br/>\# Set up the basicsr library, which includes custom CUDA ops  
python basicsr/setup.py develop  
<br/>\# Install dlib if you want to use it for face detection/cropping  
\# conda install -c conda-forge dlib  

## 2\. Download Pre-trained Models

You need several pre-trained models to start. This includes base CodeFormer/VQGAN weights, models for face detection/parsing (facelib), and potentially dlib models.

\# Download the official CodeFormer pre-trained model for inference  
python scripts/download_pretrained_models.py CodeFormer  
<br/>\# Download the facelib helper models (detection, parsing)  
python scripts/download_pretrained_models.py facelib  
<br/>\# --- Required for Training ---  
\# Download VQGAN and Stage 2 CodeFormer weights needed for training  
\# python scripts/download_pretrained_models.py CodeFormer_train  
<br/>\# --- Optional: For Dlib Face Detector ---  
\# python scripts/download_pretrained_models.py dlib  
<br/>\# --- Required for ID-CodeFormer Training ---  
\# Download the pre-trained ArcFace model for the identity loss  
\# wget -P ./weights/facelib \[<https://github.com/deepinsight/insightface/raw/master/model_zoo/arcface_torch/ms1mv3_arcface_r100_fp16.zip\>](<https://github.com/deepinsight/insightface/raw/master/model_zoo/arcface_torch/ms1mv3_arcface_r100_fp16.zip>)  
\# unzip -o -d ./weights/facelib ./weights/facelib/ms1mv3_arcface_r100_fp16.zip  

_Note: The ArcFace model download commands are commented out as they are specific to the Colab training notebook provided (Training_ID-CodeFormer.ipynb). You might need to manually download or adjust paths depending on your setup._

## 3\. Prepare Data

### 3.1 Download Dataset (FFHQ)

- The model is trained on the FFHQ dataset. If you need to train the model, download the dataset.
- You can use the provided Download_FFHQ_Dataset.ipynb notebook in Google Colab to download and transfer the ~88GB dataset to Google Drive. Be aware this takes a long time and requires significant storage.

### 3.2 Prepare Input Images for Inference

- Place test images (whole images or videos) in a designated input folder (e.g., inputs/whole_imgs).
- For testing on **cropped and aligned faces** (recommended for comparisons), first use the provided script:  
    \# Ensure dlib is installed  
    python scripts/crop_align_face.py -i \[your_input_folder\] -o inputs/cropped_faces  

- Place input images for **colorization** (grayscale, aligned 512x512) in a folder (e.g., inputs/gray_faces).
- Place input images for **inpainting** (masked, aligned 512x512) in a folder (e.g., inputs/masked_faces). Examples are provided in the repository.

## 4\. Training (ID-CodeFormer)

The training process involves multiple stages. The provided configuration options/train_id_codeformer.yml seems tailored for a combined or fine-tuning stage incorporating the identity loss.

- **Modify Configuration:**
  - Edit options/train_id_codeformer.yml.
  - Update dataroot_gt under datasets:train: to point to your FFHQ dataset location (e.g., /content/drive/My Drive/CodeFormer_Dataset/ffhq if using the Colab download notebook).
  - Adjust num_gpu, batch_size_per_gpu, num_worker_per_gpu based on your hardware.
  - Ensure pretrain_network_g points to the correct Stage 2 checkpoint (codeformer_stage2.pth).
  - Ensure the identity_opt section is present and configured (especially loss_weight).
- **Start Training:**  
    \# For PyTorch versions < 1.10  
    \# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --master_port=YOUR_PORT basicsr/train.py -opt options/train_id_codeformer.yml --launcher pytorch  
    <br/>\# For PyTorch versions >= 1.10  
    torchrun --nproc_per_node=NUM_GPUS --master_port=YOUR_PORT basicsr/train.py -opt options/train_id_codeformer.yml --launcher pytorch  
    <br/>Replace NUM_GPUS with the number of GPUs you want to use and YOUR_PORT with a free port number (e.g., 4321). Checkpoints will be saved in the experiments/ID_CodeFormer directory (or as named in the config).

_Note: The original CodeFormer training involves three stages (VQGAN, CodeFormer Stage 2 w=0, CodeFormer Stage 3 w=1). Refer to docs/train.md for details on training the original model._

## 5\. Inference

Use the following scripts for different tasks. Results are saved in the results/ directory by default.

### 5.1 Face Restoration (Cropped & Aligned Faces)

Ideal for direct comparisons or when input faces are already prepared.

python inference_codeformer.py \\  
\-w 0.5 \\  
\--has_aligned \\  
\--input_path \[image_folder | image_path\] \\  
\# Optional: --output_path \[your_output_folder\]  

- \-w: Fidelity weight (0 for better quality, 1 for better identity). Start with 0.5.

### 5.2 Whole Image Enhancement

Processes entire images, detects faces, enhances them, and optionally enhances the background.

python inference_codeformer.py \\  
\-w 0.7 \\  
\--input_path \[image_folder | image_path\] \\  
\# Optional: Enhance background  
\--bg_upsampler realesrgan \\  
\# Optional: Upsample face after restoration  
\--face_upsample \\  
\# Optional: Final image upscale factor  
\-s 2 \\  
\# Optional: Face detection model ('retinaface_resnet50', 'retinaface_mobile0.25', 'YOLOv5l', 'YOLOv5n', 'dlib')  
\--detection_model retinaface_resnet50 \\  
\# Optional: Output path  
\# --output_path \[your_output_folder\]  

- \-w: Fidelity weight. 0.7 is suggested for whole images.
- \-s: Final upscaling factor (e.g., 2 for 2x).

### 5.3 Video Enhancement

Processes video files frame by frame.

\# Ensure ffmpeg is installed (e.g., conda install -c conda-forge ffmpeg)  
python inference_codeformer.py \\  
\-w 1.0 \\  
\--input_path \[video_path.mp4\] \\  
\--bg_upsampler realesrgan \\  
\--face_upsample  
\# Optional: Specify output video FPS  
\# --save_video_fps 24  

- \-w: Fidelity weight. 1.0 often works well for videos to preserve identity.

### 5.4 Face Colorization (Cropped & Aligned Faces)

Colorizes grayscale or faded 512x512 aligned input faces.

python inference_colorization.py \\  
\--input_path \[image_folder | image_path\] \\  
\# Optional: --output_path \[your_output_folder\]  

### 5.5 Face Inpainting (Cropped & Aligned Faces)

Fills in masked regions on 512x512 aligned input faces. Mask should typically be white.

python inference_inpainting.py \\  
\--input_path \[image_folder | image_path\] \\  
\# Optional: --output_path \[your_output_folder\]  

_Refer to the original README.md and docs/train.md for more detailed information and advanced options._