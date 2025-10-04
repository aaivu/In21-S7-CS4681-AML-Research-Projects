## Finetune Real-ESRGAN on scannet++ dataset

Use these instructions to finetune a pretrained Real-ESRGAN model.

**0. My model**

You can find my model at [weights/Finetuned for scannet++.pth]()

to infer using the model:

```bash
!python inference_realesrgan.py -n Finetuned for scannet++ -i datasets/scannet++/iphone --outscale 4
```

**1. Prepare dataset**

Assume that you already have two folders:

- **gt folder** (Ground-truth, high-resolution images): *datasets/scannet++/iphone*
- **lq folder** (Low quality, low-resolution images): *datasets/scannet++/dslr*

Then, you can prepare the meta_info txt file using the script [scripts/generate_meta_info_pairdata.py](scripts/generate_meta_info_pairdata.py):

```bash
python scripts/generate_meta_info_pairdata.py --input datasets/scannet++/dslr datasets/scannet++/iphone --meta_info datasets/scannet++/meta_info/meta_info.txt
```

**2. Download pre-trained models**

Download pre-trained models into `experiments/pretrained_models`.

- *RealESRGAN_x4plus.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
    ```

- *RealESRGAN_x4plus_netD.pth*
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**3. Finetune**

Modify [options/finetune_realesrgan_x4plus_pairdata.yml](options/finetune_realesrgan_x4plus_pairdata.yml) accordingly, especially the `datasets` part:

```yml
train:
    name: ScanNet++
    type: RealESRGANPairedDataset
    dataroot_gt: datasets/scannet++  # modify to the root path of your folder
    dataroot_lq: datasets/scannet++  # modify to the root path of your folder
    meta_info: datasets/scannet++/meta_info/meta_info.txt  # modify to your own generate meta info txt
    io_backend:
        type: disk
```

We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --launcher pytorch --auto_resume
```

Finetune with **a single GPU**:
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume
```

