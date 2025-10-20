## AutoFormerV2 Fine-Tuning on CIFAR-100
This project adapts the AutoFormerV2 Neural Architecture Search (NAS) framework for fine-tuning Vision Transformer models on the CIFAR-100 dataset.
AutoFormerV2 automatically searches for efficient transformer architectures and achieves competitive accuracy on large-scale datasets like ImageNet.
Here, we fine-tune and evaluate the pretrained AutoFormerV2-T (Tiny) model using Exponential Moving Average (EMA) and standard augmentation techniques.

## Pretrained Checkpoints (Baseline Models)
You can download pretrained ImageNet weights provided by the official AutoFormerV2 Model Zoo:

Model download link:

Model | Params. | Top-1 Acc. % | Top-5 Acc. % | Model
--- |:---:|:---:|:---:|:---:
AutoFormerV2-T | 28M | 82.1 | 95.8 | [link](https://github.com/silent-chen/AutoFormerV2-model-zoo/releases/download/v1.0.0/S3-T.pth)/[config](./configs/S3-T.yaml)
AutoFormerV2-S | 50M | 83.7 | 96.4 | [link](https://github.com/silent-chen/AutoFormerV2-model-zoo/releases/download/v1.0.0/S3-S.pth)/[config](./configs/S3-S.yaml)
AutoFormerV2-B | 71M | 84.0 | 96.6 | [link](https://github.com/silent-chen/AutoFormerV2-model-zoo/releases/download/v1.0.0/S3-B.pth)/[config](./configs/S3-B.yaml)

Save downloaded checkpoints into: ./checkpoints/

## Evaluate a Pretrained Model
To evaluate a pretrained AutoFormerV2 model directly on CIFAR-100, run:
python evaluation.py \
  --data-set CIFAR100 \
  --cfg ./configs/S3-T.yaml \
  --device cuda \
  --eval \
  --batch-size 64 \
  --resume ./checkpoints/S3-T.pth

Note: Since the pretrained models were trained on ImageNet (1000 classes), the classifier head may be automatically reinitialized for CIFAR-100.
This means accuracy will be low until fine-tuning.

## Fine-Tuning the Model on CIFAR-100
To fine-tune the pretrained model and enable EMA (Exponential Moving Average) tracking, run:
python evaluation.py \
  --data-set CIFAR100 \
  --cfg ./configs/S3-T.yaml \
  --device cuda \
  --epochs 20 \
  --batch-size 16 \
  --model-ema \
  --resume ./checkpoints/S3-T.pth \
  --output_dir ./checkpoints_cifar

Automatically saves checkpoints each epoch: ./checkpoints_cifar/S3-T_finetuned_epochX.pth

## Resume Fine-Tuning from a Saved Checkpoint
If you stopped training earlier and want to continue from where you left off:
python evaluation.py \
  --data-set CIFAR100 \
  --cfg ./configs/S3-T.yaml \
  --device cuda \
  --epochs 10 \
  --batch-size 32 \
  --model-ema \
  --resume ./checkpoints_cifar/S3-T_finetuned_epoch20.pth

## Additional Fine-Tuned Checkpoints
You can also download fine-tuned AutoFormerV2 checkpoints
https://drive.google.com/drive/folders/1x3E2cOVodWLr6iXMSveomtAf-onqGAy6?usp=sharing
