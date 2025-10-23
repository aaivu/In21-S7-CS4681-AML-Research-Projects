# Create a new file: basicsr/losses/identity_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arcface_arch import ResNetArcFace
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class IdentityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, model_path='weights/facelib/ms1mv3_arcface_r100_fp16.pth'):
        super(IdentityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.facenet = ResNetArcFace(block='IRBlock', layers=[2, 2, 2, 2], use_se=False)
        self.facenet.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.facenet.eval()
        self.facenet = self.facenet.cuda()

    def forward(self, pred, target):
        pred_embedding = self.facenet(F.interpolate(pred, size=112, mode='bilinear', align_corners=True))
        target_embedding = self.facenet(F.interpolate(target, size=112, mode='bilinear', align_corners=True))
        loss = 1 - F.cosine_similarity(pred_embedding, target_embedding, dim=1)
        return loss.mean() * self.loss_weight