import os, random, math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

try:
    import decord
    decord.bridge.set_bridge('torch')
    _DECORD = True
except Exception:
    from torchvision.io import read_video
    _DECORD = False


def _read_split_list(path: str) -> List[str]:
    lines = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines


def _load_class_index(classind_path: str) -> dict:
    """classInd.txt format: '1 ApplyEyeMakeup' ..."""
    idx = {}
    if not os.path.exists(classind_path):
        return idx
    with open(classind_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            cid = int(parts[0])
            cname = parts[1]
            idx[cname] = cid - 1  # 0-based
    return idx


def _label_from_path(vpath: str) -> str:
    return Path(vpath).parent.name


def _temporal_indices(num_frames: int, frames: int, stride: int) -> np.ndarray:
    need = frames * stride
    if num_frames <= 0:
        return np.zeros(frames, dtype=np.int64)
    if num_frames >= need:
        start = random.randint(0, num_frames - need)
        idx = start + np.arange(0, need, stride)
    else:
        # sample with wrap/pad
        base = np.arange(0, num_frames)
        reps = math.ceil(need / num_frames)
        long = np.tile(base, reps)
        idx = long[:need:stride]
    return idx.astype(np.int64)


def _resize_crop_norm(frames: torch.Tensor, size: int, train: bool) -> torch.Tensor:
    # frames: (T, H, W, C) in [0,255] torch.uint8 or float
    if frames.dtype != torch.float32:
        frames = frames.float()
    frames = frames / 255.0

    T, H, W, C = frames.shape
    out = []
    for t in range(T):
        img = frames[t].permute(2,0,1)  # C,H,W
        # Resize shorter side to 256, then crop to 224
        h, w = img.shape[-2], img.shape[-1]
        scale = 256.0 / min(h, w)
        nh, nw = int(round(h*scale)), int(round(w*scale))
        img = TF.resize(img, [nh, nw], antialias=True)

        if train:
            img = TF.center_crop(img, size) if random.random()<0.15 else TF.resized_crop(
                img,
                top=random.randint(0, max(0, nh-size)),
                left=random.randint(0, max(0, nw-size)),
                height=size, width=size, size=(size, size), antialias=True
            )
            if random.random() < 0.5:
                img = TF.hflip(img)
        else:
            img = TF.center_crop(img, size)

        # Normalize (ImageNet)
        img = TF.normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        out.append(img)
    # (T, C, H, W) -> (C, T, H, W)
    vid = torch.stack(out, dim=0).permute(1,0,2,3).contiguous()
    return vid


class UCF101Clips(Dataset):
    """
    Returns: (video_tensor, label)
      - video_tensor shape: (C, T, H, W) with C=3, T=frames
      - label: int in [0, num_classes)
    """
    def __init__(self, root: str, split_txt: str, classind: Optional[str],
                 frames: int = 32, stride: int = 2, size: int = 224, train: bool = True):
        self.root = Path(root)
        self.frames = frames
        self.stride = stride
        self.size = size
        self.train = train

        self.class_map = _load_class_index(classind) if classind else {}

        lines = _read_split_list(split_txt)
        self.items: List[Tuple[str, int]] = []

        for ln in lines:
            parts = ln.split()
            if len(parts) == 2:
                rel, lbl = parts[0], int(parts[1]) - 1
            else:
                rel = parts[0]
                cname = _label_from_path(rel)
                lbl = self.class_map.get(cname, None)
                if lbl is None:
                    # try from real path if split has only filename
                    cname = _label_from_path(os.path.join("videos", rel))
                    lbl = self.class_map.get(cname, -1)
            vpath = self.root / "videos" / rel
            self.items.append((str(vpath), lbl))

    def __len__(self): return len(self.items)

    def _read_video(self, path: str) -> Tuple[torch.Tensor, int]:
        if _DECORD:
            vr = decord.VideoReader(path)
            n = len(vr)
            idx = _temporal_indices(n, self.frames, self.stride)
            frames = vr.get_batch(idx)  # (T, H, W, C) uint8 torch tensor
            return frames, n
        else:
            # torchvision fallback (slower; loads entire video then subsamples)
            video, _, _ = read_video(path, output_format="TCHW")  # (T,C,H,W) float[0,255]
            n = video.shape[0]
            idx = _temporal_indices(n, self.frames, self.stride)
            frames = video[idx].permute(0,2,3,1)  # to (T,H,W,C)
            return frames, n

    def __getitem__(self, i: int):
        path, label = self.items[i]
        frames, _ = self._read_video(path)
        vid = _resize_crop_norm(frames, self.size, train=self.train)  # (C,T,H,W)
        return vid, label
