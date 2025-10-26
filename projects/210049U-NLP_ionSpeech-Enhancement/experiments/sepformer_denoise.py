#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SepFormer Separation + Lightweight Denoiser
Run via: python sepformer_denoise.py
"""

# Install dependencies (run once)
# Uncomment and run if packages are missing
# import os
# os.system("pip install torch torchaudio speechbrain soundfile")

# Imports

import os
import torch
import torchaudio
import gc
import torch.nn as nn
from speechbrain.inference import SepformerSeparation

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def clear_memory():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

# Path to your mixture file
MIXTURE_PATH = "../src/input_data/mixture.wav"

# Load audio
mixture, sr = torchaudio.load(MIXTURE_PATH)
print(f"Mixture shape: {mixture.shape}, Sample rate: {sr}")

# Load SepFormer & separate mixture
sep_model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix",
    run_opts={"device": device}
)

# Separate mixture
est_sources = sep_model.separate_file(path=MIXTURE_PATH).cpu()  # [1, T, n_src]
print("Raw separated shape:", est_sources.shape)

# Convert [1, T, 2] -> [2, 1, T] for Conv1D
sources = est_sources.squeeze(0).T.unsqueeze(1)  # [2,1,T]
print("Corrected sources shape:", sources.shape)
clear_memory()

# Define lightweight denoiser
class SimpleDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 1, 9, padding=4)
        )

    def forward(self, x):
        return self.net(x)

denoiser = SimpleDenoiser().to(device).eval()
clear_memory()

#Denoise sources safely
def denoise_sources(denoiser, sources):
    outs = []
    for s in sources:
        x = s.unsqueeze(0).to(device)  # [1,1,T]
        with torch.no_grad():
            y = denoiser(x).squeeze().cpu()
        outs.append(y)
        clear_memory()
    return torch.stack(outs)  # [n_src, T]

denoised_sources = denoise_sources(denoiser, sources)
print("Denoised sources shape:", denoised_sources.shape)
clear_memory()


# Save results

# Create output directory
OUTPUT_DIR = "../src/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save files to output directory
torchaudio.save(os.path.join(OUTPUT_DIR, "sep_s1.wav"), sources[0], sr)
torchaudio.save(os.path.join(OUTPUT_DIR, "den_s1.wav"), denoised_sources[0].unsqueeze(0), sr)
torchaudio.save(os.path.join(OUTPUT_DIR, "sep_s2.wav"), sources[1], sr)
torchaudio.save(os.path.join(OUTPUT_DIR, "den_s2.wav"), denoised_sources[1].unsqueeze(0), sr)

print(f"Saved files to {OUTPUT_DIR}:")
print("  - sep_s1.wav, den_s1.wav")
print("  - sep_s2.wav, den_s2.wav")
