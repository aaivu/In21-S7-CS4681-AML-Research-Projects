import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_compression import SpectralCompression
from .dual_path_rnn import DualPathRNN

def _down_size(size, layers=3, kernel=3, stride=2, pad=1):
    s = size
    for _ in range(layers):
        s = (s + 2*pad - kernel) // stride + 1
    return s  # integer downsampled size

class DPCRN_TwoStage(nn.Module):
    """
    Inputs:
      mag, phase: [B, F, T] with F = n_fft//2 + 1 (e.g., 601 for n_fft=1200)
    Path:
      [B, F, T] --SCM--> [B, Fc, T] --stack 2ch--> [B, 2, Fc, T]
      --Enc--> [B, 48, F', T'] --flatten(C*F')-> [B, T', 48*F'] --DPRNN-->
      [B, T', 48*F'] --reshape--> [B, 48, F', T'] --Dec-->
      mask_c [B, 2, Fc, T'] -> upsample to [B, F, T] -> apply on mag
    """
    def __init__(self, n_fft=1200, compressed_bins=256, fixed_bins=64):
        super().__init__()
        n_bins = n_fft // 2 + 1     # e.g., 601
        self.n_bins = n_bins
        self.Fc = compressed_bins   # compressed frequency bins (e.g., 256)

        # Spectral compression for both magnitude & phase to keep channels aligned
        self.scm_mag   = SpectralCompression(n_bins=n_bins, compressed_bins=compressed_bins, fixed_bins=fixed_bins)
        self.scm_phase = SpectralCompression(n_bins=n_bins, compressed_bins=compressed_bins, fixed_bins=fixed_bins)

        # Encoder (expects [B, 2, Fc, T])
        self.enc = nn.Sequential(
            nn.Conv2d(2, 16, 3, 2, 1), nn.ELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.ELU()
        )

        # Compute F' after three stride-2 convs on Fc
        Fp = _down_size(self.Fc, layers=3, kernel=3, stride=2, pad=1)  # e.g., 256 -> 32
        self.Fp = Fp
        D = 48 * Fp  # feature dim flattened for DPRNN

        # DPRNN that preserves D
        self.dp = DualPathRNN(input_size=D, hidden_size=D // 2)

        # Decoder back to 2 channels (magnitude & aux) at compressed grid
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(48, 32, 3, 2, 1, 1), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ELU(),
            nn.ConvTranspose2d(16,  2, 3, 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, mag, phase):
        """
        mag, phase: [B, F, T]  (F = n_fft//2 + 1)
        Returns: real, imag enhanced STFT parts on the ORIGINAL grid [B, F, T]
        """
        B, F_orig, T_orig = mag.shape

        # 1) Compress both along frequency → [B, Fc, T]
        mag_c   = self.scm_mag(mag)     # [B, Fc, T]
        phase_c = self.scm_phase(phase) # [B, Fc, T]

        # 2) Build 2-channel for Conv2d (H=F, W=T)
        x = torch.stack([mag_c, phase_c], dim=1)  # [B, 2, Fc, T]

        # 3) Encoder → [B, 48, F', T']
        x = self.enc(x)                            # [B, 48, Fp, Tp]
        B, C, Fp, Tp = x.shape  # C=48

        # 4) DPRNN over time, keep D=48*Fp features
        x = x.permute(0, 3, 1, 2).reshape(B, Tp, C * Fp)  # [B, Tp, 48*Fp]
        x = self.dp(x)                                    # [B, Tp, 48*Fp]
        x = x.view(B, Tp, C, Fp).permute(0, 2, 3, 1)      # [B, 48, Fp, Tp]

        # 5) Decoder to get a 2-ch mask on the COMPRESSED grid
        mask = self.dec(x)                                 # [B, 2, Fc, T_c]
        mask_mag_c = mask[:, 0]                            # [B, Fc, T_c]

        # 6) Upsample mask to ORIGINAL grid [B, F_orig, T_orig]
        mask_mag = F.interpolate(
            mask_mag_c.unsqueeze(1),  # [B,1,Fc,T_c]
            size=(F_orig, T_orig),    # -> [B,1,F_orig,T_orig]
            mode="bilinear",
            align_corners=False
        ).squeeze(1)                   # [B, F_orig, T_orig]

        # 7) Apply mask on original magnitude, rebuild RI
        enh_mag = mag * mask_mag
        real = enh_mag * torch.cos(phase)
        imag = enh_mag * torch.sin(phase)
        return real, imag
