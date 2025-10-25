import os, math, argparse, torch, torchaudio
from torch.utils.data import DataLoader
from tabulate import tabulate
from pesq import pesq
from pystoi import stoi

# Project imports
import sys
sys.path.append("/content")
from dpcrn_project.src.dataset import DemandVCTKDataset
from dpcrn_project.src.model.dpcrn_two_stage import DPCRN_TwoStage
from dpcrn_project.src.utils import stft, istft

# ---------- helpers ----------
@torch.no_grad()
def enhance_waveform(model, noisy_wav):
    """
    noisy_wav: [1, 1, T] Float32, 48 kHz
    returns enh_wav: [T] Float32, 48 kHz
    """
    Ns = stft(noisy_wav)             # complex [1, F, T]
    mag, ph = Ns.abs(), Ns.angle()
    r, i = model(mag, ph)            # real/imag [1, F, T]
    enh_spec = torch.complex(r, i)
    enh = istft(enh_spec).squeeze(0).cpu()  # [T]
    return enh

def power(x):
    return float((x**2).mean().item())

def snr_improvement(clean, enh):
    """
    SNR_improved = 10*log10( Ps / P(s - ŝ) )
    both clean & enh are 1D tensors (same length), arbitrary fs
    """
    err = clean - enh
    Ps = power(clean)
    Pe = power(err)
    return 10.0 * math.log10((Ps + 1e-12) / (Pe + 1e-12))

def resample_to(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x
    return torchaudio.functional.resample(x, fs_in, fs_out)

def eval_batch_metrics(clean_wav_48k, noisy_wav_48k, enh_wav_48k,
                       for_pesq_stoi_fs=16000):
    """
    Returns dict with PESQ, STOI for Noisy and Enhanced, and SNRi.
    PESQ: wideband at 16 kHz, STOI: 16 kHz
    """
    # match lengths (pad/truncate) for fair metric computation
    T = min(clean_wav_48k.numel(), noisy_wav_48k.numel(), enh_wav_48k.numel())
    clean_48k = clean_wav_48k[:T].contiguous()
    noisy_48k = noisy_wav_48k[:T].contiguous()
    enh_48k   = enh_wav_48k[:T].contiguous()

    # SNRi at 48 kHz (native)
    snri = snr_improvement(clean_48k, enh_48k)

    # Resample to 16 kHz for PESQ/STOI
    clean_16k = resample_to(clean_48k.unsqueeze(0), 48000, for_pesq_stoi_fs).squeeze(0)
    noisy_16k = resample_to(noisy_48k.unsqueeze(0), 48000, for_pesq_stoi_fs).squeeze(0)
    enh_16k   = resample_to(enh_48k.unsqueeze(0),   48000, for_pesq_stoi_fs).squeeze(0)

    c = clean_16k.numpy()
    n = noisy_16k.numpy()
    e = enh_16k.numpy()

    # PESQ wideband at 16 kHz
    pesq_noisy = pesq(for_pesq_stoi_fs, c, n, 'wb')
    pesq_enh   = pesq(for_pesq_stoi_fs, c, e, 'wb')

    # STOI (supports 16 kHz)
    stoi_noisy = stoi(c, n, for_pesq_stoi_fs, extended=False)
    stoi_enh   = stoi(c, e, for_pesq_stoi_fs, extended=False)

    return {
        "PESQ_Noisy": pesq_noisy,
        "PESQ_Enh":   pesq_enh,
        "STOI_Noisy": stoi_noisy * 100.0,  # %
        "STOI_Enh":   stoi_enh   * 100.0,  # %
        "SNRi":       snri
    }

def maybe_dns_mos(wav_48k):
    """
    Optional: DNS-MOS (P.835-style proxy). Requires `dnsmos` package & model.
    Disabled by default to keep runtime light.
    """
    return None  # placeholder (turn on if you install dnsmos)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", default="/content/MiniVCTK")
    ap.add_argument("--noise_dir", default="/content/DEMAND_noise")
    ap.add_argument("--ckpt", default="/content/dpcrn_project/checkpoints/dpcrn_two_stage_ep3.pth")
    ap.add_argument("--num_samples", type=int, default=30, help="evaluate N random samples")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default="/content/dpcrn_project/results/eval_metrics.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    ds = DemandVCTKDataset(args.clean_dir, args.noise_dir, sample_len=4.0)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Model
    model = DPCRN_TwoStage().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Loop
    import csv
    rows = []
    taken = 0
    for noisy, clean in dl:
        noisy = noisy.to(device)  # [B,1,T]
        clean = clean.to(device)

        for b in range(noisy.size(0)):
            if taken >= args.num_samples:
                break

            enh = enhance_waveform(model, noisy[b:b+1])              # [T] @48k
            noisy_48k = noisy[b].squeeze(0).detach().cpu()           # [T]
            clean_48k = clean[b].squeeze(0).detach().cpu()           # [T]

            m = eval_batch_metrics(clean_48k, noisy_48k, enh, for_pesq_stoi_fs=16000)
            rows.append(m)
            taken += 1
        if taken >= args.num_samples:
            break

    # Aggregate
    def avg(key): return sum(r[key] for r in rows) / len(rows)
    summary = {
        "PESQ_Noisy": avg("PESQ_Noisy"),
        "PESQ_Enh":   avg("PESQ_Enh"),
        "STOI_Noisy": avg("STOI_Noisy"),
        "STOI_Enh":   avg("STOI_Enh"),
        "SNRi":       avg("SNRi"),
        "N": len(rows),
    }

    # Print table similar to the paper section
    table = [
        ["Noisy",      f"{summary['PESQ_Noisy']:.2f}", f"{summary['STOI_Noisy']:.1f}", "—"],
        ["Proposed",   f"{summary['PESQ_Enh']:.2f}",   f"{summary['STOI_Enh']:.1f}",   f"{summary['SNRi']:.2f} dB"],
    ]
    print("\nEvaluation on DEMAND + MiniVCTK (random subset)")
    print(tabulate(table, headers=["Model", "PESQ", "STOI (%)", "SNRi"], tablefmt="github"))
    print(f"\nComputed over N={summary['N']} files.")

    # Save row-wise CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n✅ Saved per-file metrics to {args.out_csv}")

if __name__ == "__main__":
    main()
