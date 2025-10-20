import soundfile as sf
import numpy as np
from pystoi import stoi
from pesq import pesq
import mir_eval
import librosa

# Load files
def load_audio(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono if needed
    return audio, sr

# Files
refs = ["../src/input_data/s1_clean.wav", "../src/input_data/s2_clean.wav"]
seps = ["../src/output/sep_s1.wav", "../src/output/sep_s2.wav"]
dens = ["../src/output/den_s1.wav", "../src/output/den_s2.wav"]


print("Evaluating...")

for i, (ref_path, sep_path, den_path) in enumerate(zip(refs, seps, dens), start=1):
    ref, sr = load_audio(ref_path)
    sep, _ = load_audio(sep_path)
    den, _ = load_audio(den_path)

    # Match lengths
    min_len = min(len(ref), len(sep), len(den))
    ref, sep, den = ref[:min_len], sep[:min_len], den[:min_len]

    # === SI-SDR (mir_eval) ===
    sdr_sep, _, _, _ = mir_eval.separation.bss_eval_sources(ref[None, :], sep[None, :])
    sdr_den, _, _, _ = mir_eval.separation.bss_eval_sources(ref[None, :], den[None, :])

    # === PESQ ===
    # sr = 8000 in your case
    if sr == 8000:
        mode = 'nb'  # narrowband
    elif sr >= 16000:
        mode = 'wb'  # wideband
    else:
        raise ValueError(f"Unsupported sample rate: {sr}")
    pesq_sep = pesq(sr, ref, sep, mode)
    pesq_den = pesq(sr, ref, den, mode)

    # === STOI ===
    stoi_sep = stoi(ref, sep, sr, extended=False)
    stoi_den = stoi(ref, den, sr, extended=False)

    print(f"\n🎧 Source {i}")
    print(f"  SI-SDR (Sep): {sdr_sep[0]:.2f} dB   |  SI-SDR (Den): {sdr_den[0]:.2f} dB")
    print(f"  PESQ   (Sep): {pesq_sep:.2f}        |  PESQ   (Den): {pesq_den:.2f}")
    print(f"  STOI   (Sep): {stoi_sep:.3f}        |  STOI   (Den): {stoi_den:.3f}")
