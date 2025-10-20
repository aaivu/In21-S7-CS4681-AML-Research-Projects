import os, glob, random, torch, torchaudio

class DemandVCTKDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noise_dir, snr_range=(0,15), sample_len=4.0):
        self.clean_files = glob.glob(f"{clean_dir}/**/*.wav", recursive=True)
        # 🔧 FIX: look through all subfolders, even DKITCHEN_48k/DKITCHEN
        self.noise_files = glob.glob(f"{noise_dir}/**/**/*.wav", recursive=True)
        assert len(self.clean_files) > 0, f"No clean files found in {clean_dir}"
        assert len(self.noise_files) > 0, f"No noise files found in {noise_dir}"
        print(f"✅ Found {len(self.clean_files)} clean and {len(self.noise_files)} noise files.")
        self.snr_range = snr_range
        self.sample_len = sample_len
        self.target_sr = 48000

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # --- Clean speech ---
        clean_path = self.clean_files[idx]
        clean, sr = torchaudio.load(clean_path)
        clean = torch.mean(clean, dim=0, keepdim=True)
        if sr != self.target_sr:
            clean = torchaudio.functional.resample(clean, sr, self.target_sr)
        seg_len = int(self.sample_len * self.target_sr)
        if clean.size(1) > seg_len:
            start = random.randint(0, clean.size(1)-seg_len)
            clean = clean[:, start:start+seg_len]
        else:
            clean = torch.nn.functional.pad(clean, (0, seg_len-clean.size(1)))

        # --- Noise ---
        noise_path = random.choice(self.noise_files)
        noise, sr_n = torchaudio.load(noise_path)
        noise = torch.mean(noise, dim=0, keepdim=True)
        if sr_n != self.target_sr:
            noise = torchaudio.functional.resample(noise, sr_n, self.target_sr)
        if noise.size(1) > seg_len:
            start = random.randint(0, noise.size(1)-seg_len)
            noise = noise[:, start:start+seg_len]
        else:
            noise = torch.nn.functional.pad(noise, (0, seg_len-noise.size(1)))

        # --- Mix at random SNR ---
        snr = random.uniform(*self.snr_range)
        clean_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()
        scale = torch.sqrt(clean_power/(10**(snr/10)*noise_power))
        noisy = clean + scale*noise

        return noisy, clean
