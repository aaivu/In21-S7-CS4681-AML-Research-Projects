import os, wget, tarfile, glob, torchaudio

root = "/content"
vctk_dir = f"{root}/MiniVCTK"
demand_dir = f"{root}/DEMAND_noise"

def download_and_extract(url, out_dir):
    fname = url.split("/")[-1]
    out_path = os.path.join(root, fname)
    if not os.path.exists(out_dir):
        print(f"Downloading {fname} ...")
        wget.download(url, out=root)
        print("\nExtracting...")
        with tarfile.open(out_path, "r:gz") as tar:
            tar.extractall(root)
        print("Done.")
    else:
        print(f"{out_dir} exists, skipping download.")

# ---- small subsets (~1 GB total) ----
if not os.path.exists(vctk_dir):
    download_and_extract("https://datashare.ed.ac.uk/download/10283/3791/9/VCTK-Corpus-0.92.tar.gz", vctk_dir)
    os.makedirs(vctk_dir, exist_ok=True)
    speakers = ['p225','p226','p227','p228']
    for spk in speakers:
        for wav in glob.glob(f"{root}/VCTK-Corpus-0.92/wav48/{spk}/*.wav"):
            torchaudio.save(
                os.path.join(vctk_dir, os.path.basename(wav)),
                torchaudio.load(wav)[0][:,:int(3*48000)], 48000)
else:
    print("VCTK subset ready.")

if not os.path.exists(demand_dir):
    download_and_extract("https://zenodo.org/record/1227121/files/DEMAND.tar.gz", demand_dir)
    os.makedirs(demand_dir, exist_ok=True)
    scenes = ['office','kitchen','car']
    for sc in scenes:
        for wav in glob.glob(f"{root}/DEMAND/{sc}/*.wav"):
            torchaudio.save(
                os.path.join(demand_dir, os.path.basename(wav)),
                torchaudio.load(wav)[0][:,:int(3*48000)], 48000)
else:
    print("DEMAND subset ready.")

print("✅ Dataset prepared at", vctk_dir, "and", demand_dir)
