import os
import zipfile
import tarfile
from tqdm import tqdm

DEMAND_DIR = "/content/DEMAND_noise"

# Find all candidate archive files
archives = [f for f in os.listdir(DEMAND_DIR) if f.endswith((".zip", ".tar.gz", ".tar"))]

if not archives:
    print(f"No archives found in {DEMAND_DIR}.")
else:
    print(f"Found {len(archives)} DEMAND archives:")
    for f in archives:
        print(" •", f)

    for arch in tqdm(archives, desc="Extracting DEMAND archives"):
        arch_path = os.path.join(DEMAND_DIR, arch)
        out_folder = os.path.join(DEMAND_DIR, arch.replace(".zip", "").replace(".tar.gz", "").replace(".tar", ""))
        os.makedirs(out_folder, exist_ok=True)

        try:
            # Try zip extraction first
            if zipfile.is_zipfile(arch_path):
                with zipfile.ZipFile(arch_path, "r") as zf:
                    zf.extractall(out_folder)
                print(f"✅ Extracted ZIP: {arch}")
            # Then try tar extraction
            elif tarfile.is_tarfile(arch_path):
                with tarfile.open(arch_path, "r:*") as tf:
                    tf.extractall(out_folder)
                print(f"✅ Extracted TAR: {arch}")
            else:
                print(f"⚠️ Not a recognized archive format: {arch}")
        except Exception as e:
            print(f"❌ Failed to extract {arch}: {e}")

print("\n🎧 Extraction complete!")
print("Available folders:")
print(os.listdir(DEMAND_DIR))
