#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download MiniVCTK and DEMAND-noise from Google Drive and extract so that the
data folders live at the same filesystem level as the repo folder:

  /path/
    dpcrn_project/
    MiniVCTK/
    DEMAND_noise/

Usage (run from inside dpcrn_project/):
  python scripts/download_from_gdrive.py

Options:
  --root <path>     Destination root (default: parent of repo root)
  --skip_minivctk   Skip MiniVCTK download/extract
  --skip_demand     Skip DEMAND download/extract
  --force           Redownload and re-extract even if files exist
"""

import argparse
import zipfile
from pathlib import Path
import subprocess
import sys
import hashlib

# Google Drive file IDs you provided
MINIVCTK_ID = "1wCxETjomzhJsT3Yfn-aAbA1dLg8BrP5L"
DEMAND_ID   = "1U9u5ann_VtOJ17cZ8dvUG3ikOV0jaGUe"


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "--quiet"])


def md5sum(p: Path, chunk=1024 * 1024):
    h = hashlib.md5()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def download_file(file_id: str, out_path: Path, desc: str):
    import gdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {desc}")
    print(f"  URL: {url}")
    print(f"  -> {out_path}")
    gdown.download(url, str(out_path), quiet=False)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or empty file: {out_path}")
    size_mb = out_path.stat().st_size / 1e6
    print(f"Downloaded {desc}: {out_path.name} ({size_mb:.1f} MB)")


def safe_extract_zip(zip_path: Path, dest_dir: Path):
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Prevent path traversal
        for m in zf.namelist():
            dest_path = dest_dir / m
            if not str(dest_path.resolve()).startswith(str(dest_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip: {m}")
        zf.extractall(dest_dir)
    print(f"Extracted to {dest_dir}")


def maybe_flatten_singleton(root_dir: Path):
    """If extraction created a single top-level folder, move its contents up one level."""
    entries = [p for p in root_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in inner.iterdir():
            target = root_dir / item.name
            if target.exists():
                continue
            item.rename(target)
        try:
            inner.rmdir()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Download MiniVCTK and DEMAND-noise from Google Drive and place them next to the repo folder."
    )
    parser.add_argument("--root", type=str, default=None,
                        help="Destination root (default: parent directory of the repo).")
    parser.add_argument("--skip_minivctk", action="store_true", help="Skip MiniVCTK download/extract")
    parser.add_argument("--skip_demand", action="store_true", help="Skip DEMAND download/extract")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract even if files exist")
    args = parser.parse_args()

    # Resolve repo root as scripts/../
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root.parent  # one level above dpcrn_project
    root = Path(args.root).expanduser().resolve() if args.root else default_root

    print(f"Repo root : {repo_root}")
    print(f"Target root: {root}  (expected MiniVCTK/ and DEMAND_noise/ here)")

    root.mkdir(parents=True, exist_ok=True)
    ensure_gdown()

    # Destination paths next to the repo folder
    mini_zip   = root / "MiniVCTK.zip"
    mini_dir   = root / "MiniVCTK"
    demand_zip = root / "DEMAND_noise.zip"
    demand_dir = root / "DEMAND_noise"

    # MiniVCTK
    if not args.skip_minivctk:
        if args.force or not mini_zip.exists():
            download_file(MINIVCTK_ID, mini_zip, "MiniVCTK")
        else:
            print(f"Using existing: {mini_zip.name}")

        mini_dir.mkdir(parents=True, exist_ok=True)
        if args.force or not any(mini_dir.iterdir()):
            safe_extract_zip(mini_zip, mini_dir)
            maybe_flatten_singleton(mini_dir)
        print(f"MiniVCTK folder: {mini_dir}")

    # DEMAND
    if not args.skip_demand:
        if args.force or not demand_zip.exists():
            download_file(DEMAND_ID, demand_zip, "DEMAND-noise")
        else:
            print(f"Using existing: {demand_zip.name}")

        demand_dir.mkdir(parents=True, exist_ok=True)
        if args.force or not any(demand_dir.iterdir()):
            safe_extract_zip(demand_zip, demand_dir)
            maybe_flatten_singleton(demand_dir)
        print(f"DEMAND folder: {demand_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
