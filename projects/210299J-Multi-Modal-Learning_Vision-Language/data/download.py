
#!/usr/bin/env python3
"""
nocaps_downloader.py
--------------------
Download nocaps images (from Open Images) by image IDs in nocaps JSON.

This script avoids legacy figure-eight links and fetches pixels directly
from the public S3 bucket: `open-images-dataset`.

Usage examples
--------------
# 1) From nocaps validation JSON (auto-infers split=validation)
python nocaps_downloader.py \

    --ann /path/to/nocaps_val_4500_captions.json \

    --out_dir /data/nocaps/images

# 2) From nocaps test image-info JSON
python nocaps_downloader.py \

    --ann /path/to/nocaps_test_image_info.json \

    --split test \

    --out_dir /data/nocaps/images_test

# 3) From a prepared IDs file (one per line, format: validation/<id> or test/<id>)
python nocaps_downloader.py \

    --ids_file /path/to/nocaps_val_ids.txt \

    --out_dir /data/nocaps/images

Notes
-----
* The Open Images downloader expects IDs like "validation/<id>" or "test/<id>"
  (see Google Open Images download docs). We follow the same format.
* Files are saved flat as <out_dir>/<image_id>.jpg
* No third-party deps required. Concurrency via ThreadPoolExecutor.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

OPENIMAGES_S3 = "https://open-images-dataset.s3.amazonaws.com"

def infer_split_from_filename(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    if "val" in name or "validation" in name:
        return "validation"
    if "test" in name:
        return "test"
    return None

def read_ids_from_idsfile(ids_file: str) -> List[Tuple[str, str]]:
    """Reads lines of the form `<split>/<image_id>` and returns list of tuples."""
    pairs = []
    with open(ids_file, "r") as f:
        for line in f:
            s = line.strip().replace(".jpg", "")
            if not s:
                continue
            try:
                split, image_id = s.split("/", 1)
            except ValueError:
                raise ValueError(f"Bad line in ids file (expected '<split>/<id>'): {s}")
            pairs.append((split, image_id))
    return pairs

def extract_ids_from_nocaps_json(ann_path: str, id_key: str = "open_images_id",
                                 domains: Optional[Set[str]] = None) -> List[str]:
    """Parse nocaps JSON and return list of unique Open Images IDs (hex strings)."""
    data = json.load(open(ann_path, "r"))
    imgs = data.get("images", [])
    ids = []
    for im in imgs:
        if domains is not None:
            d = im.get("domain")
            if d not in domains:
                continue
        oid = im.get(id_key) or im.get("open_images_id") or im.get("open_imagesid") or im.get("id")
        if not oid:
            continue
        # strip any extension if present
        oid = str(oid).replace(".jpg", "").strip()
        ids.append(oid)
    # deduplicate preserving order
    seen = set()
    unique = []
    for x in ids:
        if x not in seen:
            unique.append(x); seen.add(x)
    return unique

def build_id_pairs(ids: Iterable[str], split: str) -> List[Tuple[str, str]]:
    return [(split, i) for i in ids]

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def url_for(split: str, image_id: str) -> str:
    return f"{OPENIMAGES_S3}/{split}/{image_id}.jpg"

def download_one(split: str, image_id: str, out_dir: str, timeout: float = 20.0,
                 retries: int = 3, force: bool = False) -> Tuple[str, bool, str]:
    """Download one image. Returns (image_id, success, message)."""
    dest = Path(out_dir) / f"{image_id}.jpg"
    if dest.exists() and not force:
        return image_id, True, "exists"
    u = url_for(split, image_id)
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "nocaps-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    last_err = f"HTTP {resp.status}"
                    raise RuntimeError(last_err)
                content = resp.read()
            with open(dest, "wb") as f:
                f.write(content)
            return image_id, True, "ok"
        except Exception as e:
            last_err = str(e)
            # exponential backoff
            time.sleep(min(2 ** attempt, 10))
    return image_id, False, last_err

def write_ids_txt(id_pairs: List[Tuple[str, str]], path: str):
    with open(path, "w") as f:
        for split, image_id in id_pairs:
            f.write(f"{split}/{image_id}\n")

def main():
    p = argparse.ArgumentParser(description="Download nocaps images from Open Images S3")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ann", type=str, help="Path to nocaps JSON (val captions or test image-info)")
    src.add_argument("--ids_file", type=str, help="Path to a file with lines '<split>/<image_id>'")
    p.add_argument("--split", type=str, default=None,
                   choices=["train", "validation", "test", "challenge2018"],
                   help="Dataset split. If --ann is given, auto-infers from filename when possible")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save images")
    p.add_argument("--domains", type=str, nargs="*", default=None,
                   help="Optional domain filter for JSON: in-domain near-domain out-domain")
    p.add_argument("--num_workers", type=int, default=8, help="Concurrent download workers")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    p.add_argument("--retries", type=int, default=3, help="Retry attempts per file")
    p.add_argument("--force", action="store_true", help="Redownload even if file exists")
    p.add_argument("--max_images", type=int, default=None, help="Download only the first N images (debug)")
    p.add_argument("--save_ids_txt", type=str, default=None, help="Where to save generated ids list (optional)")
    args = p.parse_args()

    # Resolve input -> list of (split, id)
    if args.ids_file:
        id_pairs = read_ids_from_idsfile(args.ids_file)
        # if user also passed --split, ensure consistent
        if args.split and any(sp != args.split for sp, _ in id_pairs):
            print("[warn] --split differs from ids file entries; using ids file values", file=sys.stderr)
        # derive split if uniform, else None
        splits = {sp for sp, _ in id_pairs}
        split = args.split or (splits.pop() if len(splits) == 1 else None)
    else:
        if not args.split:
            auto = infer_split_from_filename(args.ann)
            if not auto:
                print("ERROR: Could not infer split from filename. Please pass --split.", file=sys.stderr)
                sys.exit(2)
            args.split = auto
        domains = set(args.domains) if args.domains else None
        ids = extract_ids_from_nocaps_json(args.ann, domains=domains)
        if args.max_images:
            ids = ids[:args.max_images]
        split = args.split
        id_pairs = build_id_pairs(ids, split)

    # Optionally save an ids txt (handy for Open Images official downloader too)
    if args.save_ids_txt:
        ensure_dir(Path(args.save_ids_txt).parent)
        write_ids_txt(id_pairs, args.save_ids_txt)

    ensure_dir(args.out_dir)

    total = len(id_pairs)
    print(f"[nocaps] Split={split or 'mixed'}  Count={total}  Out={args.out_dir}", flush=True)
    if total == 0:
        print("Nothing to download. Exiting.")
        return

    # Download concurrently
    ok = 0
    fail = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = [
            ex.submit(download_one, sp, iid, args.out_dir, args.timeout, args.retries, args.force)
            for (sp, iid) in id_pairs
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            image_id, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                failed.append((image_id, msg))
            if i % 50 == 0 or i == total:
                print(f"  progress: {i}/{total}  ok={ok}  fail={fail}", flush=True)

    # Summary
    print(f"[done] downloaded={ok}  failed={fail}  saved_to={args.out_dir}")
    if failed:
        fail_log = Path(args.out_dir) / "_failed.txt"
        with open(fail_log, "w") as f:
            for iid, msg in failed:
                f.write(f"{iid}\t{msg}\n")
        print(f"[note] wrote failed list: {fail_log}")

if __name__ == "__main__":
    main()
