#!/usr/bin/env python3
"""
scripts/download_data.py

Downloads the miProgram UK Residential Property Inspection Sample dataset
from Hugging Face (images and per-image metadata).
Adheres to the research-only license policy (images are not committed to git).
"""

import os
import sys
import json
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HF_REPO = "miprogram/uk-property-inspection-sample"
API_URL = f"https://huggingface.co/api/datasets/{HF_REPO}"
RAW_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

def get_file_list():
    req = urllib.request.Request(API_URL, headers={"User-Agent": "ListingAugment/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    siblings = data.get("siblings", [])
    
    images = [s["rfilename"] for s in siblings if s["rfilename"].startswith("images/") and s["rfilename"].endswith((".jpg", ".jpeg", ".png"))]
    metadata = [s["rfilename"] for s in siblings if s["rfilename"].startswith("metadata/") and s["rfilename"].endswith(".json")]
    return sorted(images), sorted(metadata)

def download_file(rel_path, dest_dir, retries=3):
    dest_path = dest_dir / rel_path
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return rel_path, "cached", dest_path.stat().st_size
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE_URL}/{rel_path}"
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ListingAugment/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                with open(dest_path, "wb") as f:
                    f.write(content)
                return rel_path, "downloaded", len(content)
        except Exception as e:
            if attempt == retries - 1:
                return rel_path, f"error: {str(e)}", 0

def main():
    parser = argparse.ArgumentParser(description="Download miProgram property inspection dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to download (e.g. 10 for staging test)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target directory for raw data")
    parser.add_argument("--workers", type=int, default=8, help="Number of download threads")
    args = parser.parse_args()

    dest_dir = Path(args.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Querying Hugging Face repository {HF_REPO}...")
    images, metadata = get_file_list()
    print(f"[*] Found {len(images)} images and {len(metadata)} metadata JSON files.")

    if args.limit:
        print(f"[*] Limiting download to first {args.limit} items.")
        images = images[:args.limit]
        # Match metadata to corresponding images
        image_stems = {Path(img).stem for img in images}
        metadata = [m for m in metadata if Path(m).stem in image_stems]

    files_to_download = images + metadata
    print(f"[*] Downloading {len(files_to_download)} files ({len(images)} images, {len(metadata)} metadata) to {dest_dir}...")

    downloaded = 0
    cached = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, f, dest_dir): f for f in files_to_download}
        for future in as_completed(futures):
            f_path, status, sz = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "cached":
                cached += 1
            else:
                print(f"[!] Failed {f_path}: {status}")
                errors += 1

    print(f"[+] Download complete: {downloaded} downloaded, {cached} cached, {errors} errors.")
    if errors > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

