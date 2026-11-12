#!/usr/bin/env python3
"""
scripts/package_for_kaggle.py

Bundles project assets, code, manifest, and the Kaggle training notebook into
a self-contained zip package ready to upload to Kaggle Datasets / Notebooks.
"""

import sys
import zipfile
from pathlib import Path

def main():
    zip_filename = "kaggle_listingaugment_bundle.zip"
    files_to_include = [
        "config.yaml",
        "requirements.txt",
        "README.md",
        "CHANGELOG.md",
        "notebooks/kaggle_training.ipynb",
        "data/manifest.csv",
        "data/synthetic_stress/labels.csv",
    ]
    
    dirs_to_include = [
        "src",
        "prompts",
    ]

    print(f"[*] Packaging project bundle for Kaggle into {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add individual files
        for f in files_to_include:
            p = Path(f)
            if p.exists():
                zf.write(p, arcname=f)
                print(f"  + Added file: {f}")
            else:
                print(f"  - Skipped missing file: {f}")

        # Add directories
        for d in dirs_to_include:
            dp = Path(d)
            if dp.exists():
                for subfile in dp.rglob("*"):
                    if subfile.is_file() and not subfile.name.startswith("."):
                        zf.write(subfile, arcname=str(subfile))
                print(f"  + Added directory: {d}")

    print(f"[+] Kaggle bundle successfully created: {zip_filename} ({Path(zip_filename).stat().st_size / 1024:.1f} KB)")
    print("[+] Instructions for Kaggle:")
    print("    1. Open Kaggle Notebooks -> Create New Notebook")
    print("    2. Set Accelerator -> GPU T4 or P100")
    print("    3. Upload notebooks/kaggle_training.ipynb or unzip kaggle_listingaugment_bundle.zip")
    print("    4. Click 'Run All' to train without putting any workload on your laptop!")

if __name__ == "__main__":
    main()

