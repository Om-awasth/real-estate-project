#!/usr/bin/env python3
"""
scripts/build_manifest.py

Builds data/manifest.csv, reports class distributions, logs excluded counts,
and executes near-duplicate cross-fold leakage detection.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import build_manifest

def main():
    parser = argparse.ArgumentParser(description="Build dataset manifest and run QC checks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    print(f"[*] Building manifest using config: {args.config}")
    df, qc = build_manifest(args.config)

    print("\n" + "="*50)
    print("           QC & INGESTION REPORT")
    print("="*50)
    print(f"Total raw images scanned: {qc['total_images']}")
    print(f"Included rated images:    {qc['total_rated_included']}")
    print(f"Excluded images:          {qc['total_excluded']}")
    print("\n[+] Final Rated Class Distribution:")
    for cls, cnt in sorted(qc["class_counts"].items()):
        print(f"  - {cls:<10}: {cnt:>4} images ({cnt/qc['total_rated_included']*100:.1f}%)")

    print("\n[-] Excluded Breakdown:")
    for reason, cnt in sorted(qc["excluded_counts"].items()):
        print(f"  - {reason:<25}: {cnt:>4} images")

    print(f"\n[+] Stratified {qc['num_folds']}-Fold Splits:")
    for fold, dist in qc["fold_distribution"].items():
        print(f"  - {fold}: {dist}")

    print(f"\n[+] Near-Duplicate Leakage Check (Hamming dist <= 6):")
    print(f"  - Total near-duplicate pairs: {qc['near_duplicate_pairs_found']}")
    print(f"  - Cross-fold leaks detected:   {qc['cross_fold_leak_count']}")
    if qc["cross_fold_leaks"]:
        print("  - Leaks details:")
        for leak in qc["cross_fold_leaks"]:
            print(f"    * {leak['image_id_1']} (Fold {leak['fold_1']}, {leak['label_1']}) <-> {leak['image_id_2']} (Fold {leak['fold_2']}, {leak['label_2']}) dist={leak['phash_dist']}")
    else:
        print("  - Zero cross-fold duplicates detected.")

    print("\n[!] Data Limitation Notice:")
    print("  The miProgram dataset does NOT provide unique Property IDs.")
    print("  While perceptual hashing prevents exact/near-duplicate image leakage across folds,")
    print("  different photos taken within the same residential property may still reside across folds.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

