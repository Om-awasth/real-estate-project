#!/usr/bin/env python3
"""
scripts/generate_synthetic_stress.py

Generates the secondary synthetic stress test set:
Degrades held-out images with realistic low-light, harsh angles, and sensor noise.
Always labeled 'synthetic' in evaluation reports.
"""

import sys
from pathlib import Path
import pandas as pd
import cv2

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augment_classical import augment_image_classical

def main():
    manifest_path = Path("data/manifest.csv")
    if not manifest_path.exists():
        print("[!] data/manifest.csv not found.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    included = df[df["is_element_rated"]]
    
    out_dir = Path("data/synthetic_stress")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating synthetic stress dataset for {len(included)} held-out images...")
    records = []

    for idx, row in included.iterrows():
        img_id = row["image_id"]
        orig_path = Path(row["filepath"])
        dest_path = out_dir / f"synthetic_stress_{img_id}.jpg"

        # Apply harsh real-world smartphone conditions
        # (tilt, underexposed evening, ISO noise and motion blur)
        stress_img = augment_image_classical(
            orig_path,
            lighting_preset="dim_evening_warm",
            camera_preset="tilt_large",
            sensor_effects=["iso_noise", "jpeg_low_quality", "motion_blur_slight"],
            seed=int(row["fold"]) * 100 + 42
        )
        cv2.imwrite(str(dest_path), stress_img)

        records.append({
            "image_id": f"synthetic_stress_{img_id}",
            "original_image_id": img_id,
            "filepath": str(dest_path),
            "condition_label": row["condition_label"],
            "fold": row["fold"],
            "dataset_type": "synthetic"
        })

    stress_df = pd.DataFrame(records)
    labels_csv = out_dir / "labels.csv"
    stress_df.to_csv(labels_csv, index=False)
    print(f"[+] Synthetic stress dataset ready: {len(stress_df)} images saved to {out_dir}")

if __name__ == "__main__":
    main()

