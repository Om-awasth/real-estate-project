"""
src/ingest.py

Ingestion and QC module:
- Loads metadata and raw image files
- Applies configurable condition label mapping
- Filters for rated element-level images
- Computes perceptual hashes (pHash)
- Performs near-duplicate detection and fold leakage audit
- Generates stratified k-fold cross-validation splits
- Produces data/manifest.csv and QC metrics
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from sklearn.model_selection import StratifiedKFold

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_image_phash(image_path: Path) -> Optional[str]:
    try:
        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception as e:
        print(f"[!] Warning: failed to compute phash for {image_path}: {e}")
        return None

def parse_metadata_and_images(raw_dir: Path) -> List[Dict[str, Any]]:
    images_dir = raw_dir / "images"
    meta_dir = raw_dir / "metadata"
    records = []

    # Iterate over available image files
    for img_path in sorted(images_dir.glob("*.jpg")):
        img_id = img_path.stem
        meta_file = meta_dir / f"{img_id}.json"
        
        meta = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[!] Error parsing metadata for {img_id}: {e}")

        # Extract taxonomy fields
        loc = meta.get("location_in_property", {})
        subj = meta.get("subject", {})
        labels = meta.get("labels", {})
        prop = meta.get("property", {})

        room_type = loc.get("room_type")
        element_type = subj.get("element_type")
        condition_rating = labels.get("condition_rating")
        archetype = prop.get("archetype")

        records.append({
            "image_id": img_id,
            "filename": img_path.name,
            "filepath": str(img_path),
            "room_type": room_type,
            "element_type": element_type,
            "raw_condition": condition_rating,
            "property_archetype": archetype,
        })
    return records

def map_condition_label(raw_condition: Optional[str], label_cfg: Dict[str, List[Any]]) -> Tuple[str, str]:
    """
    Returns (mapped_label, status_reason)
    """
    if raw_condition is None or raw_condition == "" or str(raw_condition).strip().lower() in ("null", "none"):
        return "EXCLUDED", "unrated_null"

    raw_clean = str(raw_condition).strip()

    for mapped_cat in ["GOOD", "FAIR", "POOR"]:
        allowed = label_cfg.get(mapped_cat, [])
        if raw_clean in allowed:
            return mapped_cat, "included"

    excluded_list = label_cfg.get("EXCLUDED", [])
    if raw_clean in excluded_list or raw_clean == "See Notes":
        return "EXCLUDED", f"excluded_label: {raw_clean}"

    return "EXCLUDED", f"unlisted_label: {raw_clean}"

def detect_near_duplicates(df: pd.DataFrame, threshold: int = 6) -> List[Dict[str, Any]]:
    """
    Finds pairs with pHash Hamming distance <= threshold.
    """
    hashes = []
    ids = []
    folds = []
    labels = []
    
    for _, row in df.iterrows():
        if pd.notna(row.get("phash")):
            hashes.append(imagehash.hex_to_hash(row["phash"]))
            ids.append(row["image_id"])
            folds.append(row.get("fold", -1))
            labels.append(row.get("condition_label", "EXCLUDED"))

    near_dupes = []
    n = len(hashes)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hashes[i] - hashes[j]
            if dist <= threshold:
                near_dupes.append({
                    "image_id_1": ids[i],
                    "image_id_2": ids[j],
                    "phash_dist": dist,
                    "fold_1": folds[i],
                    "fold_2": folds[j],
                    "label_1": labels[i],
                    "label_2": labels[j],
                    "cross_fold_leak": (folds[i] != folds[j]) and (folds[i] >= 0 and folds[j] >= 0)
                })
    return near_dupes

def build_manifest(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    label_cfg = cfg["label_mapping"]
    n_folds = cfg["dataset"].get("num_folds", 5)
    seed = cfg["dataset"].get("random_seed", 42)

    records = parse_metadata_and_images(raw_dir)
    if not records:
        raise ValueError(f"No images found in {raw_dir / 'images'}. Run scripts/download_data.py first.")

    df = pd.DataFrame(records)

    # Label mapping
    mapped_labels = []
    statuses = []
    for raw_cond in df["raw_condition"]:
        ml, stat = map_condition_label(raw_cond, label_cfg)
        mapped_labels.append(ml)
        statuses.append(stat)

    df["condition_label"] = mapped_labels
    df["status"] = statuses
    df["is_element_rated"] = df["condition_label"].isin(["GOOD", "FAIR", "POOR"])

    # Compute pHash for every image
    print("[*] Computing perceptual hashes (pHash)...")
    phashes = []
    for fp in df["filepath"]:
        phashes.append(compute_image_phash(Path(fp)))
    df["phash"] = phashes

    # Assign Stratified K-Folds for included rated element-level images
    df["fold"] = -1
    included_mask = df["is_element_rated"]
    included_indices = df[included_mask].index.to_numpy()
    included_labels = df.loc[included_mask, "condition_label"].to_numpy()

    n_samples = len(included_indices)
    if n_samples > 0:
        # Determine valid number of splits
        label_counts = pd.Series(included_labels).value_counts()
        min_class_count = label_counts.min()
        effective_folds = min(n_folds, n_samples)
        
        # StratifiedKFold requires at least 2 members per class for splits, otherwise fallback to simple k-fold
        if min_class_count >= 2 and effective_folds >= 2:
            splits_to_use = min(effective_folds, min_class_count)
            skf = StratifiedKFold(n_splits=splits_to_use, shuffle=True, random_state=seed)
            for fold_idx, (_, val_idx) in enumerate(skf.split(included_indices, included_labels)):
                actual_indices = included_indices[val_idx]
                df.loc[actual_indices, "fold"] = fold_idx
        elif effective_folds >= 2:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
            for fold_idx, (_, val_idx) in enumerate(kf.split(included_indices)):
                actual_indices = included_indices[val_idx]
                df.loc[actual_indices, "fold"] = fold_idx
        else:
            df.loc[included_indices, "fold"] = 0

    # Check for near duplicates & cross-fold leakage
    near_dupes = detect_near_duplicates(df[included_mask], threshold=6)
    cross_fold_leaks = [d for d in near_dupes if d["cross_fold_leak"]]

    # Save manifest
    manifest_path = Path(cfg["dataset"]["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"[+] Manifest successfully generated at {manifest_path} ({len(df)} total rows)")

    qc_summary = {
        "total_images": len(df),
        "total_rated_included": int(included_mask.sum()),
        "total_excluded": int((~included_mask).sum()),
        "class_counts": df[included_mask]["condition_label"].value_counts().to_dict(),
        "excluded_counts": df[~included_mask]["status"].value_counts().to_dict(),
        "num_folds": n_folds,
        "fold_distribution": {
            f"fold_{f}": df[df["fold"] == f]["condition_label"].value_counts().to_dict()
            for f in range(n_folds)
        },
        "near_duplicate_pairs_found": len(near_dupes),
        "cross_fold_leak_count": len(cross_fold_leaks),
        "cross_fold_leaks": cross_fold_leaks,
    }

    return df, qc_summary
