# Limitations and Operational Constraints

**Project:** ListingAugment AI  
**Document:** Pipeline and Data Limitations  
**Date:** 2026-11-10  

---

## 1. Small Sample Size & Severe Class Imbalance
- **Dataset Size:** The primary dataset (`miprogram/uk-property-inspection-sample`) provides 423 total inspection photos, but only **206** represent rated element-level subjects. The remaining 217 are room overviews or unrated elevations.
- **Minority Class Sparsity:**
  - `GOOD`: 135 images (65.5%)
  - `FAIR`: 59 images (28.6%)
  - `POOR`: 12 images (5.8%)
- **Impact:** Because the `POOR` class contains only 12 samples, training deep neural networks from scratch is impossible without extreme overfitting. Stratified 5-Fold cross-validation allocates only 2 to 3 `POOR` instances per validation fold. Confidence intervals for minority class recall remain broad.

---

## 2. Coarse Semantic Mapping & Label Subjectivity
- **Multi-Source Term Mapping:** The pipeline maps 10 granular survey terms into a 3-tier classification hierarchy:
  - `GOOD` merges: *Brand New, As New, Newly Decorated, Clean, Good*
  - `FAIR` merges: *Fair, Used, Aged, Dusty*
  - `POOR` maps: *Poor*
- **Subjective Overlap:** Distinctions between "Used" and "Fair" or "Clean" and "Good" reflect human surveyor subjectivity. When photos are augmented with dim lighting or dust noise, surveyor perception of "Aged" vs "Poor" can blur.

---

## 3. Potential Intra-Property Cross-Fold Leakage
- **Absence of Property Identifiers:** The miProgram sample metadata records property archetypes (e.g., Semi, Flat) and dates, but does **not** provide unique Property IDs.
- **Leakage Risk:** While our perceptual hashing check (`pHash` Hamming distance $\le 6$) successfully prevents duplicate or near-duplicate images from spanning folds (0 leaks found), photos of different rooms or elements within the same physical house may still reside across different cross-validation folds.

---

## 4. Evaluation and Research License Restrictions
- **Primary Data License:** The dataset is distributed under the `miprogram-sample-data-license` for **evaluation and research purposes only**.
- **Commercial Restrictions:** Redistribution of raw imagery is prohibited. Commercial deployment or fine-tuning commercial models requires a formal license from Mobile Inventory Program Limited.

---

## 5. Model Specialization Boundaries
- **Gemini 3.8 Flash:** Confirmed as an analyzer, planner, and verification judge only. It cannot generate or edit images.
- **Generative Relighting (`gemini-nano-banana-2.1`):** Generative editing is strictly constrained to lighting shifts (e.g., dim evening, flash hotspot). It must never be used for perspective or geometric changes, as diffusion/generative models hallucinate structural textures that alter condition defects.

