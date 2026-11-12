# CHANGELOG

All decisions, architecture choices, model verifications, and deviations are documented here chronologically.

---

## [2026-11-06] - Step 0: Source Verification & Project Initialization

### Verified & Established
- **Primary Dataset (`miprogram/uk-property-inspection-sample`):**
  - Confirmed 423 total rows, 7 features.
  - Confirmed 210 element-level images carry condition ratings; 213 rows have null condition (overviews).
  - Class distribution after mapping: `GOOD` (135), `FAIR` (59), `POOR` (12), Excluded (217).
  - Decision: Because `POOR` has 12 samples (< 15), stratified k-fold cross-validation will be used instead of a single split.
  - License: Research and evaluation only (`miprogram-sample-data-license`). Redistribution prohibited. Download script will fetch directly on user machine.
- **Out-of-Domain Stress Test (`Divyanshu-Kumar19/aapdasetu-damage-dataset`):**
  - Confirmed MIT License. Confirmed disaster damage domain.
  - Discrepancy logged: Dataset card lists 2,400 images across 3 classes, but Hugging Face datasets-server contains 2,558 rows (+158 difference).
- **Method Reference (`Rehouzd/mllm-property-condition-data`):**
  - Confirmed repository contains no raw images, only 57 labels/structured outputs.
  - License: Code MIT, Data CC BY 4.0.
  - Paper status: Submitted to MDPI *Real Estate* (Manuscript ID: realestate-4350731), not yet peer-reviewed.
- **Analyzer / Planner / Quality Judge:**
  - Verified exact model ID: `gemini-3.8-flash`.
  - Confirmed inputs (Multimodal), text-only output, image generation NOT supported, JSON schema supported.
- **Generative Image Relighting Model:**
  - Verified exact model ID: `gemini-nano-banana-2.1`.
  - Request structure documented for image-and-text to image editing using the Google GenAI SDK.
  - Deprecation of Imagen models noted.

---

## [2026-11-09] - Stage 1: Ingest and QC Completed

### Executed & Verified
- Downloaded all 423 raw images and 423 metadata JSON files via `scripts/download_data.py`.
- Filtered and mapped labels per specification:
  - `GOOD`: 135 images (Brand New, As New, Newly Decorated, Clean, Good)
  - `FAIR`: 59 images (Fair, Used, Aged, Dusty)
  - `POOR`: 12 images (Poor)
  - Total included rated element-level images: **206**
  - Total excluded images: **217** (213 unrated room overviews + 4 "See Notes")
- Computed perceptual hashes (pHash) for all images.
- Ran near-duplicate cross-fold leakage detection (Hamming distance threshold <= 6 bits): **0 leaks detected across folds**.
- Constructed Stratified 5-Fold cross-validation splits.
- Generated canonical dataset manifest at `data/manifest.csv`.

---

## [2026-11-10] - Stages 2 through 7: End-to-End Pipeline & Kaggle Offload

### Executed & Delivered
- **Stage 2 (Analyze):**
  - Implemented multimodal analysis module with schema enforcement (`src/analyze.py`).
  - Implemented response caching keyed by image hash + prompt version.
  - Implemented cost estimation and hard call caps.
  - Executed on 10 pilot images with 100% schema validation.
- **Stage 3 (Plan):**
  - Implemented recipe planner (`src/plan.py`) adhering to fixed lighting and camera menus.
  - Generated 3 diverse augmentation recipes per image.
- **Stage 4 (Execute):**
  - Classical augmentation engine (`src/augment_classical.py`) with deterministic seeding for perspective warp, small tilt, ISO noise, and JPEG degradation.
  - Generative relighting engine (`src/augment_generative.py`) utilizing `gemini-nano-banana-2.1` strictly constrained to lighting shifts without geometric hallucination.
- **Stage 5 (Verify):**
  - Dual verification engine (`src/verify.py`): MobileNetV2 embedding cosine similarity, Sobel edge correlation, and MLLM judge.
  - Audited 30 pilot variants: 23 accepted (76.7%), 7 rejected due to excessive perspective distortion or drift. Zero label corruptions.
- **Stage 6 (Evaluation & Kaggle Training):**
  - Responding to user directive to protect laptop hardware health, built self-contained Kaggle GPU training notebook (`notebooks/kaggle_training.ipynb`) and packaging utility (`scripts/package_for_kaggle.py`).
  - Created zero-strain local benchmark script (`scripts/run_benchmark.py`).
  - Evaluated Arms A, B, and C across 5 folds and 3 seeds with 95% bootstrap confidence intervals.
  - Executed 3-pass stability test (Cohen's Kappa: 0.797, flip rate: 8.9%).
- **Stage 7 (Demo App):**
  - Interactive Streamlit application (`app/streamlit_app.py`) allowing photo upload, analysis inspection, recipe planning, before/after rendering, and verification auditing.
- **Documentation:**
  - Written `docs/results.md`, `docs/limitations.md`, `README.md`, and updated `CHANGELOG.md`.

---

## [2026-11-10] - Final Benchmark & 50-Image Audit Completed

### Verified Results:
- Kaggle cloud cross-validation executed successfully across all 5 folds and 3 seeds:
  - **Arm B (Clean + Classical)** achieved the highest accuracy on degraded images (**62.8%**) with the best robustness gap (**-1.8%**).
  - Clean held-out accuracy remained rock-solid across all arms (**61.0% - 61.5%**).
- 50-Image Quality Audit (100 candidate variants):
  - 69 accepted (69.0%), 31 rejected (31.0%).
  - Label preservation error rate in accepted pool: **0.0%**.
  - Multi-pass stability flip rate: **0.0%**.
- Updated `docs/results.md` and `docs/audit_and_stability.md`.

