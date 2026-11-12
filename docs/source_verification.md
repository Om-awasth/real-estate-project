# Source Verification Report

**Date of Verification:** 2026-11-06  
**Project:** ListingAugment AI: Property Photo Augmentation Pipeline  
**Engineer:** Senior ML Engineer  

This document logs the pre-implementation verification of all data sources, reference repositories, and model APIs before writing project code, in accordance with Step 0 requirements.

---

## 1. Primary Dataset: miProgram UK Residential Property Inspection Sample
- **URL:** https://huggingface.co/datasets/miprogram/uk-property-inspection-sample
- **Role:** PRIMARY dataset for property condition assessment.
- **Verification Findings:**
  - **Row Count:** Verified **423** rows (Dataset card reports 423; Hugging Face `datasets-server` API confirms exactly 423 rows).
  - **Columns / Features:** Verified 7 features:
    1. `image` (Image feature)
    2. `image_id` (string)
    3. `room_type` (string)
    4. `element_type` (string)
    5. `condition_rating` (string)
    6. `property_archetype` (string)
    7. `stratum` (string)
  - **License:** `miprogram-sample-data-license` (`license: other`). Explicitly states: *"This sample dataset is provided by Mobile Inventory Program Limited for evaluation and research purposes only. Commercial use... requires a separate written licence agreement... Redistribution of this sample, in whole or in part, is not permitted without written consent."*
  - **Condition Ratings Count:** Verified that exactly **210** images carry a non-null `condition_rating`, and **213** images are null (primarily room overviews and unrated stratum items).
  - **Rating Breakdown:**
    - `Good`: 85
    - `Fair`: 35
    - `Brand New`: 14
    - `As New`: 13
    - `Poor`: 12
    - `Clean`: 12
    - `Newly Decorated`: 11
    - `Aged`: 11
    - `Dusty`: 7
    - `Used`: 6
    - `See Notes`: 4
    - `null`: 213
    - **Mapped Counts (Configured Mapping):**
      - `GOOD` (Brand New + As New + Newly Decorated + Clean + Good) = 14 + 13 + 11 + 12 + 85 = **135**
      - `FAIR` (Fair + Used + Aged + Dusty) = 35 + 6 + 11 + 7 = **59**
      - `POOR` (Poor) = **12**
      - `Excluded` (null: 213 + "See Notes": 4) = **217**
      - *Note on class balance:* `POOR` contains only 12 samples (< 15). Stratified k-fold cross-validation is mandatory per policy.
  - **TAXONOMY.md:** Read and verified. Outlines dropdown-controlled taxonomy including standardized `Conditions`, `DefectTypes`, `CeilingFinishes`, `DoorFeatures`, `AlarmStatusTypes`, etc., ensuring clean consistency without free-text annotation drift.
- **Result:** **PASS**

---

## 2. Optional Out-of-Domain Stress Test: AapdaSetu Building Damage Dataset
- **URL:** https://huggingface.co/datasets/Divyanshu-Kumar19/aapdasetu-damage-dataset
- **Role:** OPTIONAL out-of-domain stress test only. Never mixed into main condition labels.
- **Verification Findings:**
  - **License:** Verified **MIT License** (`license: mit`).
  - **Class Counts & Discrepancy Check:**
    - Dataset card table documents: `MINOR` (846) + `MAJOR` (525) + `DESTROYED` (1,029) = **2,400** images.
    - Hugging Face datasets-server API reports **2,558** total rows.
    - **Discrepancy:** Confirmed a discrepancy of **+158 images** between the dataset card documentation (2,400) and the HF repository viewer/server (2,558).
  - **Domain Confirmation:** Confirmed ground-level disaster-affected building photography (earthquake, flood, structural damage, partial roof/wall collapse, rubble). This represents extreme structural disaster damage rather than domestic tenancy/element wear.
- **Result:** **PASS** (Discrepancy documented; strictly isolated from primary condition pipeline).

---

## 3. Method Reference: MLLM Property Condition Data (Rehouzd)
- **URL:** https://github.com/Rehouzd/mllm-property-condition-data
- **Role:** METHOD REFERENCE only (multi-pass stability testing, intra-model agreement kappa, tier-flip rate calculation). Not a training source.
- **Verification Findings:**
  - **Raw Images:** Confirmed **NO raw images** exist in repository. It contains de-identified condition labels for 57 photos (labelers A & B), Gemini model structured outputs (`gemini-2.5-pro`), 5-pass stability test JSONs (`data/stability/pass_1.json` through `pass_5.json`), and analysis scripts.
  - **License:** Dual licensed:
    - Code in `scripts/`: **MIT License**
    - Data in `data/` and figures in `figures/`: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  - **Publication Status:** Paper is submitted and under review ("*Submitted to Real Estate (MDPI). Manuscript ID: realestate-4350731*"), not peer-reviewed.
- **Result:** **PASS**

---

## 4. Multi-Modal LLM: Gemini 3.8 Flash
- **URL:** https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
- **Role:** Pipeline Analyzer, Planner, and Quality Judge.
- **Verification Findings:**
  - **Exact Model ID:** `gemini-3.8-flash`
  - **Supported Inputs:** Text, Image, Video, Audio, PDF. Token input limit: **1,048,576** tokens.
  - **Supported Outputs:** **Text only**. Output token limit: **65,536** tokens.
  - **Image Generation / Editing:** **NOT SUPPORTED** (explicitly marked with `type="not-supported"` on documentation card). Under no circumstances will Gemini 3.8 Flash be called for image generation.
  - **Structured Output:** **SUPPORTED** (native JSON Schema / constrained output support).
  - **Thinking:** Supported (low, medium, high).
  - **Pricing / Limits:** Free tier available; Paid tier standard $0.75 / 1M input tokens, $3.75 / 1M output tokens (through Dec 31, 2026).
- **Result:** **PASS**

---

## 5. Generative Image Editing: Gemini Nano Banana 2.1
- **URL:** https://ai.google.dev/gemini-api/docs/changelog & https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1
- **Role:** Generative Relighting Stage.
- **Verification Findings:**
  - **Exact Model ID:** `gemini-nano-banana-2.1` (Gemini Nano Banana 2.1 GA).
  - **Family & Predecessors:** Part of the Nano Banana generative family. Preceded by `gemini-3.1-flash-image` (Nano Banana 2) and `gemini-3-pro-image` (Nano Banana Pro). Note that legacy Imagen models (`imagen-3.0`, `imagen-4.0`) are deprecated or shut down on the Gemini API; Nano Banana 2.1 is Google's active conversational image generation/editing model.
  - **Request Format for Image Editing:**
    - Via Google GenAI SDK (`google.genai`), using model `gemini-nano-banana-2.1`.
    - Input accepts multi-part list containing text instruction prompt and base64 encoded input image:
      ```python
      client = genai.Client()
      interaction = client.interactions.create(
          model="gemini-nano-banana-2.1",
          input=[
              {"type": "text", "text": generative_edit_prompt},
              {"type": "image", "data": base64_image_bytes, "mime_type": "image/jpeg"}
          ]
      )
      edited_image_bytes = base64.b64decode(interaction.output_image.data)
      ```
    - Supports fine-grained text-and-image-to-image editing, maintaining composition and surface features while altering lighting atmospheres.
- **Result:** **PASS**

---

## Verification Summary Table

| Source | Role | Checked Items | Status |
|---|---|---|---|
| **miProgram Sample** | Primary Dataset | 423 rows, 7 columns, Research-only license, 210 rated images (12 Poor), TAXONOMY.md verified | **PASS** |
| **AapdaSetu Dataset** | Optional Out-of-Domain | MIT license, Disaster damage confirmed, 158-image discrepancy (2,558 vs 2,400) reported | **PASS** |
| **Rehouzd Repository** | Method Reference | No raw images (57 labels/JSONs only), MIT + CC BY 4.0, Submitted MDPI manuscript | **PASS** |
| **Gemini 3.8 Flash** | Analyzer, Planner, Judge | Exact ID `gemini-3.8-flash`, Input Text/Image/Audio/Video/PDF, Output Text only (No image gen), JSON Schema supported | **PASS** |
| **Gemini Nano Banana 2.1** | Generative Relighter | Exact ID `gemini-nano-banana-2.1`, Image editing input format verified, Imagen deprecation noted | **PASS** |

