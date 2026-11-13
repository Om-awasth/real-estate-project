"""
app/streamlit_app.py

ListingAugment AI: Interactive Property Photo Augmentation Demo
Upload a real-estate photo to view:
1. Automated condition & lighting analysis
2. Diverse augmentation recipes preserving wear/cleanliness signals
3. Generated classical and generative relit variants
4. Strict quality & label-preservation verification decisions with explicit physical reasons (e.g. dim light, scratches obscured, camera tilt)
"""

import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Ensure project root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import load_config
from src.analyze import call_gemini_analyze, load_prompt
from src.plan import call_gemini_plan
from src.augment_classical import augment_image_classical
from src.augment_generative import execute_generative_relighting
from src.verify import verify_augmented_photo

st.set_page_config(
    page_title="ListingAugment AI",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 ListingAugment AI: Property Photo Augmentation Pipeline")
st.markdown(
    "Automated property-condition assessment models require training on photos with diverse, "
    "realistic lighting and awkward angles—while strictly preserving actual defect and wear labels."
)

cfg = load_config("config.yaml")

# Sidebar configuration
st.sidebar.header("Pipeline Settings")
dry_run = st.sidebar.checkbox("Dry-Run Mode (No API Charges)", value=True)
api_key_set = bool(os.environ.get("GEMINI_API_KEY"))
st.sidebar.info(f"API Key Active: {'Yes (GEMINI_API_KEY)' if api_key_set else 'No (Dry-Run Mode)'}")
st.sidebar.markdown(f"**Analyzer Model:** `{cfg['models']['analyzer']}`")
st.sidebar.markdown(f"**Relighter Model:** `{cfg['models']['generative_relighter']}`")

# Sample selection or upload
st.sidebar.subheader("Input Selection")
mode = st.sidebar.radio("Choose Input Method", ["Select Sample from Dataset", "Upload Custom Photo"])

sample_dir = Path("data/raw/images")
sample_files = sorted(list(sample_dir.glob("*.jpg"))) if sample_dir.exists() else []

selected_image_path = None

if mode == "Select Sample from Dataset" and sample_files:
    sample_options = [f.name for f in sample_files[:15]]
    chosen = st.sidebar.selectbox("Choose Sample Image", sample_options)
    selected_image_path = sample_dir / chosen
elif mode == "Upload Custom Photo":
    uploaded = st.file_uploader("Upload an Element Property Photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        temp_dir = Path("data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.name
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        selected_image_path = temp_path

if selected_image_path and selected_image_path.exists():
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📷 Original Input Photo")
        img_display = Image.open(selected_image_path)
        st.image(img_display, caption=f"Selected: {selected_image_path.name}", use_container_width=True)

    with col2:
        st.subheader("🔍 Stage 2: Multimodal Analysis")
        with st.spinner("Analyzing property photo cues..."):
            prompt_text = load_prompt()
            analysis = call_gemini_analyze(
                selected_image_path,
                prompt_text,
                cfg,
                metadata={"room_type": "interior", "element_type": "surface", "raw_condition": "Good"},
                dry_run=dry_run
            )

        st.json({
            "Room Type": analysis.room_type,
            "Element Focus": analysis.element_focus,
            "Lighting": analysis.lighting.model_dump(),
            "Camera Geometry": analysis.camera.model_dump(),
            "Condition Wear Signals": [c.model_dump() for c in analysis.condition_signals],
            "Must Preserve (Defects & Finish)": analysis.must_preserve
        })

    st.markdown("---")
    st.subheader("📋 Stage 3: Planned Augmentation Recipes")
    plan = call_gemini_plan(selected_image_path.stem, analysis, cfg, dry_run=dry_run)

    cols = st.columns(len(plan.recipes))
    for i, r in enumerate(plan.recipes):
        with cols[i]:
            st.markdown(f"**Recipe {i+1}: {r.id.split('_')[-1].capitalize()}**")
            st.caption(f"Method: `{r.method}`")
            st.write(f"💡 **Lighting:** `{r.lighting_preset}`")
            st.write(f"📐 **Camera:** `{r.camera_preset}`")
            st.write(f"⚡ **Sensor:** {r.sensor_effects}")
            st.write(f"📝 *{r.rationale}*")

    st.markdown("---")
    st.subheader("⚡ Stages 4 & 5: Augment & Verify Pipeline")

    aug_cols = st.columns(len(plan.recipes))
    out_dir = Path("data/demo_augmented")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, recipe in enumerate(plan.recipes):
        aug_save_path = out_dir / f"{recipe.id}.jpg"
        with st.spinner(f"Generating & Auditing Recipe {i+1}..."):
            if recipe.method == "classical":
                aug_img = augment_image_classical(
                    selected_image_path,
                    lighting_preset=recipe.lighting_preset,
                    camera_preset=recipe.camera_preset,
                    sensor_effects=recipe.sensor_effects,
                    seed=42
                )
            elif recipe.method == "generative":
                aug_img = execute_generative_relighting(
                    selected_image_path,
                    lighting_preset=recipe.lighting_preset or "dim_evening_warm",
                    must_preserve=analysis.must_preserve,
                    cfg=cfg,
                    dry_run=dry_run
                )
            else: # both
                gen_img = execute_generative_relighting(
                    selected_image_path,
                    lighting_preset=recipe.lighting_preset or "low_light_noisy",
                    must_preserve=analysis.must_preserve,
                    cfg=cfg,
                    dry_run=dry_run
                )
                temp_p = out_dir / "tmp.jpg"
                cv2.imwrite(str(temp_p), gen_img)
                aug_img = augment_image_classical(
                    temp_p,
                    lighting_preset=None,
                    camera_preset=recipe.camera_preset,
                    sensor_effects=recipe.sensor_effects,
                    seed=42
                )
                if temp_p.exists():
                    temp_p.unlink()

            cv2.imwrite(str(aug_save_path), aug_img)

            # Verification Audit with Physical Diagnostics
            report = verify_augmented_photo(
                selected_image_path,
                aug_save_path,
                recipe,
                analysis.must_preserve,
                cfg,
                dry_run=dry_run
            )

        with aug_cols[i]:
            st.image(Image.open(aug_save_path), caption=f"Variant: {recipe.id}", use_container_width=True)
            
            # Decision Badge
            if report.final_decision == "accept":
                st.success(f"✅ **ACCEPTED** (Realism: {report.judge_result.realism}/5)")
                st.markdown(f"**Verification Reason:** {report.judge_result.reason}")
            else:
                st.error("❌ **REJECTED**")
                # Prominent concrete physical rejection reason
                st.markdown(f"**🚫 Reason for Rejection:**\n> *{report.judge_result.reason}*")

            # Physical Diagnostic Inspection Details
            with st.expander("🔍 Physical Inspection Diagnostics"):
                d = report.diagnostics
                st.markdown(f"- 💡 **Illumination Level:** `{d.mean_brightness}/255` {'(⚠️ Too Dim)' if d.is_dim_light else ('(⚠️ Flash Glare)' if d.is_glare_blowout else '(Balanced)'}")
                st.markdown(f"- 🔎 **Scratches & Cracks:** {'✅ Preserved' if d.scratches_preserved else '⚠️ Obscured / Softened'}")
                st.markdown(f"- 📐 **Geometry / Tilt:** `{d.tilt_severity.replace('_', ' ').capitalize()}`")
                st.markdown(f"- 🌫️ **Sharpness / Blur:** `{d.blur_score}` {'(⚠️ Motion Blur)' if d.is_excessively_blurred else '(Sharp)'}")
                st.markdown(f"- 🎯 **Semantic Similarity:** `{report.embedding_similarity:.3f}` (Threshold $\ge 0.65$)")
                st.markdown(f"- 📏 **Edge Correlation:** `{report.edge_similarity:.3f}`")
else:
    st.info("👈 Please select or upload a property element photo in the sidebar to begin.")
