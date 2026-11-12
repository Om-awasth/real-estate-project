"""
src/plan.py

Plan stage of the ListingAugment pipeline:
- Generates 3 to 5 targeted augmentation recipes from the fixed menu
- Enforces condition preservation rules and lighting divergence
- Outputs structured JSON conforming strictly to Plan schema
- Supports Gemini 3.8 Flash and deterministic dry-run generation
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field

from src.analyze import AnalyzeResult

ALLOWED_LIGHTING = [
    "dim_evening_warm",
    "overcast_cool",
    "harsh_phone_flash",
    "backlit_window_glare",
    "mixed_tungsten_daylight",
    "low_light_noisy"
]

ALLOWED_CAMERA = [
    "tilt_small",       # 3-6 deg
    "tilt_large",       # 7-12 deg
    "low_angle",
    "high_angle",
    "wide_angle_phone",
    "off_center_crop"
]

ALLOWED_SENSOR = [
    "iso_noise",
    "jpeg_low_quality",
    "motion_blur_slight",
    "exposure_bias"
]

class RecipeModel(BaseModel):
    id: str = Field(description="Unique recipe identifier")
    lighting_preset: Optional[Literal[
        "dim_evening_warm", "overcast_cool", "harsh_phone_flash",
        "backlit_window_glare", "mixed_tungsten_daylight", "low_light_noisy"
    ]] = None
    camera_preset: Optional[Literal[
        "tilt_small", "tilt_large", "low_angle", "high_angle",
        "wide_angle_phone", "off_center_crop"
    ]] = None
    sensor_effects: List[Literal[
        "iso_noise", "jpeg_low_quality", "motion_blur_slight", "exposure_bias"
    ]] = Field(default_factory=list)
    method: Literal["classical", "generative", "both"]
    generative_edit_instruction: Optional[str] = None
    rationale: str

class PlanResult(BaseModel):
    recipes: List[RecipeModel] = Field(min_length=3, max_length=5)

PROMPT_VERSION = "v1.0"

def load_prompt(prompt_path: str = "prompts/plan.txt") -> str:
    with open(prompt_path, "r") as f:
        return f.read().strip()

def compute_cache_key(image_id: str, analyze_json: str, prompt_text: str) -> str:
    content = f"{image_id}:{analyze_json}:{PROMPT_VERSION}:{prompt_text}".encode("utf-8")
    return hashlib.sha256(content).hexdigest()

def generate_dry_run_plan(image_id: str, analysis: AnalyzeResult) -> PlanResult:
    """
    Produces diverse, rule-compliant augmentation recipes based on the analysis.
    Avoids lighting similar to the current lighting.
    """
    cur_brightness = analysis.lighting.brightness
    cur_temp = analysis.lighting.color_temperature

    # Select divergent lighting presets
    candidate_lighting = []
    if cur_brightness >= 3:
        candidate_lighting.extend(["dim_evening_warm", "low_light_noisy", "backlit_window_glare"])
    else:
        candidate_lighting.extend(["harsh_phone_flash", "overcast_cool", "mixed_tungsten_daylight"])

    if cur_temp == "warm":
        candidate_lighting.append("overcast_cool")
    elif cur_temp == "cool":
        candidate_lighting.append("dim_evening_warm")
    else:
        candidate_lighting.append("mixed_tungsten_daylight")

    # Deduplicate while preserving order
    chosen_lighting = []
    for l in candidate_lighting:
        if l not in chosen_lighting and l in ALLOWED_LIGHTING:
            chosen_lighting.append(l)

    preserve_summary = "; ".join(analysis.must_preserve) if analysis.must_preserve else "all surface textures and wear"

    recipes = [
        RecipeModel(
            id=f"{image_id}_recipe_1_classical_handheld",
            lighting_preset=chosen_lighting[0] if chosen_lighting else "dim_evening_warm",
            camera_preset="tilt_small",
            sensor_effects=["iso_noise", "jpeg_low_quality"],
            method="classical",
            generative_edit_instruction=None,
            rationale=f"Simulate amateur handheld phone capture with slight tilt and compression artifacts while strictly preserving: {preserve_summary}."
        ),
        RecipeModel(
            id=f"{image_id}_recipe_2_generative_relight",
            lighting_preset=chosen_lighting[1] if len(chosen_lighting) > 1 else "harsh_phone_flash",
            camera_preset=None,
            sensor_effects=[],
            method="generative",
            generative_edit_instruction=f"Change lighting to realistic {chosen_lighting[1] if len(chosen_lighting) > 1 else 'harsh_phone_flash'}. Keep perfectly identical: {preserve_summary}.",
            rationale=f"Test illumination robustness under realistic non-professional lighting shift without modifying geometric integrity."
        ),
        RecipeModel(
            id=f"{image_id}_recipe_3_hybrid_amateur",
            lighting_preset=chosen_lighting[2] if len(chosen_lighting) > 2 else "low_light_noisy",
            camera_preset="off_center_crop",
            sensor_effects=["motion_blur_slight", "exposure_bias"],
            method="both",
            generative_edit_instruction=f"Relight with {chosen_lighting[2] if len(chosen_lighting) > 2 else 'low_light_noisy'}. Strictly maintain: {preserve_summary}.",
            rationale=f"Stress-test against combined off-center handheld framing and non-uniform lighting."
        )
    ]

    return PlanResult(recipes=recipes)

def call_gemini_plan(
    image_id: str,
    analysis: AnalyzeResult,
    cfg: Dict[str, Any],
    dry_run: bool = False
) -> PlanResult:
    prompt_text = load_prompt()
    cache_dir = Path(cfg["api"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    analyze_json = analysis.model_dump_json()
    cache_key = compute_cache_key(image_id, analyze_json, prompt_text)
    cache_file = cache_dir / f"plan_{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return PlanResult.model_validate(data)
        except Exception:
            pass

    api_key = os.environ.get("GEMINI_API_KEY")
    if dry_run or not api_key:
        result = generate_dry_run_plan(image_id, analysis)
        with open(cache_file, "w") as f:
            f.write(result.model_dump_json(indent=2))
        return result

    # Live call to Gemini 3.8 Flash
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model_id = cfg["models"]["planner"]
    temperature = cfg["api"]["temperature"]["plan"]

    full_prompt = f"{prompt_text}\n\nInput Analyze JSON:\n{analyze_json}"
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PlanResult,
                temperature=temperature
            )
        )
        result = PlanResult.model_validate_json(response.text)
        with open(cache_file, "w") as f:
            f.write(result.model_dump_json(indent=2))
        return result
    except Exception as e:
        print(f"[!] Warning: Planner API call failed for {image_id}: {e}. Falling back to rule-based planner.")
        result = generate_dry_run_plan(image_id, analysis)
        with open(cache_file, "w") as f:
            f.write(result.model_dump_json(indent=2))
        return result

def plan_augmentations(
    analyses: Dict[str, AnalyzeResult],
    cfg: Dict[str, Any],
    dry_run: bool = False
) -> Dict[str, PlanResult]:
    print("\n" + "="*50)
    print("      STAGE 3: PLAN AUGMENTATION RECIPES")
    print("="*50)
    print(f"Generating recipe plans for {len(analyses)} images...")
    plans = {}
    for img_id, analysis in analyses.items():
        plans[img_id] = call_gemini_plan(img_id, analysis, cfg, dry_run=dry_run)
        print(f"[+] {img_id}: Planned {len(plans[img_id].recipes)} recipes.")
    return plans

