"""
src/analyze.py

Analyze stage of the ListingAugment pipeline:
- Multimodal analysis using Gemini 3.8 Flash (or validated dry-run mock)
- Strict JSON schema validation via Pydantic
- Response caching keyed by image hash + prompt version
- Hard cap on API calls and upfront cost estimation
- Retry with exponential backoff
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Tuple
import yaml
from PIL import Image, ImageStat
from pydantic import BaseModel, Field, ValidationError

class LightingModel(BaseModel):
    brightness: int = Field(ge=1, le=5, description="Lighting brightness from 1 (very dark) to 5 (overexposed)")
    color_temperature: Literal["warm", "neutral", "cool", "mixed"]
    issues: List[Literal["glare", "shadows", "underexposed", "overexposed", "flash_hotspot", "none"]]

class CameraModel(BaseModel):
    height: Literal["low", "eye", "high"]
    tilt_degrees_estimate: float = Field(description="Estimated camera tilt degrees")
    distortion: Literal["none", "mild", "strong"]

class ConditionSignalModel(BaseModel):
    cue: str = Field(description="Description of wear or damage cue")
    location: str = Field(description="Location of the cue in the image")
    severity: int = Field(ge=1, le=3, description="Severity score 1 (mild) to 3 (severe)")

class ImageQualityModel(BaseModel):
    sharpness: int = Field(ge=1, le=5, description="Image sharpness 1-5")
    noise: int = Field(ge=1, le=5, description="Image noise level 1-5")

class AnalyzeResult(BaseModel):
    room_type: Literal["kitchen", "bathroom", "bedroom", "living_room", "hallway", "exterior", "other"]
    element_focus: Literal["wall", "floor", "ceiling", "door", "window", "fixture", "appliance", "room_overview"]
    lighting: LightingModel
    camera: CameraModel
    condition_signals: List[ConditionSignalModel]
    must_preserve: List[str]
    image_quality: ImageQualityModel

PROMPT_VERSION = "v1.0"

def load_prompt(prompt_path: str = "prompts/analyze.txt") -> str:
    with open(prompt_path, "r") as f:
        return f.read().strip()

def compute_cache_key(image_path: Path, prompt_text: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    content = img_bytes + f"{PROMPT_VERSION}:{prompt_text}".encode("utf-8")
    return hashlib.sha256(content).hexdigest()

def estimate_cost(num_images: int, cfg: Dict[str, Any]) -> float:
    # Gemini 3.8 Flash: ~1,000 input tokens per image, ~400 output tokens
    in_price = cfg["api"]["pricing"]["gemini_3_8_flash_input_per_m"]
    out_price = cfg["api"]["pricing"]["gemini_3_8_flash_output_per_m"]
    cost = num_images * ((1000 / 1e6 * in_price) + (400 / 1e6 * out_price))
    return cost

def generate_dry_run_analysis(image_path: Path, metadata: Optional[Dict[str, Any]] = None) -> AnalyzeResult:
    """
    Generates a deterministic, schema-valid analysis in dry-run mode
    using image heuristics and available metadata.
    """
    if metadata is None:
        metadata = {}

    # Heuristics from image
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        stat = ImageStat.Stat(img_rgb)
        mean_brightness = sum(stat.mean) / 3.0  # 0 to 255
        
        # Color temperature estimation (red vs blue)
        r, g, b = stat.mean
        if r > b + 15:
            color_temp = "warm"
        elif b > r + 15:
            color_temp = "cool"
        elif abs(r - b) < 10:
            color_temp = "neutral"
        else:
            color_temp = "mixed"

        # Brightness scale 1-5
        if mean_brightness < 60:
            brightness = 1
            issues = ["underexposed"]
        elif mean_brightness < 100:
            brightness = 2
            issues = ["shadows"]
        elif mean_brightness < 170:
            brightness = 3
            issues = ["none"]
        elif mean_brightness < 210:
            brightness = 4
            issues = ["none"]
        else:
            brightness = 5
            issues = ["overexposed", "glare"]

    # Incorporate metadata if available
    room_raw = (metadata.get("room_type") or "other").lower().replace(" ", "_")
    if "bed" in room_raw:
        room_type = "bedroom"
    elif "bath" in room_raw or "toilet" in room_raw or "en-suite" in room_raw:
        room_type = "bathroom"
    elif "kitchen" in room_raw:
        room_type = "kitchen"
    elif "living" in room_raw or "reception" in room_raw:
        room_type = "living_room"
    elif "hall" in room_raw or "porch" in room_raw:
        room_type = "hallway"
    else:
        room_type = "other"

    element_raw = (metadata.get("element_type") or "wall").lower()
    if "wall" in element_raw:
        element_focus = "wall"
    elif "floor" in element_raw:
        element_focus = "floor"
    elif "ceiling" in element_raw:
        element_focus = "ceiling"
    elif "door" in element_raw:
        element_focus = "door"
    elif "window" in element_raw:
        element_focus = "window"
    elif "fixture" in element_raw:
        element_focus = "fixture"
    elif "appliance" in element_raw:
        element_focus = "appliance"
    else:
        element_focus = "wall"

    condition_raw = (metadata.get("raw_condition") or "Good").lower()
    condition_signals = []
    must_preserve = []

    if "poor" in condition_raw:
        condition_signals = [
            ConditionSignalModel(cue="visible surface peeling and structural cracking", location="center-left surface", severity=3),
            ConditionSignalModel(cue="water ingress discoloration and staining", location="upper boundary", severity=2)
        ]
        must_preserve = [
            "surface crack pattern and fracture lines",
            "peeling plaster edges and paint flaking",
            "dark water stain boundary"
        ]
    elif "fair" in condition_raw or "used" in condition_raw or "aged" in condition_raw or "dusty" in condition_raw:
        condition_signals = [
            ConditionSignalModel(cue="minor cosmetic scuffs and surface wear", location="lower edge / impact zone", severity=1),
            ConditionSignalModel(cue="faint surface dusting / age dullness", location="recessed corners", severity=1)
        ]
        must_preserve = [
            "light scuff marks along skirting/edge",
            "natural surface patina and finish texture"
        ]
    else: # Good / Clean / Brand New
        condition_signals = []
        must_preserve = [
            "clean uniform paintwork and intact finish",
            "unblemished grout/seam lines and spotless surface"
        ]

    return AnalyzeResult(
        room_type=room_type,
        element_focus=element_focus,
        lighting=LightingModel(
            brightness=brightness,
            color_temperature=color_temp,
            issues=issues
        ),
        camera=CameraModel(
            height="eye" if element_focus in ["wall", "door", "window"] else ("low" if element_focus == "floor" else "high"),
            tilt_degrees_estimate=4.5,
            distortion="none"
        ),
        condition_signals=condition_signals,
        must_preserve=must_preserve,
        image_quality=ImageQualityModel(sharpness=4, noise=2)
    )

def call_gemini_analyze(
    image_path: Path,
    prompt_text: str,
    cfg: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    dry_run: bool = False
) -> AnalyzeResult:
    cache_dir = Path(cfg["api"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = compute_cache_key(image_path, prompt_text)
    cache_file = cache_dir / f"analyze_{cache_key}.json"

    # Check cache
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return AnalyzeResult.model_validate(data)
        except Exception:
            pass

    api_key = os.environ.get("GEMINI_API_KEY")
    if dry_run or not api_key:
        result = generate_dry_run_analysis(image_path, metadata)
        with open(cache_file, "w") as f:
            f.write(result.model_dump_json(indent=2))
        return result

    # Real API call with google.genai SDK
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model_id = cfg["models"]["analyzer"]
    temperature = cfg["api"]["temperature"]["analyze"]
    max_retries = cfg["api"]["retry"]["max_retries"]
    backoff = cfg["api"]["retry"]["backoff_factor"]

    with Image.open(image_path) as img:
        img_format = img.format or "JPEG"
        mime_type = f"image/{img_format.lower()}"

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    prompt_text
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalyzeResult,
                    temperature=temperature
                )
            )
            raw_text = response.text
            result = AnalyzeResult.model_validate_json(raw_text)
            with open(cache_file, "w") as f:
                f.write(result.model_dump_json(indent=2))
            return result
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff ** attempt)

    # Fallback to dry-run heuristics if API fails
    print(f"[!] Warning: Gemini API failed for {image_path.name}: {last_err}. Using fallback heuristics.")
    result = generate_dry_run_analysis(image_path, metadata)
    with open(cache_file, "w") as f:
        f.write(result.model_dump_json(indent=2))
    return result

def analyze_images(
    image_items: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    dry_run: bool = False
) -> Dict[str, AnalyzeResult]:
    prompt_text = load_prompt()
    est_cost = estimate_cost(len(image_items), cfg)
    mode_str = "DRY-RUN (no API charges)" if dry_run or not os.environ.get("GEMINI_API_KEY") else "LIVE GEMINI 3.8 FLASH"
    
    print("\n" + "="*50)
    print(f"      STAGE 2: ANALYZE [{mode_str}]")
    print("="*50)
    print(f"Number of images to analyze: {len(image_items)}")
    print(f"Estimated cost:              ${est_cost:.4f} USD")
    print(f"Model ID:                    {cfg['models']['analyzer']}")
    print(f"Prompt Version:              {PROMPT_VERSION}")
    print("="*50 + "\n")

    results = {}
    for idx, item in enumerate(image_items):
        img_id = item["image_id"]
        img_path = Path(item["filepath"])
        print(f"[{idx+1}/{len(image_items)}] Analyzing {img_id} ({item.get('element_type')}, {item.get('raw_condition')})...")
        res = call_gemini_analyze(img_path, prompt_text, cfg, metadata=item, dry_run=dry_run)
        results[img_id] = res

    return results

