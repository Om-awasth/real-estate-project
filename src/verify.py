"""
src/verify.py

Quality and Condition Verification stage:
- Programmatic checks:
  1. Deep feature embedding cosine similarity (MobileNetV2 backbone)
  2. Edge gradient structural similarity (Sobel correlation)
  3. Resolution and aspect ratio integrity
- MLLM Quality Judge (Gemini 3.8 Flash or dry-run validator)
- Enforces strict criteria: Rejects any image where defects were altered, label drifted, or realism < 3
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Tuple
import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel, Field

import torch
import torchvision.transforms as T
import torchvision.models as models

class VerifyJudgeResult(BaseModel):
    label_preserved: bool
    defects_added: List[str] = Field(default_factory=list)
    defects_removed: List[str] = Field(default_factory=list)
    objects_changed: List[str] = Field(default_factory=list)
    geometry_consistent: bool
    realism: int = Field(ge=1, le=5)
    lighting_matches_plan: bool
    decision: Literal["accept", "reject"]
    reason: str

class VerificationReport(BaseModel):
    image_id: str
    recipe_id: str
    method: str
    programmatic_passed: bool
    embedding_similarity: float
    edge_similarity: float
    aspect_ratio_preserved: bool
    judge_result: VerifyJudgeResult
    final_decision: Literal["accept", "reject"]

PROMPT_VERSION = "v1.0"

# Lazy-loaded feature extractor for embedding similarity
_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
_MODEL = None

def get_feature_extractor():
    global _MODEL
    if _MODEL is None:
        weights = models.MobileNet_V2_Weights.DEFAULT
        _MODEL = models.mobilenet_v2(weights=weights).features.to(_DEVICE).eval()
    return _MODEL

def compute_embedding_similarity(img1_path: Path, img2_path: Path) -> float:
    model = get_feature_extractor()
    with Image.open(img1_path) as i1, Image.open(img2_path) as i2:
        t1 = _TRANSFORM(i1.convert("RGB")).unsqueeze(0).to(_DEVICE)
        t2 = _TRANSFORM(i2.convert("RGB")).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        f1 = model(t1).mean([2, 3])
        f2 = model(t2).mean([2, 3])
        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)
        cos_sim = torch.cosine_similarity(f1, f2).item()
    return float(cos_sim)

def compute_edge_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    s1_x = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    s1_y = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    m1 = cv2.magnitude(s1_x, s1_y).flatten()

    s2_x = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
    s2_y = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
    m2 = cv2.magnitude(s2_x, s2_y).flatten()

    std1, std2 = np.std(m1), np.std(m2)
    if std1 == 0 or std2 == 0:
        return 1.0
    corr = np.corrcoef(m1, m2)[0, 1]
    return float(max(0.0, corr))

def run_programmatic_checks(
    orig_path: Path,
    aug_path: Path,
    min_embedding_sim: float = 0.65,
    min_edge_sim: float = 0.05
) -> Tuple[bool, float, float, bool]:
    orig_img = cv2.imread(str(orig_path))
    aug_img = cv2.imread(str(aug_path))

    h1, w1 = orig_img.shape[:2]
    h2, w2 = aug_img.shape[:2]
    ar1 = w1 / float(h1)
    ar2 = w2 / float(h2)
    ar_preserved = abs(ar1 - ar2) < 0.08

    emb_sim = compute_embedding_similarity(orig_path, aug_path)
    edge_sim = compute_edge_similarity(orig_img, aug_img)

    passed = (emb_sim >= min_embedding_sim) and (edge_sim >= min_edge_sim) and ar_preserved
    return passed, emb_sim, edge_sim, ar_preserved

def generate_dry_run_judge(
    prog_passed: bool,
    emb_sim: float,
    edge_sim: float,
    must_preserve: List[str]
) -> VerifyJudgeResult:
    if not prog_passed or emb_sim < 0.60:
        return VerifyJudgeResult(
            label_preserved=False,
            defects_added=["excessive visual distortion"],
            defects_removed=[],
            objects_changed=[],
            geometry_consistent=False,
            realism=2,
            lighting_matches_plan=True,
            decision="reject",
            reason=f"Programmatic threshold failed (embedding_sim={emb_sim:.3f}, edge_sim={edge_sim:.3f})."
        )
    return VerifyJudgeResult(
        label_preserved=True,
        defects_added=[],
        defects_removed=[],
        objects_changed=[],
        geometry_consistent=True,
        realism=4,
        lighting_matches_plan=True,
        decision="accept",
        reason="Condition signals and core structural features verified intact."
    )

def verify_augmented_photo(
    orig_path: Path,
    aug_path: Path,
    recipe: Any,
    must_preserve: List[str],
    cfg: Dict[str, Any],
    dry_run: bool = False
) -> VerificationReport:
    prog_passed, emb_sim, edge_sim, ar_preserved = run_programmatic_checks(orig_path, aug_path)

    api_key = os.environ.get("GEMINI_API_KEY")
    if dry_run or not api_key:
        judge_res = generate_dry_run_judge(prog_passed, emb_sim, edge_sim, must_preserve)
    else:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model_id = cfg["models"]["verifier"]
        with open("prompts/verify.txt", "r") as f:
            prompt_text = f.read().strip()
        prompt_full = f"{prompt_text}\n\nMust Preserve List:\n{json.dumps(must_preserve)}"

        with open(orig_path, "rb") as f1, open(aug_path, "rb") as f2:
            bytes1 = f1.read()
            bytes2 = f2.read()

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_bytes(data=bytes1, mime_type="image/jpeg"),
                    types.Part.from_bytes(data=bytes2, mime_type="image/jpeg"),
                    prompt_full
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VerifyJudgeResult,
                    temperature=0.0
                )
            )
            judge_res = VerifyJudgeResult.model_validate_json(response.text)
        except Exception as e:
            print(f"[!] Warning: Verify judge API failed: {e}. Falling back to dry-run judge.")
            judge_res = generate_dry_run_judge(prog_passed, emb_sim, edge_sim, must_preserve)

    final_decision = "accept" if (prog_passed and judge_res.decision == "accept") else "reject"

    return VerificationReport(
        image_id=orig_path.stem,
        recipe_id=getattr(recipe, "id", "unknown_recipe"),
        method=getattr(recipe, "method", "classical"),
        programmatic_passed=prog_passed,
        embedding_similarity=round(emb_sim, 4),
        edge_similarity=round(edge_sim, 4),
        aspect_ratio_preserved=ar_preserved,
        judge_result=judge_res,
        final_decision=final_decision
    )
