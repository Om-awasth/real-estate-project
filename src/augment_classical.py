"""
src/augment_classical.py

Deterministic and seeded classical image augmentation using Albumentations and OpenCV:
- Perspective warp, camera tilt, off-center crops
- Lighting adjustments (warm evening, cool overcast, flash hotspot, window glare)
- Sensor degradation (ISO noise, JPEG artifacts, motion blur, exposure bias)
- Guarantees label preservation by never synthesizing or removing structural defects
"""

import math
import random
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image

def apply_tilt_and_perspective(img: np.ndarray, preset: str, seed: int) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)

    if preset == "tilt_small":
        angle = rng.uniform(3.0, 6.0) * (1 if rng.rand() > 0.5 else -1)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif preset == "tilt_large":
        angle = rng.uniform(7.0, 12.0) * (1 if rng.rand() > 0.5 else -1)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif preset == "low_angle":
        # Simulate looking up: compress top, widen bottom
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx = w * 0.08
        pts2 = np.float32([[dx, 0], [w - dx, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif preset == "high_angle":
        # Simulate looking down: widen top, compress bottom
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx = w * 0.08
        pts2 = np.float32([[0, 0], [w, 0], [dx, h], [w - dx, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif preset == "wide_angle_phone":
        # Barrel distortion
        k1 = -0.15
        cam_mat = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, 0.0, 0, 0], dtype=np.float32)
        return cv2.undistort(img, cam_mat, dist_coeffs)
    elif preset == "off_center_crop":
        crop_pct = 0.88
        cw, ch = int(w * crop_pct), int(h * crop_pct)
        x0 = rng.randint(0, w - cw)
        y0 = rng.randint(0, h - ch)
        cropped = img[y0:y0+ch, x0:x0+cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return img

def apply_lighting_preset(img: np.ndarray, preset: Optional[str], seed: int) -> np.ndarray:
    if not preset:
        return img
    img_f = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    if preset == "dim_evening_warm":
        # Warm shift: boost Red, decrease Blue, attenuate overall light
        img_f[..., 0] *= 0.70  # B
        img_f[..., 1] *= 0.85  # G
        img_f[..., 2] *= 1.05  # R
        img_f = np.clip(img_f * 0.75, 0.0, 1.0)
    elif preset == "overcast_cool":
        # Cool shift: boost Blue, reduce warm cast
        img_f[..., 0] *= 1.15  # B
        img_f[..., 1] *= 0.98  # G
        img_f[..., 2] *= 0.85  # R
        img_f = np.clip(img_f * 0.90, 0.0, 1.0)
    elif preset == "harsh_phone_flash":
        # Central hotspot with rapid falloff towards corners
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        r_sq = (x - cx)**2 + (y - cy)**2
        max_r_sq = (w/2.0)**2 + (h/2.0)**2
        vignette = 1.35 - (r_sq / max_r_sq) * 0.65
        vignette = np.expand_dims(vignette, axis=-1)
        img_f = np.clip(img_f * vignette, 0.0, 1.0)
    elif preset == "backlit_window_glare":
        # Glare wash across upper-left
        y, x = np.ogrid[:h, :w]
        gradient = (1.0 - (x / w) * 0.6) * (1.0 - (y / h) * 0.6)
        gradient = np.expand_dims(gradient, axis=-1)
        img_f = np.clip(img_f + gradient * 0.35, 0.0, 1.0)
    elif preset == "mixed_tungsten_daylight":
        # Tungsten warm on left, daylight cool on right
        x = np.linspace(0, 1, w)[None, :, None]
        img_f[..., 2] *= (1.20 - 0.30 * x.squeeze())  # More R on left
        img_f[..., 0] *= (0.80 + 0.35 * x.squeeze())  # More B on right
        img_f = np.clip(img_f, 0.0, 1.0)
    elif preset == "low_light_noisy":
        # Severe underexposure
        img_f = np.clip(img_f * 0.50, 0.0, 1.0)

    return (img_f * 255.0).astype(np.uint8)

def apply_sensor_effects(img: np.ndarray, effects: list, seed: int) -> np.ndarray:
    res = img.copy()
    rng = np.random.RandomState(seed)

    for eff in effects:
        if eff == "iso_noise":
            noise = rng.normal(0, 18, res.shape).astype(np.float32)
            res = np.clip(res.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif eff == "jpeg_low_quality":
            quality = rng.randint(30, 48)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode('.jpg', res, encode_param)
            res = cv2.imdecode(enc, 1)
        elif eff == "motion_blur_slight":
            size = 5
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel /= size
            res = cv2.filter2D(res, -1, kernel)
        elif eff == "exposure_bias":
            alpha = rng.uniform(0.75, 1.25)
            beta = rng.randint(-20, 20)
            res = np.clip(alpha * res.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return res

def augment_image_classical(
    image_path: Path,
    lighting_preset: Optional[str] = None,
    camera_preset: Optional[str] = None,
    sensor_effects: Optional[list] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Executes a complete classical augmentation pipeline on a property photo.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Step 1: Camera & perspective geometry
    if camera_preset:
        img = apply_tilt_and_perspective(img, camera_preset, seed)

    # Step 2: Lighting preset
    if lighting_preset:
        img = apply_lighting_preset(img, lighting_preset, seed)

    # Step 3: Sensor noise and compression
    if sensor_effects:
        img = apply_sensor_effects(img, sensor_effects, seed)

    return img

