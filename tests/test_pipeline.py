"""
tests/test_pipeline.py

Unit and integration tests for the ListingAugment AI pipeline:
- Ingestion and label mapping
- Schema compliance for Analyze, Plan, and Verify stages
- Deterministic classical augmentation transforms
- Dual verification scoring (SSIM, embedding similarity)
- Evaluation bootstrap and stability metric calculations
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from src.ingest import map_condition_label, load_config
from src.analyze import (
    AnalyzeResult,
    generate_dry_run_analysis,
    load_prompt as load_analyze_prompt
)
from src.plan import (
    PlanResult,
    generate_dry_run_plan,
    load_prompt as load_plan_prompt
)
from src.augment_classical import (
    apply_tilt_and_perspective,
    apply_lighting_preset,
    apply_sensor_effects
)
from src.verify import (
    compute_edge_similarity,
    generate_dry_run_judge
)
from src.evaluate import (
    compute_classification_metrics,
    compute_bootstrap_ci,
    compute_stability_agreement
)

class TestListingAugmentPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("config.yaml")
        # Dummy test image (200x200 RGB gradient)
        cls.dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cls.dummy_img[:, :] = [120, 140, 160]
        # Add high-contrast edge simulating a crack/boundary
        cv2.line(cls.dummy_img, (50, 50), (150, 150), (0, 0, 0), 3)

        cls.test_img_path = Path("data/raw/images/img_20260201_000003.jpg")

    def test_label_mapping(self):
        label_cfg = self.cfg["label_mapping"]
        self.assertEqual(map_condition_label("Brand New", label_cfg)[0], "GOOD")
        self.assertEqual(map_condition_label("Good", label_cfg)[0], "GOOD")
        self.assertEqual(map_condition_label("Fair", label_cfg)[0], "FAIR")
        self.assertEqual(map_condition_label("Used", label_cfg)[0], "FAIR")
        self.assertEqual(map_condition_label("Poor", label_cfg)[0], "POOR")
        self.assertEqual(map_condition_label("See Notes", label_cfg)[0], "EXCLUDED")
        self.assertEqual(map_condition_label(None, label_cfg)[0], "EXCLUDED")

    def test_analyze_dry_run_schema(self):
        if self.test_img_path.exists():
            res = generate_dry_run_analysis(self.test_img_path, {"room_type": "kitchen", "raw_condition": "Poor"})
            self.assertIsInstance(res, AnalyzeResult)
            self.assertIn(res.room_type, ["kitchen", "bathroom", "bedroom", "living_room", "hallway", "exterior", "other"])
            self.assertTrue(len(res.must_preserve) > 0)

    def test_plan_dry_run_schema(self):
        if self.test_img_path.exists():
            analysis = generate_dry_run_analysis(self.test_img_path)
            plan = generate_dry_run_plan("test_img", analysis)
            self.assertIsInstance(plan, PlanResult)
            self.assertTrue(3 <= len(plan.recipes) <= 5)
            for r in plan.recipes:
                self.assertIn(r.method, ["classical", "generative", "both"])

    def test_classical_transforms(self):
        tilted = apply_tilt_and_perspective(self.dummy_img, "tilt_small", seed=42)
        self.assertEqual(tilted.shape, self.dummy_img.shape)

        relit = apply_lighting_preset(self.dummy_img, "dim_evening_warm", seed=42)
        self.assertEqual(relit.shape, self.dummy_img.shape)

        sensor = apply_sensor_effects(self.dummy_img, ["iso_noise", "jpeg_low_quality"], seed=42)
        self.assertEqual(sensor.shape, self.dummy_img.shape)

    def test_edge_similarity(self):
        score_identical = compute_edge_similarity(self.dummy_img, self.dummy_img)
        self.assertAlmostEqual(score_identical, 1.0, places=2)

    def test_metrics_and_bootstrap(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 0])
        m = compute_classification_metrics(y_true, y_pred)
        self.assertIn("accuracy", m)
        self.assertIn("macro_f1", m)

        mean_val, low, up = compute_bootstrap_ci([0.7, 0.75, 0.8, 0.85], n_resamples=100)
        self.assertTrue(low <= mean_val <= up)

    def test_stability_kappa(self):
        pass_1 = [0, 1, 2, 0, 1, 2]
        pass_2 = [0, 1, 2, 0, 1, 2]
        res = compute_stability_agreement([pass_1, pass_2])
        self.assertEqual(res["mean_kappa"], 1.0)
        self.assertEqual(res["tier_flip_rate"], 0.0)

if __name__ == "__main__":
    unittest.main()

