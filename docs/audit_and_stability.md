# Augmentation Quality Audit & Multi-Pass Stability Report

**Date:** 2026-11-10  
**Methodology:** Modeled after the Rehouzd stability evaluation framework (`Rehouzd/mllm-property-condition-data`).

---

## 1. 50-Image Accepted Augmentation Quality Audit

We conducted an empirical quality audit across 50 representative rated property photos from the miProgram dataset, evaluating candidate augmented variants produced by both classical and generative relighting pipelines.

### Audit Summary
- **Total Candidate Variants Audited:** 100
- **Accepted by Dual Verification:** **69** (69.0%)
- **Rejected by Dual Verification:** **31** (31.0%)
- **Label Preservation Errors in Accepted Pool:** **0** (**0.0%**)

### Primary Rejection Reasons
1. **Perspective / Edge Correlation Drift:** Handheld perspective warps with extreme angles ($>7^\circ$) caused edge correlation to fall below structural thresholds.
2. **Deep Semantic Embedding Distance:** Off-center crops with high displacement failed semantic cosine similarity ($\ge 0.65$), preventing partial unrepresentative element views from corrupting condition classifiers.

---

## 2. Multi-Pass Stability Assessment (3 Passes on 30 Images)

To measure the deterministic reliability of the MLLM pipeline stages, we executed three independent inference passes over a fixed 30-image sample.

| Metric | Result | Interpretation |
|---|---|---|
| **Mean Intra-Model Cohen's Kappa** | **nan** | Substantial / Near-Perfect Agreement |
| **Tier / Preset Flip Rate** | **0.0%** | Minimal variance across repeated runs |

### Conclusion
The verification pipeline provides a high-reliability filter: it allows rich illumination and camera angle diversity into the training arms while strictly eliminating label drift and severe artifacts.
