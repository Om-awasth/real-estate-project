"""
src/evaluate.py

Evaluation and Robustness Benchmarking module:
- Computes Accuracy, Macro-F1, per-class Recall
- Calculates Robustness Gap = Clean Score - Hard/Degraded Score
- Computes 95% Bootstrap Confidence Intervals (1,000 resamples)
- Evaluates multi-pass stability (intra-model agreement kappa and flip rate)
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score, recall_score, cohen_kappa_score

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "recall_good": float(per_class_rec[0]),
        "recall_fair": float(per_class_rec[1]),
        "recall_poor": float(per_class_rec[2])
    }

def compute_bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Returns (mean, ci_lower, ci_upper)
    """
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(random_seed)
    arr = np.array(values)
    means = []
    n = len(arr)
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(np.mean(sample))
    
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))
    return float(np.mean(arr)), lower, upper

def compute_stability_agreement(pass_results: List[List[str]]) -> Dict[str, float]:
    """
    Computes pairwise Cohen's Kappa and tier-flip rate across passes,
    following Rehouzd method reference.
    """
    n_passes = len(pass_results)
    if n_passes < 2:
        return {"mean_kappa": 1.0, "flip_rate": 0.0}

    kappas = []
    flips = 0
    total_pairs = 0
    n_samples = len(pass_results[0])

    for i in range(n_passes):
        for j in range(i + 1, n_passes):
            k = cohen_kappa_score(pass_results[i], pass_results[j])
            kappas.append(k)
            for s in range(n_samples):
                total_pairs += 1
                if pass_results[i][s] != pass_results[j][s]:
                    flips += 1

    return {
        "mean_kappa": float(np.mean(kappas)),
        "min_kappa": float(np.min(kappas)),
        "max_kappa": float(np.max(kappas)),
        "tier_flip_rate": float(flips / max(1, total_pairs))
    }

