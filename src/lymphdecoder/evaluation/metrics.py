"""Evaluation metrics for imbalanced multi-class classification and feature stability."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Macro F1, MCC, balanced accuracy and accuracy."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|; two empty sets have similarity 1."""
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def stability(feature_sets: Sequence[Sequence]) -> float:
    """Mean pairwise Jaccard index of the feature sets selected in different folds."""
    sets = [set(s) for s in feature_sets]
    pairs = list(combinations(sets, 2))
    if not pairs:
        return float("nan")
    return float(np.mean([jaccard(a, b) for a, b in pairs]))
