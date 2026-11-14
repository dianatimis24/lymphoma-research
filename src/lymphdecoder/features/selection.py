"""Feature selection: missing-value handling, variance filter and mRMR.

Every step is fitted on the training samples only and then applied unchanged to
the test samples, so no information from the test set reaches the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def discretize(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Discretize each column into ``n_bins`` equal-frequency bins (codes 0..n_bins-1)."""
    X = np.asarray(X, dtype=float)
    codes = np.zeros(X.shape, dtype=np.int64)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for j in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:, j], qs))
        codes[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return codes


def mutual_information(a: np.ndarray, B: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Mutual information (in nats) between a discrete vector ``a`` (values 0..n_a-1)
    and every column of a discrete matrix ``B`` (values 0..n_b-1)."""
    n, p = B.shape
    codes = a[:, None] * n_b + B + (np.arange(p) * n_a * n_b)[None, :]
    joint = np.bincount(codes.ravel(), minlength=p * n_a * n_b).reshape(p, n_a, n_b) / n
    pa = joint.sum(axis=2, keepdims=True)
    pb = joint.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log(joint / (pa * pb))
    return np.nansum(terms, axis=(1, 2))


def mrmr(X: np.ndarray, y: np.ndarray, k: int, n_bins: int = 5,
         criterion: str = "MIQ") -> list[int]:
    """Minimum redundancy maximum relevance feature selection.

    Features are discretized into equal-frequency bins. The first selected feature
    is the one with the highest mutual information with the class. Each next
    feature maximizes ``relevance / mean redundancy`` (MIQ) or
    ``relevance - mean redundancy`` (MID), where redundancy is the mutual
    information with the already selected features. Returns column indices.
    """
    if criterion not in ("MIQ", "MID"):
        raise ValueError("criterion must be 'MIQ' or 'MID'")
    D = discretize(X, n_bins)
    classes, y_codes = np.unique(y, return_inverse=True)
    relevance = mutual_information(y_codes, D, len(classes), n_bins)
    k = min(k, D.shape[1])
    selected = [int(np.argmax(relevance))]
    redundancy_sum = np.zeros(D.shape[1])
    available = np.ones(D.shape[1], dtype=bool)
    available[selected[0]] = False
    while len(selected) < k:
        last = selected[-1]
        redundancy_sum += mutual_information(D[:, last], D, n_bins, n_bins)
        redundancy = redundancy_sum / len(selected)
        if criterion == "MIQ":
            score = relevance / (redundancy + 1e-12)
        else:
            score = relevance - redundancy
        score[~available] = -np.inf
        best = int(np.argmax(score))
        selected.append(best)
        available[best] = False
    return selected


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Training-only feature selection for a samples x features dataframe.

    1. features missing in more than ``max_missing`` of the training samples are removed;
    2. remaining missing values are imputed with the training median of the feature;
    3. the ``variance_quantile`` fraction of features with the lowest variance is removed;
    4. ``n_features`` features are chosen with mRMR.
    """

    def __init__(self, n_features: int = 50, max_missing: float = 0.2,
                 variance_quantile: float = 0.2, n_bins: int = 5, criterion: str = "MIQ"):
        self.n_features = n_features
        self.max_missing = max_missing
        self.variance_quantile = variance_quantile
        self.n_bins = n_bins
        self.criterion = criterion

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        missing = X.isna().mean()
        cols = missing.index[missing <= self.max_missing]
        if len(cols) == 0:
            raise ValueError("No feature has few enough missing values")
        self.medians_ = X[cols].median()
        filled = X[cols].fillna(self.medians_)
        variances = filled.var()
        cutoff = variances.quantile(self.variance_quantile)
        cols = variances.index[variances > cutoff] if (variances > cutoff).any() else variances.index
        chosen = mrmr(filled[cols].to_numpy(), np.asarray(y), self.n_features,
                      self.n_bins, self.criterion)
        self.features_ = list(cols[chosen])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).reindex(columns=self.features_)
        return X.fillna(self.medians_[self.features_])
