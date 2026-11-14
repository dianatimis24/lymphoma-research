"""Evaluation protocols: fixed split, (nested) cross-validation and the leaky variant.

In the leakage-free protocols, DMR identification, missing-value handling and
feature selection are fitted on the training samples of each fold only. The
leaky variant identifies DMRs and selects features on all samples before the
split; it exists only to measure how much this overestimates performance (E5).
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from lymphdecoder.features.dmr import make_dmr_finder
from lymphdecoder.features.regions import decode_keys, region_id
from lymphdecoder.features.selection import FeatureSelector
from lymphdecoder.models.classifiers import make_classifier

from .metrics import classification_metrics, stability

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All parameters of one run of the pipeline."""

    bin_size: int = 1000
    dmr_method: str = "naive"
    min_cpgs: int = 10
    min_sample_fraction: float = 0.5
    dmr_threshold: float = 25.0
    n_features: int = 50
    max_missing: float = 0.2
    variance_quantile: float = 0.2
    n_bins: int = 5
    criterion: str = "MIQ"
    classifier: str = "xgboost"
    classifier_params: dict = field(default_factory=dict)
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def _dmr_keys(store, ids, labels, cfg: PipelineConfig) -> np.ndarray:
    finder = make_dmr_finder(cfg.dmr_method, min_cpgs=cfg.min_cpgs,
                             min_sample_fraction=cfg.min_sample_fraction,
                             threshold=cfg.dmr_threshold)
    return finder.find(store, list(ids), labels).index.to_numpy()


def _fit_predict(X_train, y_train, X_test, cfg: PipelineConfig, X_select=None, y_select=None):
    """Fit selection (on X_select if given, else on X_train) and classifier; predict X_test."""
    selector = FeatureSelector(cfg.n_features, cfg.max_missing, cfg.variance_quantile,
                               cfg.n_bins, cfg.criterion)
    if X_select is None:
        selector.fit(X_train, y_train)
    else:
        selector.fit(X_select, y_select)
    model = make_classifier(cfg.classifier, cfg.seed, **cfg.classifier_params)
    model.fit(selector.transform(X_train), y_train)
    return model.predict(selector.transform(X_test)), selector.features_


def run_fold(store, labels: pd.Series, train_ids, test_ids, cfg: PipelineConfig,
             leaky: bool = False) -> dict:
    """Train on ``train_ids`` and predict ``test_ids``.

    With ``leaky=True`` DMRs and selected features are computed on all samples.
    """
    train_ids, test_ids = list(train_ids), list(test_ids)
    all_ids = train_ids + test_ids
    keys = _dmr_keys(store, all_ids if leaky else train_ids, labels, cfg)
    if len(keys) == 0:
        raise ValueError("No DMR found on the training samples")
    X_train = store.matrix(train_ids, keys)
    X_test = store.matrix(test_ids, keys)
    y_train = labels[train_ids].to_numpy()
    if leaky:
        X_all = pd.concat([X_train, X_test])
        pred, features = _fit_predict(X_train, y_train, X_test, cfg,
                                      X_all, labels[all_ids].to_numpy())
    else:
        pred, features = _fit_predict(X_train, y_train, X_test, cfg)
    return {"test_ids": test_ids, "y_true": labels[test_ids].to_numpy(), "y_pred": pred,
            "n_dmrs": len(keys), "features": features}


def make_splitter(labels: pd.Series, n_splits: int, n_repeats: int = 1, seed: int = 42):
    """Stratified splitter; the number of folds is reduced if a class is too small."""
    smallest = labels.value_counts().min()
    if smallest < 2:
        raise ValueError("Every class needs at least 2 samples; merge the smallest classes first")
    if smallest < n_splits:
        warnings.warn(f"Smallest class has {smallest} samples: using {smallest} folds "
                      f"instead of {n_splits}")
        n_splits = int(smallest)
    if n_repeats > 1:
        return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _inner_select(store, labels, train_ids, cfg, grid, inner_splits) -> PipelineConfig:
    """Choose the configuration from ``grid`` with the best inner-CV macro F1.

    DMRs and feature matrices are computed once per inner fold and reused for all
    candidates, because they do not depend on the grid parameters.
    """
    y = labels[train_ids]
    splitter = make_splitter(y, inner_splits, seed=cfg.seed)
    candidates = expand_grid(cfg, grid)
    scores = np.zeros(len(candidates))
    ids = np.asarray(train_ids)
    for tr, va in splitter.split(ids, y.to_numpy()):
        keys = _dmr_keys(store, ids[tr], labels, cfg)
        X_tr, X_va = store.matrix(ids[tr], keys), store.matrix(ids[va], keys)
        y_tr, y_va = labels[ids[tr]].to_numpy(), labels[ids[va]].to_numpy()
        for i, cand in enumerate(candidates):
            pred, _ = _fit_predict(X_tr, y_tr, X_va, cand)
            scores[i] += classification_metrics(y_va, pred)["macro_f1"]
    return candidates[int(np.argmax(scores))]


def expand_grid(cfg: PipelineConfig, grid: dict | None) -> list[PipelineConfig]:
    """All configurations obtained by combining the values in ``grid``.

    Keys are config fields; ``classifier_params`` values are dicts.
    """
    if not grid:
        return [cfg]
    keys = list(grid)
    return [replace(cfg, **dict(zip(keys, values)))
            for values in itertools.product(*(grid[k] for k in keys))]


def cross_validate(store, labels: pd.Series, cfg: PipelineConfig, n_splits: int = 5,
                   n_repeats: int = 1, grid: dict | None = None, inner_splits: int = 3,
                   leaky: bool = False) -> "CVResult":
    """Stratified cross-validation; with a ``grid`` it becomes nested cross-validation."""
    ids = labels.index.to_numpy()
    splitter = make_splitter(labels, n_splits, n_repeats, cfg.seed)
    folds = []
    for k, (tr, te) in enumerate(splitter.split(ids, labels.to_numpy())):
        fold_cfg = cfg
        if grid:
            fold_cfg = _inner_select(store, labels, ids[tr], cfg, grid, inner_splits)
        result = run_fold(store, labels, ids[tr], ids[te], fold_cfg, leaky=leaky)
        result["fold"] = k
        result["config"] = fold_cfg.to_dict()
        folds.append(result)
        logger.info("Fold %d: macro F1 %.3f", k,
                    classification_metrics(result["y_true"], result["y_pred"])["macro_f1"])
    return CVResult(folds, cfg.bin_size)


def fixed_split(store, labels: pd.Series, test_ids, cfg: PipelineConfig) -> "CVResult":
    """Single train/test split given by a list of test samples (E0)."""
    test_ids = [t for t in test_ids if t in labels.index]
    train_ids = [i for i in labels.index if i not in set(test_ids)]
    result = run_fold(store, labels, train_ids, test_ids, cfg)
    result["fold"] = 0
    result["config"] = cfg.to_dict()
    return CVResult([result], cfg.bin_size)


class CVResult:
    """Predictions and selected features of all folds, with summary metrics."""

    def __init__(self, folds: list[dict], bin_size: int):
        self.folds = folds
        self.bin_size = bin_size

    def predictions(self) -> pd.DataFrame:
        return pd.concat([pd.DataFrame({"sample_id": f["test_ids"], "fold": f["fold"],
                                        "y_true": f["y_true"], "y_pred": f["y_pred"]})
                          for f in self.folds], ignore_index=True)

    def fold_metrics(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            m = classification_metrics(f["y_true"], f["y_pred"])
            rows.append({"fold": f["fold"], "n_dmrs": f["n_dmrs"], **m})
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        fm = self.fold_metrics()
        out = {}
        for col in ["macro_f1", "mcc", "balanced_accuracy", "accuracy", "n_dmrs"]:
            out[f"{col}_mean"] = float(fm[col].mean())
            out[f"{col}_std"] = float(fm[col].std(ddof=1)) if len(fm) > 1 else 0.0
        out["feature_stability"] = stability([f["features"] for f in self.folds])
        return out

    def selected_features(self) -> pd.DataFrame:
        """Selected regions per fold, with readable ids (chr_start_end)."""
        rows = []
        for f in self.folds:
            regions = decode_keys(np.asarray(f["features"]), self.bin_size)
            for rid in region_id(regions):
                rows.append({"fold": f["fold"], "region": rid})
        return pd.DataFrame(rows)
