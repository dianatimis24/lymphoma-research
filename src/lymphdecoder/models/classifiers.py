"""Classifiers used in the experiments.

All classifiers take class imbalance into account through balanced class weights.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


class BalancedXGBoost(BaseEstimator, ClassifierMixin):
    """XGBoost with string labels and balanced sample weights."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 10,
                 learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        import xgboost as xgb
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        random_state=self.random_state, n_jobs=1)
        self.model_.fit(np.asarray(X), self.encoder_.transform(y),
                        sample_weight=compute_sample_weight("balanced", y))
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.model_.predict(np.asarray(X)))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))


def _l1_penalty() -> dict:
    """L1 penalty arguments for the installed scikit-learn version."""
    from sklearn import __version__
    major, minor = (int(x) for x in __version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return {"l1_ratio": 1.0}
    return {"penalty": "l1"}


def make_classifier(name: str, random_state: int = 42, **params):
    """Create a classifier by name: ``xgboost``, ``rf``, ``svm`` or ``lasso``."""
    if name == "xgboost":
        return BalancedXGBoost(random_state=random_state, **params)
    if name == "rf":
        defaults = {"n_estimators": 500, "class_weight": "balanced"}
        return RandomForestClassifier(random_state=random_state, **{**defaults, **params})
    if name == "svm":
        defaults = {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}
        return make_pipeline(StandardScaler(),
                             SVC(random_state=random_state, **{**defaults, **params}))
    if name == "lasso":
        defaults = {"solver": "saga", "C": 1.0, "class_weight": "balanced", "max_iter": 5000}
        defaults.update(_l1_penalty())
        return make_pipeline(StandardScaler(),
                             LogisticRegression(random_state=random_state, **{**defaults, **params}))
    raise ValueError(f"Unknown classifier {name!r}")


def fresh(model):
    """Unfitted copy of a model, used in every fold."""
    return clone(model)
