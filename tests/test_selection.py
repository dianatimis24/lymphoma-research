import numpy as np
import pandas as pd
import pytest

from lymphdecoder.features import selection


def test_discretize_gives_equal_frequency_codes():
    X = np.arange(100, dtype=float).reshape(-1, 1)
    codes = selection.discretize(X, 4)
    assert np.bincount(codes[:, 0]).tolist() == [25, 25, 25, 25]


def test_mutual_information_of_identical_variables_is_entropy():
    a = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    mi = selection.mutual_information(a, a[:, None], 4, 4)
    assert mi[0] == pytest.approx(np.log(4))


def test_mutual_information_of_independent_variables_is_zero():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    assert selection.mutual_information(a, b[:, None], 2, 2)[0] == pytest.approx(0.0)


def make_problem(rng, n=90):
    y = np.repeat(["A", "B", "C"], n // 3)
    informative1 = np.where(y == "A", 80, 20) + rng.normal(0, 3, n)
    duplicate = informative1 + rng.normal(0, 0.1, n)
    informative2 = np.where(y == "B", 80, 20) + rng.normal(0, 3, n)
    noise = rng.normal(50, 10, (n, 5))
    X = np.column_stack([informative1, duplicate, informative2, noise])
    return X, y


@pytest.mark.parametrize("criterion", ["MIQ", "MID"])
def test_mrmr_prefers_complementary_over_redundant_features(criterion):
    X, y = make_problem(np.random.default_rng(0))
    chosen = selection.mrmr(X, y, 2, criterion=criterion)
    assert set(chosen) & {0, 1}
    assert 2 in chosen


def test_feature_selector_uses_only_training_data():
    rng = np.random.default_rng(0)
    X, y = make_problem(rng)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X.iloc[0, 2] = np.nan
    sel = selection.FeatureSelector(n_features=3, variance_quantile=0.0).fit(X, y)
    assert len(sel.features_) == 3
    test = pd.DataFrame([[np.nan] * X.shape[1]], columns=X.columns)
    out = sel.transform(test)
    assert out.iloc[0].tolist() == pytest.approx(sel.medians_[sel.features_].tolist())


def test_feature_selector_drops_features_with_many_missing_values():
    rng = np.random.default_rng(0)
    X, y = make_problem(rng)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X.loc[X.index[:40], "f2"] = np.nan
    sel = selection.FeatureSelector(n_features=5, max_missing=0.2, variance_quantile=0.0).fit(X, y)
    assert "f2" not in sel.features_


def test_variance_filter_removes_constant_features():
    rng = np.random.default_rng(0)
    X, y = make_problem(rng)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X["const"] = 5.0
    sel = selection.FeatureSelector(n_features=8, variance_quantile=0.2).fit(X, y)
    assert "const" not in sel.features_
