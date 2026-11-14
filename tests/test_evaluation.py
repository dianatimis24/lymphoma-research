import json

import numpy as np
import pandas as pd
import pytest

from lymphdecoder.evaluation import cv, metrics
from lymphdecoder.models.classifiers import make_classifier
from lymphdecoder.utils import reproducibility


@pytest.mark.parametrize("name", ["xgboost", "rf", "svm", "lasso"])
def test_classifiers_work_with_string_labels(name):
    rng = np.random.default_rng(0)
    y = np.repeat(["DLBCL", "FL", "Normal"], 10)
    X = rng.normal(0, 1, (30, 4)) + np.repeat(np.eye(3, 4) * 5, 10, axis=0)
    model = make_classifier(name).fit(X, y)
    assert set(model.predict(X)) <= {"DLBCL", "FL", "Normal"}
    assert (model.predict(X) == y).mean() > 0.9


def test_unknown_classifier_is_rejected():
    with pytest.raises(ValueError):
        make_classifier("knn")


def test_jaccard_and_stability():
    assert metrics.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert metrics.stability([[1, 2], [1, 2], [1, 2]]) == pytest.approx(1.0)
    assert metrics.stability([[1, 2], [3, 4]]) == pytest.approx(0.0)


def test_metrics_of_perfect_prediction():
    m = metrics.classification_metrics(["a", "b", "b"], ["a", "b", "b"])
    assert m["macro_f1"] == m["mcc"] == m["accuracy"] == pytest.approx(1.0)


FAST = cv.PipelineConfig(n_features=10, classifier="rf",
                         classifier_params={"n_estimators": 50})


def test_cross_validation_separates_synthetic_classes(synthetic):
    store, labels = synthetic
    result = cv.cross_validate(store, labels, FAST, n_splits=4)
    summary = result.summary()
    assert summary["macro_f1_mean"] > 0.9
    assert summary["n_dmrs_mean"] == pytest.approx(15)
    assert len(result.predictions()) == len(labels)
    assert set(result.predictions()["sample_id"]) == set(labels.index)


def test_dmrs_are_computed_only_on_training_samples(synthetic, monkeypatch):
    store, labels = synthetic
    from lymphdecoder.features import dmr
    seen = []
    original = dmr.dmrs_from_store

    def spy(store_, ids, *args, **kwargs):
        ids = list(ids)
        seen.append(set(ids))
        return original(store_, ids, *args, **kwargs)

    monkeypatch.setattr(dmr, "dmrs_from_store", spy)
    ids = labels.index.to_numpy()
    train, test = ids[::2], ids[1::2]
    cv.run_fold(store, labels, train, test, FAST)
    assert seen[-1] == set(train)
    cv.run_fold(store, labels, train, test, FAST, leaky=True)
    assert seen[-1] == set(ids)


def test_folds_are_reduced_for_small_classes():
    labels = pd.Series(["A"] * 10 + ["B"] * 3)
    with pytest.warns(UserWarning, match="3 folds"):
        splitter = cv.make_splitter(labels, 5)
    assert splitter.get_n_splits() == 3
    with pytest.raises(ValueError):
        cv.make_splitter(pd.Series(["A"] * 10 + ["B"]), 5)


def test_nested_cross_validation_chooses_from_grid(synthetic):
    store, labels = synthetic
    grid = {"n_features": [5, 10]}
    result = cv.cross_validate(store, labels, FAST, n_splits=3, grid=grid, inner_splits=2)
    chosen = {f["config"]["n_features"] for f in result.folds}
    assert chosen <= {5, 10}
    assert result.summary()["macro_f1_mean"] > 0.9


def test_expand_grid_combines_values():
    configs = cv.expand_grid(FAST, {"n_features": [5, 10], "classifier": ["rf", "svm"]})
    assert len(configs) == 4
    assert cv.expand_grid(FAST, None) == [FAST]


def test_fixed_split_uses_given_test_samples(synthetic):
    store, labels = synthetic
    test_ids = ["A0", "B0", "C0", "unknown"]
    result = cv.fixed_split(store, labels, test_ids, FAST)
    assert sorted(result.predictions()["sample_id"]) == ["A0", "B0", "C0"]


def test_run_is_saved_with_config_and_commit(synthetic, tmp_path):
    store, labels = synthetic
    result = cv.cross_validate(store, labels, FAST, n_splits=2)
    folder = reproducibility.save_run(tmp_path / "run", FAST.to_dict(), result)
    info = json.loads((folder / "run.json").read_text())
    assert info["config"]["n_features"] == 10
    assert "git_commit" in info["run"]
    assert "macro_f1_mean" in info["summary"]
    assert (folder / "predictions.csv").exists()
    selected = pd.read_csv(folder / "selected_features.csv")
    assert selected["region"].str.startswith("chr1_").all()
