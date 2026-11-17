import json

import numpy as np
import pandas as pd
import pytest
import yaml

from lymphdecoder.evaluation import exploration, plots
from lymphdecoder.experiments import Experiments, match_test_samples, run
from lymphdecoder.io import formats

from conftest import SIGNAL_BINS, synthetic_sample


def test_test_set_entries_are_matched_by_file_name():
    ids = ["GSE1/GSM1_cpg", "GSE2/Young1", "GSE2/Young2"]
    entries = ["GSM1_cpg.txt", "Young1.csv", " ", "missing.txt", "GSE2/Young2"]
    assert match_test_samples(entries, ids) == ["GSE1/GSM1_cpg", "GSE2/Young1", "GSE2/Young2"]


def test_flagged_samples_and_summary():
    summary = pd.DataFrame({"n_cpgs": [5000, 5000, 10, 5000],
                            "mean_methylation": [50, 51, 49, 50],
                            "fraction_0": [0.2, 0.2, 0.2, 0.2],
                            "fraction_100": [0.1, 0.1, 0.1, 0.1],
                            "median_coverage": [30, 30, 30, 30]},
                           index=["a", "b", "c", "d"])
    flags = exploration.flag_samples(summary, min_cpgs=1000)
    assert list(flags.index) == ["c"]
    assert flags.loc["c", "reason"] == "too_few_cpgs"


def test_low_coverage_sample_is_flagged_within_its_platform():
    normal = {f"CTRL{i}": (79.0 + 0.3 * (i % 4), 0.30 + 0.01 * (i % 5), 18.0 + i % 6)
              for i in range(1, 13)}
    normal["CTRL11"] = (83.0, 0.74, 2.0)
    rows = [{"sample_id": k, "platform": "wgbs", "n_cpgs": 26_000_000, "mean_methylation": m,
             "fraction_0": 0.08, "fraction_100": f, "median_coverage": c}
            for k, (m, f, c) in normal.items()]
    rows += [{"sample_id": f"D{i}", "platform": "rrbs", "n_cpgs": 2_800_000,
              "mean_methylation": 40.0 + i, "fraction_0": 0.27, "fraction_100": 0.05 + 0.01 * i,
              "median_coverage": 70.0 + i} for i in range(8)]
    summary = pd.DataFrame(rows).set_index("sample_id")
    flags = exploration.flag_samples(summary)
    assert list(flags.index) == ["CTRL11"]
    assert "low_coverage" in flags.loc["CTRL11", "reason"]
    assert "outlier_fraction_100" in flags.loc["CTRL11", "reason"]


def test_plots_are_written(tmp_path):
    cm = pd.DataFrame([[5, 1], [0, 4]], index=["A", "B"], columns=["A", "B"])
    assert plots.plot_confusion(cm, tmp_path / "cm.png").stat().st_size > 0
    coords = np.random.default_rng(0).normal(size=(6, 2))
    groups = pd.Series(list("AABBCC"))
    assert plots.plot_embedding(coords, groups, tmp_path / "emb.png").stat().st_size > 0


@pytest.fixture
def project(tmp_path):
    """Processed samples of 3 classes on 2 platforms, annotations and configuration."""
    rng = np.random.default_rng(0)
    processed = tmp_path / "processed"
    rows = []
    for label in SIGNAL_BINS:
        for i in range(6):
            platform = "rrbs" if i % 2 == 0 else "wgbs"
            df = synthetic_sample(label, rng)
            sid = f"{platform.upper()}/{label}{i}"
            path = processed / f"{sid}.tsv"
            formats.write(formats.finalize(df), path)
            rows.append({"sample_id": sid, "file": f"{sid}.tsv", "class": label,
                         "platform": platform, "dataset": platform.upper(),
                         "sample_type": "tissue" if label != "C" else "plasma"})
    pd.DataFrame(rows).to_csv(processed / "samples.csv", index=False)
    (tmp_path / "islands.bed").write_text(
        "".join(f"chr1\t{b * 1000 + 100}\t{b * 1000 + 400}\tisland{b}\n" for b in range(40)))
    (tmp_path / "test_set.txt").write_text("A0.tsv\nB1.tsv\nC2.tsv\n")
    config = {
        "metadata": "processed/samples.csv", "cache": "cache", "results": "results",
        "regions": {"bins": {"type": "bins", "bin_size": 1000},
                    "islands": {"type": "bed", "file": "islands.bed", "min_cpgs": 3}},
        "pipeline": {"n_features": 8, "classifier": "rf",
                     "classifier_params": {"n_estimators": 50}},
        "cv": {"n_splits": 3},
        "e0": {"test_set": "test_set.txt"},
        "e2": {"regions": ["bins", "islands"]},
        "e3": {"top_variance": 30, "method": "pca", "min_sample_fraction": 0.8},
        "e4": {"strategies": [
            {"name": "DMR bins", "regions": "bins"},
            {"name": "DMR islands", "regions": "islands"},
            {"name": "All islands", "regions": "islands", "dmr_method": "all"},
        ]},
    }
    path = tmp_path / "experiments.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_all_experiments_run_and_save_results(project):
    out = run(project, ["all"])
    results = project.parent / "results"
    assert out["e0"]["macro_f1_mean"] > 0.6
    assert (results / "e0" / "confusion_matrix.png").exists()
    assert len(out["e1"]) == 18
    overlap = out["e2"]
    shared = overlap[(overlap["platform_a"] == "rrbs") & (overlap["platform_b"] == "wgbs")]
    assert len(shared) == 2
    assert (shared["shared"] == 40).all()
    ari = out["e3"].set_index("grouping")["ari"]
    assert ari["class"] > 0.9
    assert ari["class"] > ari["platform"]
    assert (results / "e3" / "embedding_platform.png").exists()
    e4 = out["e4"].set_index("strategy")
    assert e4.loc["DMR bins", "macro_f1_mean"] > 0.9
    assert e4.loc["DMR islands", "macro_f1_mean"] > 0.9
    assert (results / "e4" / "comparison.csv").exists()
    e5 = out["e5"]
    assert list(e5["protocol"]) == ["leakage_free", "leaky"]
    info = json.loads((results / "e5" / "leaky" / "run.json").read_text())
    assert "git_commit" in info["run"]


def test_classes_can_be_restricted(project):
    raw = yaml.safe_load(project.read_text())
    raw["classes"] = ["A", "B"]
    project.write_text(yaml.safe_dump(raw))
    exp = Experiments(project)
    assert set(exp.labels) == {"A", "B"}
    assert len(exp.meta) == 12


def test_unknown_experiment_and_region_set_are_reported(project):
    with pytest.raises(ValueError, match="Unknown experiment"):
        run(project, ["e9"])
    with pytest.raises(ValueError, match="Unknown region set"):
        Experiments(project).store("promoters")
