import numpy as np
import pandas as pd
import pytest

from lymphdecoder import data
from lymphdecoder.features import dmr
from lymphdecoder.features.bin_store import BinStore
from lymphdecoder.features.regions import region_key

from conftest import synthetic_sample


def write_meta(path, rows, extra=None):
    df = pd.DataFrame(rows, columns=data.REQUIRED_COLUMNS)
    if extra is not None:
        df["exclude"] = extra
    df.to_csv(path, index=False)


def test_metadata_is_loaded_and_paths_resolved(tmp_path):
    rows = [["s1", "DLBCL/s1.tsv", "DLBCL", "rrbs", "GSE66329", "tissue"],
            ["s2", "Normal/s2.tsv", "Normal", "emseq", "GSE259312", "plasma"]]
    write_meta(tmp_path / "meta.csv", rows)
    meta = data.load_metadata(tmp_path / "meta.csv")
    assert list(meta.index) == ["s1", "s2"]
    assert meta.loc["s1", "file"].endswith("s1.tsv")
    assert str(tmp_path.resolve()) in meta.loc["s1", "file"]


def test_excluded_samples_are_removed(tmp_path):
    rows = [["s1", "a.tsv", "DLBCL", "rrbs", "d", "tissue"],
            ["mix1", "b.tsv", "Mixed", "rrbs", "d", "tissue"]]
    write_meta(tmp_path / "meta.csv", rows, extra=[False, True])
    assert list(data.load_metadata(tmp_path / "meta.csv").index) == ["s1"]


def test_metadata_errors_are_reported(tmp_path):
    rows = [["s1", "a.tsv", "DLBCL", "rrbs", "d", "tissue"],
            ["s1", "b.tsv", "FL", "rrbs", "d", "tissue"]]
    write_meta(tmp_path / "dup.csv", rows)
    with pytest.raises(ValueError, match="Duplicate"):
        data.load_metadata(tmp_path / "dup.csv")
    pd.DataFrame({"sample_id": ["s1"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Missing"):
        data.load_metadata(tmp_path / "bad.csv")


def test_small_classes_are_merged_but_normal_is_kept():
    labels = pd.Series(["DLBCL"] * 10 + ["MCL"] * 4 + ["CLL"] * 6 + ["Normal"] * 3)
    merged = data.merge_small_classes(labels, min_size=8)
    assert merged.value_counts().to_dict() == {"DLBCL": 10, "Other lymphoma": 10, "Normal": 3}


def test_metadata_template_lists_files_by_class_folder(tmp_path):
    (tmp_path / "DLBCL").mkdir()
    (tmp_path / "DLBCL" / "x.tsv").write_text("")
    (tmp_path / "Normal").mkdir()
    (tmp_path / "Normal" / "y.tsv").write_text("")
    tpl = data.metadata_template(tmp_path)
    assert tpl[["sample_id", "class"]].values.tolist() == [["DLBCL/x", "DLBCL"], ["Normal/y", "Normal"]]


def test_store_saves_bin_means_and_builds_matrix(tmp_path):
    store = BinStore(tmp_path / "bins")
    df = pd.DataFrame({"chr": ["chr1"] * 3 + ["chr2"], "base": [1, 2, 1500, 10],
                       "coverage": [10, 10, 10, 10], "freqC": [10.0, 30.0, 50.0, 70.0]})
    store.add("DLBCL/s1", df)
    assert "DLBCL/s1" in store
    table = store.get("DLBCL/s1")
    assert table["n_cpgs"].tolist() == [2, 1, 1]
    keys = region_key(pd.Series(["chr1", "chr1", "chr3"]), pd.Series([1, 1001, 1]))
    X = store.matrix(["DLBCL/s1"], keys)
    assert X.iloc[0, 0] == pytest.approx(20.0)
    assert X.iloc[0, 1] == pytest.approx(50.0)
    assert np.isnan(X.iloc[0, 2])
    X2 = store.matrix(["DLBCL/s1"], keys, min_cpgs=2)
    assert np.isnan(X2.iloc[0, 1])


def test_store_build_skips_existing_samples(tmp_path):
    store = BinStore(tmp_path / "bins")
    calls = []

    def loader(sid):
        calls.append(sid)
        return pd.DataFrame({"chr": ["chr1"], "base": [1], "coverage": [10], "freqC": [5.0]})

    store.build(["a", "b"], loader)
    store.build(["a", "b", "c"], loader)
    assert calls == ["a", "b", "c"]


def test_store_dmrs_match_in_memory_dmrs(tmp_path):
    rng = np.random.default_rng(1)
    samples = {f"{c}{i}": synthetic_sample(c, rng) for c in "ABC" for i in range(3)}
    labels = {s: s[0] for s in samples}
    store = BinStore(tmp_path / "bins")
    for sid, df in samples.items():
        store.add(sid, df)
    from_store = dmr.dmrs_from_store(store, list(samples), pd.Series(labels))
    in_memory = dmr.identify_dmrs(samples, labels)
    assert len(from_store) == len(in_memory) == 15
    assert from_store["n_pairs"].tolist() == in_memory["n_pairs"].tolist()
