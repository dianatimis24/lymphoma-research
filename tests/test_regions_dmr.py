import numpy as np
import pandas as pd
import pytest

from lymphdecoder.features import dmr, regions


def make_sample(values_by_bin, n_cpgs=12, chrom="chr1", bin_size=1000):
    """One sample with ``n_cpgs`` CpGs in each bin, all with the given value."""
    rows = []
    for bin_index, value in values_by_bin.items():
        start = bin_index * bin_size + 1
        for k in range(n_cpgs):
            rows.append([chrom, start + 10 * k, 30, value])
    df = pd.DataFrame(rows, columns=["chr", "base", "coverage", "freqC"])
    return df


def test_bins_are_one_based_and_closed():
    df = pd.DataFrame({"chr": ["chr1"] * 3, "base": [1, 1000, 1001], "freqC": [0.0, 0.0, 0.0]})
    binned = regions.assign_bins(df, 1000)
    assert binned["start"].tolist() == [1, 1, 1001]
    assert binned["end"].tolist() == [1000, 1000, 2000]


def test_bin_means_apply_cpg_density_filter():
    df = pd.concat([make_sample({0: 40.0}, n_cpgs=10), make_sample({1: 80.0}, n_cpgs=9)])
    means = regions.bin_means(df, 1000, min_cpgs=10)
    assert means[["start", "mean", "n_cpgs"]].values.tolist() == [[1, 40.0, 10]]


def test_region_values_average_cpgs_and_leave_missing_as_nan():
    df = pd.DataFrame({"chr": ["chr1"] * 4, "base": [5, 10, 15, 500],
                       "freqC": [10.0, 20.0, 60.0, 90.0]})
    reg = pd.DataFrame({"chr": ["chr1", "chr1", "chr2"], "start": [1, 100, 1], "end": [15, 200, 50]})
    values = regions.region_values(df, reg)
    assert values["chr1_1_15"] == pytest.approx(30.0)
    assert np.isnan(values["chr1_100_200"])
    assert np.isnan(values["chr2_1_50"])


def test_feature_matrix_has_one_row_per_sample():
    reg = pd.DataFrame({"chr": ["chr1"], "start": [1], "end": [1000]})
    samples = {"s1": make_sample({0: 10.0}), "s2": make_sample({1: 50.0})}
    matrix = regions.feature_matrix(samples, reg)
    assert matrix.shape == (2, 1)
    assert matrix.loc["s1", "chr1_1_1000"] == pytest.approx(10.0)
    assert np.isnan(matrix.loc["s2", "chr1_1_1000"])


def test_class_means_give_equal_weight_to_samples():
    samples = {"a1": make_sample({0: 10.0}, n_cpgs=10),
               "a2": make_sample({0: 30.0}, n_cpgs=40)}
    labels = {"a1": "A", "a2": "A"}
    means = dmr.class_means(samples, labels)
    assert means.loc[("chr1", 1, 1000), "A"] == pytest.approx(20.0)


def test_class_means_require_bin_in_enough_samples():
    samples = {"a1": make_sample({0: 10.0, 1: 10.0}), "a2": make_sample({0: 10.0}),
               "a3": make_sample({0: 10.0})}
    labels = {s: "A" for s in samples}
    means = dmr.class_means(samples, labels, min_sample_fraction=0.5)
    assert ("chr1", 1, 1000) in means.index
    assert ("chr1", 1001, 2000) not in means.index


def test_dmrs_use_all_pairs_of_classes():
    samples = {
        "a": make_sample({0: 10.0, 1: 50.0, 2: 50.0}),
        "b": make_sample({0: 40.0, 1: 50.0, 2: 50.0}),
        "c": make_sample({0: 10.0, 1: 50.0, 2: 90.0}),
    }
    labels = {"a": "A", "b": "B", "c": "C"}
    result = dmr.identify_dmrs(samples, labels, threshold=25.0)
    assert result["start"].tolist() == [1, 2001]
    assert result["n_pairs"].tolist() == [2, 2]
    assert result["max_diff"].tolist() == pytest.approx([30.0, 40.0])


def test_threshold_is_inclusive():
    means = pd.DataFrame({"A": [0.0, 0.0], "B": [25.0, 24.9]},
                         index=pd.MultiIndex.from_tuples([("chr1", 1, 1000), ("chr1", 1001, 2000)],
                                                         names=["chr", "start", "end"]))
    result = dmr.naive_dmrs(means, 25.0)
    assert list(result.index) == [("chr1", 1, 1000)]


def test_bins_missing_in_one_class_are_not_dmrs():
    means = pd.DataFrame({"A": [0.0], "B": [np.nan]},
                         index=pd.MultiIndex.from_tuples([("chr1", 1, 1000)],
                                                         names=["chr", "start", "end"]))
    assert dmr.naive_dmrs(means, 25.0).empty
