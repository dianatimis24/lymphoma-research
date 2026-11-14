import numpy as np
import pandas as pd

from lymphdecoder.preprocessing import qc


def sample(rows):
    return pd.DataFrame(rows, columns=["chr", "base", "coverage", "freqC"]).astype(
        {"coverage": "Int64"})


def test_overlapping_and_adjacent_intervals_are_merged():
    panel = pd.DataFrame({"chr": ["chr1", "chr1", "chr1", "chr2"],
                          "start": [100, 150, 200, 10],
                          "end": [180, 200, 250, 20]})
    merged = qc.merge_intervals(panel)
    assert merged.values.tolist() == [["chr1", 100, 250], ["chr2", 10, 20]]


def test_panel_uses_zero_based_half_open_coordinates():
    starts, ends = np.array([100]), np.array([110])
    positions = np.array([100, 101, 110, 111])
    assert qc.in_intervals(positions, starts, ends).tolist() == [False, True, True, False]


def test_filter_by_panel_keeps_only_covered_cpgs():
    panel = pd.DataFrame({"chr": ["chr1", "chr2"], "start": [1000, 0], "end": [2000, 50]})
    df = sample([["chr1", 999, 20, 10.0], ["chr1", 1001, 20, 20.0], ["chr1", 2000, 20, 30.0],
                 ["chr1", 2001, 20, 40.0], ["chr2", 50, 20, 50.0], ["chr3", 5, 20, 60.0]])
    out = qc.filter_by_panel(df, panel)
    assert list(zip(out["chr"], out["base"])) == [("chr1", 1001), ("chr1", 2000), ("chr2", 50)]


def test_read_panel_merges_intervals(tmp_path):
    path = tmp_path / "panel.bed"
    path.write_text("chr1\t100\t200\tx\nchr1\t150\t300\ty\n")
    panel = qc.read_panel(path)
    assert panel.values.tolist() == [["chr1", 100, 300]]


def test_coverage_filter_keeps_reads_at_threshold():
    df = sample([["chr1", 1, 9, 10.0], ["chr1", 2, 10, 20.0], ["chr1", 3, 30, 30.0]])
    out = qc.filter_by_coverage(df, 10)
    assert out["base"].tolist() == [2, 3]


def test_coverage_filter_keeps_array_samples():
    df = sample([["chr1", 1, pd.NA, 10.0], ["chr1", 2, pd.NA, 20.0]])
    assert len(qc.filter_by_coverage(df, 10)) == 2
