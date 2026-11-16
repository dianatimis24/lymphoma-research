from pathlib import Path

import pandas as pd
import pytest

from lymphdecoder.io import formats

DATA = Path(__file__).parent / "data"


def test_array_keeps_positions_and_percentages():
    df = formats.read_array(DATA / "array_example.tsv")
    assert list(df.columns) == formats.COLUMNS
    assert len(df) == 7
    assert df.loc[0, "base"] == 10525
    assert df.loc[0, "freqC"] == pytest.approx(97.6627)
    assert df["coverage"].isna().all()
    assert (df["strand"] == "*").all()


def test_array_fraction_scale_is_converted(tmp_path):
    path = tmp_path / "beta.tsv"
    pd.DataFrame({"chr": ["chr1", "chr1"], "base": [10525, 10848],
                  "beta": [0.976627, 0.5]}).to_csv(path, sep="\t", index=False)
    df = formats.read_array(path)
    assert df["freqC"].tolist() == pytest.approx([97.6627, 50.0])


def test_explicit_scale_overrides_detection():
    values = pd.Series([0.2, 0.9])
    assert formats.to_percent(values, "percent").tolist() == pytest.approx([0.2, 0.9])
    with pytest.raises(ValueError):
        formats.to_percent(values, "unknown")


def test_wgbs_bed_is_shifted_to_one_based():
    df = formats.read_wgbs_bed(DATA / "wgbs_example.bed")
    assert len(df) == 12
    row = df[df["base"] == 10525].iloc[0]
    assert row["coverage"] == 2
    assert row["freqC"] == pytest.approx(100.0)
    assert df[df["base"] == 10471]["freqC"].iloc[0] == pytest.approx(50.0)


def test_same_cpg_has_same_position_on_all_platforms():
    array = formats.read_array(DATA / "array_example.tsv")
    wgbs = formats.read_wgbs_bed(DATA / "wgbs_example.bed")
    rrbs = formats.read_rrbs_methylkit(DATA / "rrbs_example.tsv")
    for df in (array, wgbs, rrbs):
        assert 10525 in set(df["base"])


def test_rrbs_strands_are_merged_with_coverage_weights():
    df = formats.read_rrbs_methylkit(DATA / "rrbs_example.tsv")
    assert len(df) == 10
    row = df[df["base"] == 133180].iloc[0]
    assert row["coverage"] == 45 + 28
    expected = (45 * 93.333 + 28 * 96.429) / (45 + 28)
    assert row["freqC"] == pytest.approx(expected)
    assert 133181 not in set(df["base"])
    assert (df["strand"] == "*").all()


def test_rrbs_without_merging_keeps_reverse_positions():
    df = formats.read_rrbs_methylkit(DATA / "rrbs_example.tsv", merge_strands=False)
    assert len(df) == 11
    row = df[df["base"] == 133181].iloc[0]
    assert row["strand"] == "-"
    assert row["coverage"] == 28
    assert set(df["strand"]) == {"+", "-"}


def test_rows_are_sorted_by_chromosome_and_position(tmp_path):
    path = tmp_path / "unsorted.tsv"
    pd.DataFrame({"chr": ["chr10", "chr2", "chrX", "chr2", "chrUn_KI270302v1"],
                  "base": [5, 9, 1, 3, 7],
                  "freqC": [1, 2, 3, 4, 5]}).to_csv(path, sep="\t", index=False)
    df = formats.read_array(path, scale="percent")
    assert list(zip(df["chr"], df["base"])) == [("chr2", 3), ("chr2", 9), ("chr10", 5), ("chrX", 1)]


def test_split_matrix_creates_one_sample_per_column(tmp_path):
    path = tmp_path / "matrix.tsv"
    pd.DataFrame({"chrom": ["chr1", "chr1"], "position": [100, 200],
                  "S1": [0.1, 0.2], "S2": [0.3, None]}).to_csv(path, sep="\t", index=False)
    samples = formats.split_matrix(path, ["S1", "S2"], chr_col="chrom", base_col="position")
    assert set(samples) == {"S1", "S2"}
    assert samples["S1"]["freqC"].tolist() == pytest.approx([10.0, 20.0])
    assert len(samples["S2"]) == 1


def test_written_file_can_be_read_back(tmp_path):
    df = formats.read_rrbs_methylkit(DATA / "rrbs_example.tsv")
    out = tmp_path / "sample.tsv"
    formats.write(df, out)
    back = formats.read(out)
    pd.testing.assert_frame_equal(df, back)


def test_array_written_without_coverage_can_be_read_back(tmp_path):
    df = formats.read_array(DATA / "array_example.tsv")
    out = tmp_path / "sample.tsv"
    formats.convert(DATA / "array_example.tsv", "array", out)
    back = formats.read(out)
    assert back["coverage"].isna().all()
    assert back["base"].tolist() == df["base"].tolist()


def test_invalid_values_are_rejected():
    bad = pd.DataFrame({"chr": ["chr1"], "base": [0], "coverage": [pd.NA], "freqC": [50.0]})
    with pytest.raises(ValueError):
        formats.validate(bad)
    bad = pd.DataFrame({"chr": ["chr1"], "base": [5], "coverage": [pd.NA], "freqC": [150.0]})
    with pytest.raises(ValueError):
        formats.validate(bad)


def test_wgbs_strand_column_is_used_when_present(tmp_path):
    path = tmp_path / "s.bed"
    path.write_text("chr1\t10468\t10469\t2\t2\t+\nchr1\t10469\t10470\t1\t2\t-\n")
    df = formats.read_wgbs_bed(path)
    assert df[["base", "strand"]].values.tolist() == [[10469, "+"], [10470, "-"]]


def test_invalid_strand_is_rejected():
    bad = pd.DataFrame({"chr": ["chr1"], "base": [5], "strand": ["F"], "coverage": [pd.NA],
                        "freqC": [50.0]})
    with pytest.raises(ValueError, match="strand"):
        formats.validate(bad)


def test_rrbs_without_header_and_with_extra_columns():
    df = formats.read_rrbs_methylkit(DATA / "rrbs_noheader_example.txt")
    assert df["base"].tolist() == [10497, 10525, 10542]
    row = df[df["base"] == 10497].iloc[0]
    assert row["coverage"] == 117 + 143
    assert row["freqC"] == pytest.approx(117 * 82.051282051282 / 260)
    kept = formats.read_rrbs_methylkit(DATA / "rrbs_noheader_example.txt", merge_strands=False)
    assert len(kept) == 5
