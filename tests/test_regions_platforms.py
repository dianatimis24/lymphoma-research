import gzip

import numpy as np
import pandas as pd
import pytest

from lymphdecoder.features import annotations, dmr, regions
from lymphdecoder.features.bin_store import BinStore, RegionStore

from conftest import synthetic_sample


def sparse_sample(label, rng, keep=2):
    """Array-like sample: only ``keep`` CpGs per bin of a synthetic sample."""
    df = synthetic_sample(label, rng)
    df = df.groupby(df["base"] // 1000).head(keep).reset_index(drop=True)
    df["coverage"] = pd.NA
    return df


def test_min_cpgs_can_depend_on_platform():
    spec = {"default": 10, "array_450k": 1}
    assert dmr.resolve_min_cpgs(spec, "array_450k") == 1
    assert dmr.resolve_min_cpgs(spec, "rrbs") == 10
    assert dmr.resolve_min_cpgs({"rrbs": 5}, "wgbs") == 10
    assert dmr.resolve_min_cpgs(7, "anything") == 7


def test_array_samples_need_a_lower_density_threshold(tmp_path):
    rng = np.random.default_rng(0)
    store = BinStore(tmp_path / "bins")
    labels, platforms = {}, {}
    for label in "AB":
        for i in range(4):
            sid = f"{label}{i}"
            store.add(sid, sparse_sample(label, rng))
            labels[sid], platforms[sid] = label, "array_450k"
    labels, platforms = pd.Series(labels), pd.Series(platforms)
    uniform = dmr.dmrs_from_store(store, labels.index, labels, min_cpgs=10, platforms=platforms)
    per_platform = dmr.dmrs_from_store(store, labels.index, labels,
                                       min_cpgs={"default": 10, "array_450k": 1},
                                       platforms=platforms)
    assert uniform.empty
    assert len(per_platform) == 10


def test_all_regions_finder_ignores_classes(synthetic):
    store, labels = synthetic
    finder = dmr.make_dmr_finder("all", min_cpgs=10, min_sample_fraction=0.5)
    found = finder.find(store, labels.index, labels)
    assert len(found) == 40


def test_region_stats_count_cpgs_and_skip_empty_regions():
    df = pd.DataFrame({"chr": ["chr1"] * 4 + ["chr2"], "base": [5, 10, 15, 500, 7],
                       "freqC": [10.0, 20.0, 60.0, 90.0, 40.0]})
    reg = pd.DataFrame({"chr": ["chr1", "chr1", "chr2", "chr3"], "start": [1, 100, 1, 1],
                        "end": [15, 200, 10, 10]})
    stats = regions.region_stats(df, reg)
    assert list(stats.index) == [0, 2]
    assert stats.loc[0, "mean"] == pytest.approx(30.0)
    assert stats["n_cpgs"].tolist() == [3, 1]


def write_gz(path, text):
    with gzip.open(path, "wt") as f:
        f.write(text)


def test_ucsc_cpg_islands_are_read_and_shifted(tmp_path):
    path = tmp_path / "cpgIslandExt.txt.gz"
    write_gz(path, "585\tchr1\t28735\t29737\tCpG: 111\t1002\t111\t731\t22.2\t73\t0.85\n"
                   "585\tchr1\t29000\t29900\tCpG: 50\t900\t50\t600\t10\t60\t0.8\n"
                   "586\tchr1_KI270706v1_random\t1\t500\tCpG: 3\t499\t3\t10\t1\t2\t0.1\n")
    islands = annotations.read_cpg_islands(path)
    assert islands[["chr", "start", "end"]].values.tolist() == [["chr1", 28736, 29900]]


def test_bed_regions_are_read(tmp_path):
    path = tmp_path / "r.bed"
    path.write_text("chr2\t99\t200\tgeneB\nchr1\t0\t50\tgeneA\n")
    reg = annotations.read_bed_regions(path)
    assert reg.values.tolist() == [["chr1", 1, 50, "geneA"], ["chr2", 100, 200, "geneB"]]


def gene_row(name, chrom, strand, tx_start, tx_end, gene):
    return "\t".join(map(str, [0, name, chrom, strand, tx_start, tx_end, tx_start, tx_end, 1,
                               f"{tx_start},", f"{tx_end},", 0, gene, "cmpl", "cmpl", "0,"]))


def test_promoters_are_strand_aware_and_merged(tmp_path):
    path = tmp_path / "ncbiRefSeq.txt.gz"
    write_gz(path, "\n".join([
        gene_row("NM_1", "chr1", "+", 9999, 20000, "PLUS"),
        gene_row("NM_2", "chr1", "+", 10099, 20000, "PLUS"),
        gene_row("NM_3", "chr2", "-", 5000, 30000, "MINUS"),
    ]) + "\n")
    prom = annotations.read_promoters(path, upstream=1500, downstream=500)
    assert prom.values.tolist() == [["chr1", 8500, 10599, "PLUS"],
                                    ["chr2", 29501, 31500, "MINUS"]]


def test_region_store_uses_row_numbers_and_names(tmp_path):
    reg = pd.DataFrame({"chr": ["chr1", "chr1"], "start": [1, 1001], "end": [1000, 2000],
                        "name": ["a", "b"]})
    store = RegionStore(tmp_path / "islands", reg)
    df = pd.DataFrame({"chr": ["chr1"] * 3, "base": [5, 10, 1500], "coverage": [10] * 3,
                       "freqC": [10.0, 30.0, 80.0]})
    store.add("s1", df)
    X = store.matrix(["s1"], np.array([0, 1]))
    assert X.iloc[0].tolist() == pytest.approx([20.0, 80.0])
    assert store.region_names([1]) == ["chr1_1001_2000"]
    other = reg.assign(end=[900, 2000])
    with pytest.raises(ValueError, match="other regions"):
        RegionStore(tmp_path / "islands", other)
