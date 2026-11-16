import gzip
import time

import numpy as np
import pandas as pd
import pytest
import yaml

from lymphdecoder.io import formats
from lymphdecoder.preprocessing.liftover import Chain, lift, parse_chain
from lymphdecoder.processing import ProcessingConfig, process

CHAIN = """chain 1000 chr1 249250621 + 0 300 chr1 248956422 + 100 420 1
100 20 0
200

chain 50 chr1 249250621 + 50 80 chr1 248956422 + 5000 5030 2
30

chain 900 chr2 243199373 + 0 100 chr2 242193529 - 0 100 3
100

chain 800 chr3 198022430 + 0 100 chr5 181538259 + 0 100 4
100
"""


def write_chain(path):
    with gzip.open(path, "wt") as f:
        f.write(CHAIN)
    return path


def test_chain_blocks_are_parsed_with_gaps():
    blocks = parse_chain(CHAIN.splitlines())
    first = blocks[blocks["chain"] == 0]
    assert first[["start", "end", "offset"]].values.tolist() == [[0, 100, 100], [120, 320, 80]]
    assert set(blocks["chain"]) == {0, 1}


def test_positions_are_mapped_through_blocks():
    chain = Chain(parse_chain(CHAIN.splitlines()))
    mapped = chain.map_positions("chr1", np.array([1, 100, 101, 120, 121, 320, 321]))
    assert mapped.tolist() == [101, 200, -1, -1, 201, 400, -1]


def test_higher_scoring_chain_wins_where_chains_overlap():
    chain = Chain(parse_chain(CHAIN.splitlines()))
    assert chain.map_positions("chr1", np.array([60]))[0] == 160


def test_reverse_strand_and_other_chromosome_chains_are_ignored():
    chain = Chain(parse_chain(CHAIN.splitlines()))
    assert chain.map_positions("chr2", np.array([10]))[0] == -1
    assert chain.map_positions("chr3", np.array([10]))[0] == -1


def test_lift_drops_unmapped_cpgs_and_keeps_the_format():
    chain = Chain(parse_chain(CHAIN.splitlines()))
    df = formats.finalize(pd.DataFrame({"chr": ["chr1", "chr1", "chr2"], "base": [10, 110, 10],
                                        "coverage": [12, 15, 20], "freqC": [50.0, 60.0, 70.0]}))
    out = lift(df, chain)
    assert out[["chr", "base", "coverage", "freqC"]].values.tolist() == [["chr1", 110, 12, 50.0]]
    assert list(out.columns) == formats.COLUMNS


def test_processing_applies_liftover_from_the_configuration(tmp_path):
    write_chain(tmp_path / "hg19ToHg38.over.chain.gz")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "s1.tsv").write_text("chr\tbase\tfreqC\nchr1\t10\t40\nchr1\t150\t80\nchr1\t310\t20\n")
    config = {"output": "processed", "min_coverage": 1,
              "datasets": [{"name": "OLD", "reader": "array", "folder": "raw", "pattern": "*.tsv",
                            "class": "DLBCL", "platform": "array_450k", "sample_type": "tissue",
                            "liftover": "hg19ToHg38.over.chain.gz"}]}
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump(config))
    meta = process(ProcessingConfig.load(path))
    out = formats.read(tmp_path / "processed" / meta.loc[0, "file"])
    assert out["base"].tolist() == [110, 230, 390]


def test_chain_with_many_blocks_is_indexed_quickly():
    rng = np.random.default_rng(0)
    lines = []
    pos = 0
    for c in range(300):
        lines.append(f"chain {1000 - c} chr1 249250621 + {pos} {pos + 10**6} chr1 248956422 + {pos + 7} {pos + 10**6 + 7} {c}")
        for _ in range(999):
            lines.append("900 100 100")
        lines.append("900")
        pos += 10**6
    start = time.time()
    chain = Chain(parse_chain(lines))
    positions = rng.integers(1, pos, 1_000_000)
    chain.map_positions("chr1", positions)
    assert time.time() - start < 30
