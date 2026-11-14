"""Conversion of methylation files from different platforms to one common format.

Common format (one file per sample, tab-separated, hg38):

    chr       chromosome name, e.g. ``chr1``
    base      1-based position of the cytosine of the CpG on the forward strand
    coverage  number of reads covering the CpG (missing for arrays)
    freqC     methylation percentage, between 0 and 100

Rows are sorted by chromosome (chr1..chr22, chrX, chrY, chrM) and position.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS = ["chr", "base", "coverage", "freqC"]
CHROMOSOME_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
SCALES = ("auto", "fraction", "percent")


def to_percent(values: pd.Series, scale: str = "auto") -> pd.Series:
    """Bring methylation values to the 0-100 scale.

    ``scale`` is ``"fraction"`` for values in [0, 1], ``"percent"`` for values in
    [0, 100] and ``"auto"`` to decide from the data: a sample whose maximum is at
    most 1 is considered to be on the fraction scale.
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    values = values.astype(float)
    if scale == "auto":
        scale = "fraction" if values.max() <= 1.0 else "percent"
        logger.info("Detected %s scale", scale)
    return values * 100.0 if scale == "fraction" else values


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns and rows, set dtypes and validate a common-format dataframe."""
    df = df.dropna(subset=["chr", "base", "freqC"]).copy()
    if "coverage" not in df.columns:
        df["coverage"] = pd.NA
    df["base"] = df["base"].astype("int64")
    df["coverage"] = df["coverage"].astype("Int64")
    df["freqC"] = df["freqC"].astype(float)
    df = df[df["chr"].isin(CHROMOSOME_ORDER)]
    df["chr"] = pd.Categorical(df["chr"], categories=CHROMOSOME_ORDER, ordered=True)
    df = df.sort_values(["chr", "base"]).reset_index(drop=True)
    df["chr"] = df["chr"].astype(str)
    validate(df)
    return df[COLUMNS]


def validate(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if a dataframe does not respect the common format."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if (df["base"] < 1).any():
        raise ValueError("Positions must be 1-based (base >= 1)")
    if ((df["freqC"] < 0) | (df["freqC"] > 100)).any():
        raise ValueError("freqC must be between 0 and 100")
    if df.duplicated(["chr", "base"]).any():
        raise ValueError("Duplicate (chr, base) positions")


def read_array(path: str | Path, scale: str = "auto") -> pd.DataFrame:
    """Read one array sample (see ``readers.ArrayReader``)."""
    from .readers import ArrayReader
    return ArrayReader(scale=scale).read(path)


def read_wgbs_bed(path: str | Path) -> pd.DataFrame:
    """Read one WGBS BED file (see ``readers.WgbsBedReader``)."""
    from .readers import WgbsBedReader
    return WgbsBedReader().read(path)


def read_rrbs_methylkit(path: str | Path, merge_strands: bool = True) -> pd.DataFrame:
    """Read one RRBS methylKit file (see ``readers.RrbsMethylKitReader``)."""
    from .readers import RrbsMethylKitReader
    return RrbsMethylKitReader(merge_strands=merge_strands).read(path)


def split_matrix(path: str | Path, sample_columns: list[str], chr_col: str = "chr",
                 base_col: str = "base", scale: str = "auto",
                 sep: str = "\t") -> dict[str, pd.DataFrame]:
    """Split a matrix with one column per sample (see ``readers.MatrixReader``)."""
    from .readers import MatrixReader
    return MatrixReader(chr_col, base_col, scale, sep).read_samples(path, sample_columns)


def convert(path: str | Path, reader: str, output: str | Path, **params) -> pd.DataFrame:
    """Convert one single-sample file with the named reader and write it to ``output``."""
    from .readers import make_reader
    df = make_reader(reader, **params).read(path)
    write(df, output)
    return df


def write(df: pd.DataFrame, output: str | Path) -> None:
    """Write a common-format dataframe; missing coverage is written as ``NA``."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, sep="\t", index=False, na_rep="NA")


def read(path: str | Path) -> pd.DataFrame:
    """Read a file written by :func:`write` and check that it respects the format."""
    df = pd.read_csv(path, sep="\t", na_values="NA",
                     dtype={"chr": str, "base": np.int64, "coverage": "Int64", "freqC": float})
    validate(df)
    return df
