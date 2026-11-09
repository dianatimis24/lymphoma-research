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
    """Read one array sample with columns ``chr``, ``base`` and a methylation column.

    The methylation column is ``freqC`` or, if absent, the third column (beta values).
    Arrays give no read coverage.
    """
    df = pd.read_csv(path, sep="\t")
    value_col = "freqC" if "freqC" in df.columns else df.columns[2]
    out = pd.DataFrame({
        "chr": df["chr"],
        "base": df["base"],
        "freqC": to_percent(df[value_col], scale),
    })
    return finalize(out)


def read_wgbs_bed(path: str | Path) -> pd.DataFrame:
    """Read a WGBS BED file: ``chr start end methylated_reads total_reads``.

    BED coordinates are 0-based, so the 1-based position is ``start + 1``.
    """
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["chr", "start", "end", "methylated", "coverage"],
                     comment="#")
    df = df[df["coverage"] > 0]
    out = pd.DataFrame({
        "chr": df["chr"],
        "base": df["start"] + 1,
        "coverage": df["coverage"],
        "freqC": df["methylated"] * 100.0 / df["coverage"],
    })
    return finalize(out)


def read_rrbs_methylkit(path: str | Path, merge_strands: bool = True) -> pd.DataFrame:
    """Read an RRBS file in methylKit format.

    Columns: ``chrBase chr base strand coverage freqC freqT``, with 1-based
    positions and strand ``F`` or ``R``. A CpG measured on the reverse strand is
    reported at the position of its G, one base after the C on the forward strand.
    With ``merge_strands`` the two measurements of a CpG are combined into one,
    weighting the methylation percentages by coverage.
    """
    df = pd.read_csv(path, sep="\t")
    base = df["base"].astype("int64")
    if merge_strands:
        base = base.where(df["strand"] == "F", base - 1)
    out = pd.DataFrame({
        "chr": df["chr"],
        "base": base,
        "coverage": df["coverage"].astype("int64"),
        "methylated": df["coverage"] * df["freqC"] / 100.0,
    })
    out = out.groupby(["chr", "base"], as_index=False)[["coverage", "methylated"]].sum()
    out["freqC"] = out["methylated"] * 100.0 / out["coverage"]
    return finalize(out.drop(columns="methylated"))


def split_matrix(path: str | Path, sample_columns: list[str], chr_col: str = "chr",
                 base_col: str = "base", scale: str = "auto",
                 sep: str = "\t") -> dict[str, pd.DataFrame]:
    """Split a matrix with one column per sample into common-format dataframes.

    Used for array matrices (beta values) and for the EM-seq rate matrix, which
    store all samples of a dataset in a single file.
    """
    df = pd.read_csv(path, sep=sep, usecols=[chr_col, base_col] + sample_columns)
    samples = {}
    for col in sample_columns:
        out = pd.DataFrame({
            "chr": df[chr_col],
            "base": df[base_col],
            "freqC": to_percent(df[col], scale),
        })
        samples[col] = finalize(out)
    return samples


READERS = {
    "array": read_array,
    "wgbs": read_wgbs_bed,
    "rrbs": read_rrbs_methylkit,
}


def convert(path: str | Path, platform: str, output: str | Path, **kwargs) -> pd.DataFrame:
    """Convert one file to the common format and write it to ``output``."""
    if platform not in READERS:
        raise ValueError(f"Unknown platform {platform!r}, expected one of {list(READERS)}")
    df = READERS[platform](path, **kwargs)
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
