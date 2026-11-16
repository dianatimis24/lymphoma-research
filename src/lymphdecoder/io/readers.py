"""Readers that convert files of different platforms to the common format.

Each input format is handled by one class. Classes are registered under a name,
so a new format is supported by writing a new class and naming it in the
dataset configuration; no other code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .formats import finalize, to_percent

READERS: dict[str, type["Reader"]] = {}


def register(name: str):
    """Class decorator that makes a reader available under ``name``."""
    def decorator(cls):
        if name in READERS:
            raise ValueError(f"Reader {name!r} is already registered")
        cls.name = name
        READERS[name] = cls
        return cls
    return decorator


def make_reader(name: str, **params) -> "Reader":
    """Create the reader registered under ``name`` with the given parameters."""
    if name not in READERS:
        raise ValueError(f"Unknown reader {name!r}, expected one of {sorted(READERS)}")
    return READERS[name](**params)


class Reader(ABC):
    """Reads files that contain one sample each."""

    name = "abstract"

    @abstractmethod
    def read(self, path: str | Path) -> pd.DataFrame:
        """Return the sample in the common format."""


class MultiSampleReader(ABC):
    """Reads files that contain several samples, one column per sample."""

    name = "abstract"

    @abstractmethod
    def read_samples(self, path: str | Path, columns: list[str]) -> dict[str, pd.DataFrame]:
        """Return one common-format dataframe for each requested column."""


@register("array")
class ArrayReader(Reader):
    """Array sample with columns ``chr``, ``base`` and a methylation column.

    The methylation column is ``freqC`` or, if absent, the third column.
    """

    def __init__(self, scale: str = "auto", sep: str = "\t"):
        self.scale = scale
        self.sep = sep

    def read(self, path):
        df = pd.read_csv(path, sep=self.sep)
        value_col = "freqC" if "freqC" in df.columns else df.columns[2]
        return finalize(pd.DataFrame({"chr": df["chr"], "base": df["base"],
                                      "freqC": to_percent(df[value_col], self.scale)}))


@register("wgbs_bed")
class WgbsBedReader(Reader):
    """WGBS BED file ``chr start end methylated_reads total_reads`` (0-based start).

    A sixth column with the strand (``+``/``-``) is used if present; otherwise the
    strand is unknown (``*``).
    """

    def read(self, path):
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
        if df.shape[1] not in (5, 6):
            raise ValueError(f"{path}: expected 5 or 6 columns, found {df.shape[1]}")
        df.columns = ["chr", "start", "end", "methylated", "coverage", "strand"][:df.shape[1]]
        df = df[df["coverage"] > 0]
        return finalize(pd.DataFrame({
            "chr": df["chr"],
            "base": df["start"] + 1,
            "strand": df["strand"] if "strand" in df.columns else "*",
            "coverage": df["coverage"],
            "freqC": df["methylated"] * 100.0 / df["coverage"],
        }))


@register("rrbs_methylkit")
class RrbsMethylKitReader(Reader):
    """RRBS file in methylKit format: ``chrBase chr base strand coverage freqC freqT``.

    ``strand`` is ``F`` or ``R``. A CpG measured on the reverse strand is reported
    at the position of its G, one base after the C on the forward strand.

    * ``merge_strands=True`` (default): the two measurements of a CpG are combined,
      weighting them by coverage; the row gets strand ``*`` and the position of
      the C on the + strand, so it matches arrays and WGBS.
    * ``merge_strands=False``: every measurement is kept with strand ``+`` or ``-``
      and the position given in the file.

    The header line is optional, and columns after ``freqT`` (present in some
    files, sometimes with a trailing tab) are ignored.
    """

    COLUMNS = ["chrBase", "chr", "base", "strand", "coverage", "freqC", "freqT"]

    def __init__(self, merge_strands: bool = True):
        self.merge_strands = merge_strands

    def read(self, path):
        df = pd.read_csv(path, sep="\t", header=None, usecols=range(len(self.COLUMNS)),
                         dtype=str, skip_blank_lines=True)
        df.columns = self.COLUMNS
        if len(df) and df.iloc[0]["chrBase"] == "chrBase":
            df = df.iloc[1:]
        df = df.dropna(subset=["chr", "base", "strand", "coverage", "freqC"])
        df["base"] = df["base"].astype("int64")
        df["coverage"] = df["coverage"].astype("int64")
        df["freqC"] = df["freqC"].astype(float)
        base = df["base"]
        strand = df["strand"].str.strip().map({"F": "+", "R": "-", "+": "+", "-": "-"})
        if strand.isna().any():
            raise ValueError(f"{path}: strand must be F/R or +/-")
        if self.merge_strands:
            base = base.where(strand == "+", base - 1)
            strand = pd.Series("*", index=df.index)
        out = pd.DataFrame({"chr": df["chr"], "base": base, "strand": strand,
                            "coverage": df["coverage"].astype("int64"),
                            "methylated": df["coverage"] * df["freqC"] / 100.0})
        out = out.groupby(["chr", "base", "strand"], as_index=False)[["coverage", "methylated"]].sum()
        out["freqC"] = out["methylated"] * 100.0 / out["coverage"]
        return finalize(out.drop(columns="methylated"))


@register("matrix")
class MatrixReader(MultiSampleReader):
    """Matrix with one column per sample (array beta values, EM-seq rate tables)."""

    def __init__(self, chr_col: str = "chr", base_col: str = "base",
                 scale: str = "auto", sep: str = "\t"):
        self.chr_col = chr_col
        self.base_col = base_col
        self.scale = scale
        self.sep = sep

    def read_samples(self, path, columns):
        df = pd.read_csv(path, sep=self.sep, usecols=[self.chr_col, self.base_col] + list(columns))
        return {col: finalize(pd.DataFrame({"chr": df[self.chr_col], "base": df[self.base_col],
                                            "freqC": to_percent(df[col], self.scale)}))
                for col in columns}
