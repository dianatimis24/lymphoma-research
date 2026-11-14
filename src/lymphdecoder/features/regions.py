"""Aggregation of CpG methylation values into genomic regions (bins)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lymphdecoder.io.formats import CHROMOSOME_ORDER

REGION_COLUMNS = ["chr", "start", "end"]
_CHR_RANK = {c: i for i, c in enumerate(CHROMOSOME_ORDER)}
_KEY_FACTOR = 10**10


def region_key(chrom: pd.Series, start: pd.Series) -> np.ndarray:
    """Encode (chromosome, start) as one int64, which makes region lookups fast."""
    ranks = chrom.map(_CHR_RANK).to_numpy(dtype=np.int64)
    return ranks * _KEY_FACTOR + start.to_numpy(dtype=np.int64)


def decode_keys(keys: np.ndarray, bin_size: int) -> pd.DataFrame:
    """Inverse of :func:`region_key` for bins of a given size."""
    keys = np.asarray(keys, dtype=np.int64)
    start = keys % _KEY_FACTOR
    chrom = [CHROMOSOME_ORDER[r] for r in keys // _KEY_FACTOR]
    return pd.DataFrame({"chr": chrom, "start": start, "end": start + bin_size - 1})


def assign_bins(df: pd.DataFrame, bin_size: int = 1000) -> pd.DataFrame:
    """Add the 1-based, closed bin ``[start, end]`` that contains each CpG."""
    out = df.copy()
    out["start"] = ((out["base"] - 1) // bin_size) * bin_size + 1
    out["end"] = out["start"] + bin_size - 1
    return out


def bin_means(df: pd.DataFrame, bin_size: int = 1000, min_cpgs: int = 1) -> pd.DataFrame:
    """Mean methylation of each bin of one sample.

    Bins with fewer than ``min_cpgs`` CpGs measured in the sample are discarded.
    Returns columns ``chr``, ``start``, ``end``, ``mean``, ``n_cpgs``.
    """
    binned = assign_bins(df, bin_size)
    stats = (binned.groupby(REGION_COLUMNS, as_index=False, observed=True)["freqC"]
             .agg(mean="mean", n_cpgs="count"))
    return stats[stats["n_cpgs"] >= min_cpgs].reset_index(drop=True)


def region_id(regions: pd.DataFrame) -> pd.Series:
    """Identifier ``chr_start_end`` used as feature name."""
    return (regions["chr"].astype(str) + "_" + regions["start"].astype(str)
            + "_" + regions["end"].astype(str))


def region_values(df: pd.DataFrame, regions: pd.DataFrame) -> pd.Series:
    """Mean methylation of one sample in each of the given 1-based, closed regions.

    Regions may have any length (bins, promoters, CpG islands). Regions without
    any measured CpG get NaN. The result is indexed by region id.
    """
    regions = regions.reset_index(drop=True)
    ids = region_id(regions)
    values = np.full(len(regions), np.nan)
    for chrom, group in regions.groupby("chr"):
        sample = df[df["chr"] == chrom].sort_values("base")
        bases = sample["base"].to_numpy()
        cum = np.concatenate([[0.0], sample["freqC"].to_numpy().cumsum()])
        lo = bases.searchsorted(group["start"].to_numpy(), side="left")
        hi = bases.searchsorted(group["end"].to_numpy(), side="right")
        n = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            values[group.index] = np.where(n > 0, (cum[hi] - cum[lo]) / n, np.nan)
    return pd.Series(values, index=ids.to_numpy())


def feature_matrix(samples: dict[str, pd.DataFrame], regions: pd.DataFrame) -> pd.DataFrame:
    """Samples x regions matrix of mean methylation values; missing values stay NaN."""
    rows = {name: region_values(df, regions) for name, df in samples.items()}
    return pd.DataFrame.from_dict(rows, orient="index")
