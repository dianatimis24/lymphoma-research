"""Quality control: Twist panel filtering and coverage filtering.

All functions receive and return dataframes in the common format
(``chr``, ``base``, ``coverage``, ``freqC``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_panel(path: str | Path) -> pd.DataFrame:
    """Read a panel BED file (0-based, half-open intervals) and merge overlapping intervals."""
    panel = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                        names=["chr", "start", "end"], comment="#")
    return merge_intervals(panel)


def merge_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping or adjacent intervals on each chromosome."""
    merged = []
    for chrom, group in intervals.sort_values(["chr", "start"]).groupby("chr", sort=False):
        starts = group["start"].to_numpy()
        ends = group["end"].to_numpy()
        cur_start, cur_end = starts[0], ends[0]
        for start, end in zip(starts[1:], ends[1:]):
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((chrom, cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((chrom, cur_start, cur_end))
    return pd.DataFrame(merged, columns=["chr", "start", "end"])


def in_intervals(positions: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Return a mask of the 1-based positions that fall in sorted, non-overlapping
    0-based half-open intervals."""
    zero_based = positions - 1
    idx = np.searchsorted(starts, zero_based, side="right") - 1
    valid = idx >= 0
    mask = np.zeros(len(positions), dtype=bool)
    mask[valid] = zero_based[valid] < ends[idx[valid]]
    return mask


def filter_by_panel(df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only the CpGs covered by the panel intervals."""
    keep = np.zeros(len(df), dtype=bool)
    for chrom, intervals in panel.groupby("chr"):
        rows = (df["chr"] == chrom).to_numpy()
        if rows.any():
            keep[rows] = in_intervals(df.loc[rows, "base"].to_numpy(),
                                      intervals["start"].to_numpy(),
                                      intervals["end"].to_numpy())
    return df[keep].reset_index(drop=True)


def filter_by_coverage(df: pd.DataFrame, min_coverage: int = 10) -> pd.DataFrame:
    """Remove CpGs covered by fewer than ``min_coverage`` reads.

    Array samples have no coverage and are kept unchanged.
    """
    if df["coverage"].isna().all():
        return df.reset_index(drop=True)
    return df[df["coverage"] >= min_coverage].reset_index(drop=True)
