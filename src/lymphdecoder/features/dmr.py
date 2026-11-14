"""Naive identification of differentially methylated regions (DMRs).

The genome is split into bins; a bin is a DMR if the mean methylation of two
classes differs by at least ``threshold`` percentage points. All pairs of classes
are compared and the union of the resulting bins is returned.

Class means are accumulated one sample at a time, so the per-sample bin means
never have to be held in memory together.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .regions import REGION_COLUMNS, bin_means, decode_keys, region_key


def accumulate_class_means(tables: Iterable[tuple[str, pd.DataFrame]],
                           min_cpgs: int = 10,
                           min_sample_fraction: float = 0.5) -> pd.DataFrame:
    """Mean methylation of each bin for each class.

    ``tables`` yields ``(label, table)`` pairs, where ``table`` is indexed by bin
    key and has the columns ``mean`` and ``n_cpgs`` (as stored by ``BinStore``).
    The mean of a class is the average of the per-sample bin means, so every
    sample has the same weight. A bin is kept for a class only if it has at least
    ``min_cpgs`` CpGs in at least ``min_sample_fraction`` of the class samples.
    Returns a dataframe indexed by bin key with one column per class.
    """
    sums: dict[str, pd.Series] = {}
    counts: dict[str, pd.Series] = {}
    sizes: dict[str, int] = {}
    for label, table in tables:
        dense = table.loc[table["n_cpgs"] >= min_cpgs, "mean"]
        ones = pd.Series(1, index=dense.index, dtype=np.int32)
        if label in sums:
            sums[label] = sums[label].add(dense, fill_value=0.0)
            counts[label] = counts[label].add(ones, fill_value=0)
        else:
            sums[label], counts[label] = dense.astype(float), ones
        sizes[label] = sizes.get(label, 0) + 1

    columns = {}
    for label in sorted(sums):
        enough = counts[label] >= min_sample_fraction * sizes[label]
        columns[label] = (sums[label] / counts[label])[enough]
    return pd.DataFrame(columns).sort_index()


def naive_dmrs(means: pd.DataFrame, threshold: float = 25.0) -> pd.DataFrame:
    """DMRs from a bins x classes table of mean methylation values.

    Every pair of classes is compared on the bins present in both. Returns the
    union of the DMRs indexed like ``means``, with the columns ``n_pairs`` (the
    number of class pairs in which the bin is differentially methylated) and
    ``max_diff``.
    """
    hits = []
    for a, b in combinations(means.columns, 2):
        diff = (means[a] - means[b]).abs().dropna()
        hits.append(diff[diff >= threshold])
    if not hits or all(h.empty for h in hits):
        return pd.DataFrame({"n_pairs": pd.Series(dtype=int), "max_diff": pd.Series(dtype=float)})
    all_hits = pd.concat(hits)
    result = all_hits.groupby(level=list(range(all_hits.index.nlevels))).agg(["count", "max"])
    return result.rename(columns={"count": "n_pairs", "max": "max_diff"}).sort_index()


def dmrs_from_store(store, sample_ids: Iterable[str], labels: pd.Series,
                    min_cpgs: int = 10, min_sample_fraction: float = 0.5,
                    threshold: float = 25.0) -> pd.DataFrame:
    """DMRs computed only from the given samples of a ``BinStore``.

    Returns a dataframe indexed by bin key with ``n_pairs`` and ``max_diff``.
    """
    tables = ((labels[sid], store.get(sid)) for sid in sample_ids)
    means = accumulate_class_means(tables, min_cpgs, min_sample_fraction)
    return naive_dmrs(means, threshold)


def class_means(samples: dict[str, pd.DataFrame], labels: dict[str, str],
                bin_size: int = 1000, min_cpgs: int = 10,
                min_sample_fraction: float = 0.5) -> pd.DataFrame:
    """Class means for samples held in memory, indexed by (chr, start, end)."""
    def tables():
        for name, df in samples.items():
            m = bin_means(df, bin_size, min_cpgs=1)
            yield labels[name], pd.DataFrame(
                {"mean": m["mean"].to_numpy(), "n_cpgs": m["n_cpgs"].to_numpy()},
                index=region_key(m["chr"], m["start"]))

    means = accumulate_class_means(tables(), min_cpgs, min_sample_fraction)
    means.index = pd.MultiIndex.from_frame(decode_keys(means.index.to_numpy(), bin_size))
    return means


def identify_dmrs(samples: dict[str, pd.DataFrame], labels: dict[str, str],
                  bin_size: int = 1000, min_cpgs: int = 10,
                  min_sample_fraction: float = 0.5, threshold: float = 25.0) -> pd.DataFrame:
    """DMRs for samples held in memory, with columns chr, start, end, n_pairs, max_diff."""
    means = class_means(samples, labels, bin_size, min_cpgs, min_sample_fraction)
    result = naive_dmrs(means, threshold)
    if result.empty:
        return pd.DataFrame(columns=REGION_COLUMNS + ["n_pairs", "max_diff"])
    return result.reset_index().sort_values(REGION_COLUMNS).reset_index(drop=True)
