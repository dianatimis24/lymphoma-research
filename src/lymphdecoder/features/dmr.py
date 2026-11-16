"""Identification of differentially methylated regions (DMRs).

The regions are bins of fixed size or annotated regions (promoters, CpG
islands). A region is a DMR if the mean methylation of two classes differs by at
least ``threshold`` percentage points. All pairs of classes are compared and the
union of the resulting regions is returned.

Class means are accumulated one sample at a time, so the per-sample region means
never have to be held in memory together.

The CpG density filter (minimum number of measured CpGs in a region) can depend
on the platform: arrays measure far fewer CpGs than sequencing, so a threshold
suited to sequencing would remove almost every region of an array sample.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Iterable, Mapping, Union

import numpy as np
import pandas as pd

from .regions import REGION_COLUMNS, bin_means, decode_keys, region_key

MinCpGs = Union[int, Mapping[str, int]]


def resolve_min_cpgs(min_cpgs: MinCpGs, platform: str | None) -> int:
    """Density threshold for a platform.

    ``min_cpgs`` is a number (same threshold for all platforms) or a mapping from
    platform to threshold, with an optional ``default`` entry (10 if absent).
    """
    if isinstance(min_cpgs, Mapping):
        if platform is not None and platform in min_cpgs:
            return int(min_cpgs[platform])
        return int(min_cpgs.get("default", 10))
    return int(min_cpgs)


def accumulate_class_means(tables: Iterable[tuple[str, pd.DataFrame, int]],
                           min_sample_fraction: float = 0.5) -> pd.DataFrame:
    """Mean methylation of each region for each class.

    ``tables`` yields ``(label, table, min_cpgs)`` triples, where ``table`` is
    indexed by region key and has the columns ``mean`` and ``n_cpgs`` and
    ``min_cpgs`` is the density threshold for that sample. The mean of a class is
    the average of the per-sample means, so every sample has the same weight. A
    region is kept for a class only if it passes the density threshold in at
    least ``min_sample_fraction`` of the class samples. Returns a dataframe indexed
    by region key with one column per class.
    """
    sums: dict[str, pd.Series] = {}
    counts: dict[str, pd.Series] = {}
    sizes: dict[str, int] = {}
    for label, table, min_cpgs in tables:
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
    """DMRs from a regions x classes table of mean methylation values.

    Every pair of classes is compared on the regions present in both. Returns the
    union of the DMRs indexed like ``means``, with the columns ``n_pairs`` (the
    number of class pairs in which the region is differentially methylated) and
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


def _store_tables(store, sample_ids, labels, min_cpgs, platforms):
    for sid in sample_ids:
        platform = platforms[sid] if platforms is not None else None
        yield labels[sid], store.get(sid), resolve_min_cpgs(min_cpgs, platform)


def dmrs_from_store(store, sample_ids: Iterable[str], labels: pd.Series,
                    min_cpgs: MinCpGs = 10, min_sample_fraction: float = 0.5,
                    threshold: float = 25.0, platforms: pd.Series | None = None) -> pd.DataFrame:
    """DMRs computed only from the given samples of a store.

    Returns a dataframe indexed by region key with ``n_pairs`` and ``max_diff``.
    """
    tables = _store_tables(store, sample_ids, labels, min_cpgs, platforms)
    return naive_dmrs(accumulate_class_means(tables, min_sample_fraction), threshold)


def class_means(samples: dict[str, pd.DataFrame], labels: dict[str, str],
                bin_size: int = 1000, min_cpgs: MinCpGs = 10,
                min_sample_fraction: float = 0.5,
                platforms: dict[str, str] | None = None) -> pd.DataFrame:
    """Class means of bins for samples held in memory, indexed by (chr, start, end)."""
    def tables():
        for name, df in samples.items():
            m = bin_means(df, bin_size, min_cpgs=1)
            platform = platforms.get(name) if platforms else None
            yield labels[name], pd.DataFrame(
                {"mean": m["mean"].to_numpy(), "n_cpgs": m["n_cpgs"].to_numpy()},
                index=region_key(m["chr"], m["start"])), resolve_min_cpgs(min_cpgs, platform)

    means = accumulate_class_means(tables(), min_sample_fraction)
    means.index = pd.MultiIndex.from_frame(decode_keys(means.index.to_numpy(), bin_size))
    return means


def identify_dmrs(samples: dict[str, pd.DataFrame], labels: dict[str, str],
                  bin_size: int = 1000, min_cpgs: MinCpGs = 10,
                  min_sample_fraction: float = 0.5, threshold: float = 25.0,
                  platforms: dict[str, str] | None = None) -> pd.DataFrame:
    """DMRs for samples held in memory, with columns chr, start, end, n_pairs, max_diff."""
    means = class_means(samples, labels, bin_size, min_cpgs, min_sample_fraction, platforms)
    result = naive_dmrs(means, threshold)
    if result.empty:
        return pd.DataFrame(columns=REGION_COLUMNS + ["n_pairs", "max_diff"])
    return result.reset_index().sort_values(REGION_COLUMNS).reset_index(drop=True)


DMR_FINDERS: dict[str, type["DMRFinder"]] = {}


def register_dmr_finder(name: str):
    """Class decorator that makes a region selection method available under ``name``."""
    def decorator(cls):
        cls.name = name
        DMR_FINDERS[name] = cls
        return cls
    return decorator


def make_dmr_finder(name: str, **params) -> "DMRFinder":
    if name not in DMR_FINDERS:
        raise ValueError(f"Unknown DMR method {name!r}, expected one of {sorted(DMR_FINDERS)}")
    return DMR_FINDERS[name](**params)


class DMRFinder(ABC):
    """Selects the candidate regions (features) from the training samples of a store."""

    name = "abstract"

    def __init__(self, min_cpgs: MinCpGs = 10, min_sample_fraction: float = 0.5,
                 threshold: float = 25.0):
        self.min_cpgs = min_cpgs
        self.min_sample_fraction = min_sample_fraction
        self.threshold = threshold

    @abstractmethod
    def find(self, store, sample_ids: Iterable[str], labels: pd.Series,
             platforms: pd.Series | None = None) -> pd.DataFrame:
        """Return a dataframe indexed by region key, one row per selected region."""


@register_dmr_finder("naive")
class NaiveDMRFinder(DMRFinder):
    """Regions whose class means differ by at least ``threshold`` for some pair of classes."""

    def find(self, store, sample_ids, labels, platforms=None):
        return dmrs_from_store(store, sample_ids, labels, self.min_cpgs,
                               self.min_sample_fraction, self.threshold, platforms)


@register_dmr_finder("all")
class AllRegionsFinder(DMRFinder):
    """Every region measured densely enough in ``min_sample_fraction`` of the samples.

    No class information is used (``threshold`` is ignored); the selection of
    informative regions is left to the feature selection step.
    """

    def find(self, store, sample_ids, labels, platforms=None):
        ids = list(sample_ids)
        single = pd.Series("all", index=ids)
        tables = _store_tables(store, ids, single, self.min_cpgs, platforms)
        means = accumulate_class_means(tables, self.min_sample_fraction)
        return pd.DataFrame({"mean": means["all"]}) if not means.empty else means
