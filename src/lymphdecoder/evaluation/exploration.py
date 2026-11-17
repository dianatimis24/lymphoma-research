"""Exploratory analyses: data summaries (E1), platform overlap (E2) and natural grouping (E3)."""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from lymphdecoder.features.dmr import resolve_min_cpgs


def sample_summary(sample_ids: Iterable[str], loader: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    """Per-sample statistics of the converted files, used to check the conversion (E1)."""
    rows = []
    for sid in sample_ids:
        df = loader(sid)
        cov = df["coverage"].astype("float")
        rows.append({
            "sample_id": sid,
            "n_cpgs": len(df),
            "mean_methylation": df["freqC"].mean() if len(df) else np.nan,
            "fraction_0": (df["freqC"] == 0).mean() if len(df) else np.nan,
            "fraction_100": (df["freqC"] == 100).mean() if len(df) else np.nan,
            "median_coverage": cov.median() if cov.notna().any() else np.nan,
            "chromosomes": df["chr"].nunique(),
        })
    return pd.DataFrame(rows).set_index("sample_id")


def _robust_z(values: pd.Series) -> pd.Series:
    """Distance from the median in units of the scaled median absolute deviation."""
    median = values.median()
    mad = 1.4826 * (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(0.0, index=values.index)
    return (values - median) / mad


def flag_samples(summary: pd.DataFrame, min_cpgs: int = 1000, max_z: float = 3.5,
                 min_coverage_ratio: float = 0.25) -> pd.DataFrame:
    """Samples that look wrong compared with the other samples of the same platform.

    Platforms measure very different numbers of CpGs and levels of methylation,
    so every sample is compared only with its own platform. Robust statistics
    (median and median absolute deviation) are used, so that one bad sample does
    not hide itself by inflating the spread. A sample is flagged if it has fewer
    than ``min_cpgs`` CpGs, a median coverage below ``min_coverage_ratio`` times
    the platform median, or a mean methylation, fraction of fully methylated or
    fraction of unmethylated CpGs more than ``max_z`` robust deviations away from
    the platform median. The ``reason`` column lists the triggered checks.
    """
    groups = summary["platform"] if "platform" in summary.columns else pd.Series(
        "all", index=summary.index)
    flags = pd.DataFrame(False, index=summary.index,
                         columns=["too_few_cpgs", "low_coverage", "outlier_mean",
                                  "outlier_fraction_100", "outlier_fraction_0"])
    flags["too_few_cpgs"] = summary["n_cpgs"] < min_cpgs
    for _, idx in groups.groupby(groups).groups.items():
        part = summary.loc[idx]
        cov = part["median_coverage"]
        if cov.notna().any():
            flags.loc[idx, "low_coverage"] = cov < min_coverage_ratio * cov.median()
        if len(part) >= 4:
            flags.loc[idx, "outlier_mean"] = _robust_z(part["mean_methylation"]).abs() > max_z
            flags.loc[idx, "outlier_fraction_100"] = _robust_z(part["fraction_100"]).abs() > max_z
            flags.loc[idx, "outlier_fraction_0"] = _robust_z(part["fraction_0"]).abs() > max_z
    flagged = flags[flags.any(axis=1)].copy()
    flagged["reason"] = [", ".join(c for c in flags.columns if row[c])
                         for _, row in flagged.iterrows()]
    return flagged


def platform_overlap(store, platforms: pd.Series, min_cpgs=1,
                     min_sample_fraction: float = 0.5) -> pd.DataFrame:
    """Regions measured on each pair of platforms (E2).

    A region counts for a platform if it passes the density threshold in at least
    ``min_sample_fraction`` of the platform's samples.
    """
    measured = {}
    for platform, ids in platforms.groupby(platforms).groups.items():
        counts = None
        threshold = resolve_min_cpgs(min_cpgs, platform)
        for sid in ids:
            table = store.get(sid)
            ones = pd.Series(1, index=table.index[table["n_cpgs"] >= threshold])
            counts = ones if counts is None else counts.add(ones, fill_value=0)
        measured[platform] = set(counts.index[counts >= min_sample_fraction * len(ids)])
    rows = [{"platform_a": p, "platform_b": p, "regions_a": len(s), "regions_b": len(s),
             "shared": len(s), "jaccard": 1.0 if s else np.nan} for p, s in measured.items()]
    for a, b in combinations(sorted(measured), 2):
        shared = len(measured[a] & measured[b])
        union = len(measured[a] | measured[b])
        rows.append({"platform_a": a, "platform_b": b, "regions_a": len(measured[a]),
                     "regions_b": len(measured[b]), "shared": shared,
                     "jaccard": shared / union if union else np.nan})
    return pd.DataFrame(rows)


def top_variance(X: pd.DataFrame, n: int) -> pd.DataFrame:
    """Median-imputed matrix restricted to the ``n`` most variable columns."""
    filled = X.fillna(X.median())
    filled = filled.loc[:, filled.notna().all()]
    keep = filled.var().sort_values(ascending=False).index[:n]
    return filled[keep]


def embed(X: pd.DataFrame, method: str = "umap", seed: int = 42) -> np.ndarray:
    """Two-dimensional embedding of the samples (UMAP, or PCA if UMAP is not installed)."""
    Z = StandardScaler().fit_transform(X)
    if method == "umap":
        try:
            import umap
        except ImportError:
            method = "pca"
        else:
            n_neighbors = max(2, min(15, len(X) - 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return umap.UMAP(n_neighbors=n_neighbors, random_state=seed).fit_transform(Z)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(Z)
    raise ValueError(f"Unknown embedding method {method!r}")


def grouping_agreement(X: pd.DataFrame, groupings: dict[str, pd.Series], n_clusters: int,
                       n_components: int = 10, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """How strongly the samples group by each known grouping (E3).

    The samples are standardised and projected on their first principal
    components. Two complementary measures are computed for every grouping
    (class, platform, sample type, dataset):

    * ``silhouette``: how well the known groups are separated in that space,
      from -1 (wrong) to 1 (perfectly separated); no clustering is involved;
    * ``ari``: agreement (adjusted Rand index) between the groups and an
      unsupervised Ward clustering into ``n_clusters`` clusters, which is more
      stable than k-means on small sets of samples.

    Returns the cluster of each sample and a table with both measures.
    """
    Z = StandardScaler().fit_transform(X)
    n_components = max(1, min(n_components, Z.shape[0] - 1, Z.shape[1]))
    P = PCA(n_components=n_components, random_state=seed).fit_transform(Z)
    clusters = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(P)
    clusters = pd.Series(clusters, index=X.index, name="cluster")
    rows = []
    for name, g in groupings.items():
        labels = g.loc[X.index]
        n_groups = labels.nunique()
        sil = silhouette_score(P, labels) if 1 < n_groups < len(labels) else np.nan
        rows.append({"grouping": name, "n_groups": n_groups, "silhouette": sil,
                     "ari": adjusted_rand_score(labels, clusters)})
    return clusters, pd.DataFrame(rows)
