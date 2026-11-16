"""Exploratory analyses: data summaries (E1), platform overlap (E2) and natural grouping (E3)."""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
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


def flag_samples(summary: pd.DataFrame, min_cpgs: int = 1000) -> pd.DataFrame:
    """Samples that look wrong: too few CpGs or a mean methylation far from the others."""
    flags = pd.DataFrame(index=summary.index)
    flags["too_few_cpgs"] = summary["n_cpgs"] < min_cpgs
    m = summary["mean_methylation"]
    flags["outlier_mean"] = (m - m.median()).abs() > 3 * (m.std(ddof=0) + 1e-9)
    return flags[flags.any(axis=1)]


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
    """Cluster the samples and compare the clusters with each grouping (E3).

    The samples are clustered with k-means on the first principal components;
    the adjusted Rand index (ARI) measures the agreement of the clusters with the
    classes, the platforms, the sample types, etc. Returns the cluster of each
    sample and a table with one ARI per grouping.
    """
    Z = StandardScaler().fit_transform(X)
    n_components = max(1, min(n_components, Z.shape[0] - 1, Z.shape[1]))
    P = PCA(n_components=n_components, random_state=seed).fit_transform(Z)
    clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(P)
    clusters = pd.Series(clusters, index=X.index, name="cluster")
    rows = [{"grouping": name, "n_groups": g.loc[X.index].nunique(),
             "ari": adjusted_rand_score(g.loc[X.index], clusters)}
            for name, g in groupings.items()]
    return clusters, pd.DataFrame(rows)
