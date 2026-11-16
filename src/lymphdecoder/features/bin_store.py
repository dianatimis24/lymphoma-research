"""Caches of per-sample region statistics.

The mean methylation of each region of a sample depends neither on the class
labels nor on the cross-validation folds, so it is computed once per sample and
saved to disk. DMR identification and feature extraction then only read these
small files, which keeps memory use low even for hundreds of samples.

Two kinds of regions are supported with the same interface:

* ``BinStore``: bins of fixed size covering the genome;
* ``RegionStore``: a fixed list of annotated regions (promoters, CpG islands).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .regions import bin_means, decode_keys, region_id, region_key, region_stats


class FeatureStore(ABC):
    """Per-sample tables indexed by region key, with columns ``mean`` and ``n_cpgs``."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Region table of one sample in the common format."""

    @abstractmethod
    def region_names(self, keys: np.ndarray) -> list[str]:
        """Readable names (``chr_start_end``) of the given region keys."""

    def path(self, sample_id: str) -> Path:
        safe = sample_id.replace("/", "__").replace("\\", "__")
        return self.folder / f"{safe}.pkl"

    def add(self, sample_id: str, df: pd.DataFrame) -> None:
        self.compute(df).to_pickle(self.path(sample_id))

    def build(self, sample_ids: Iterable[str], loader: Callable[[str], pd.DataFrame],
              overwrite: bool = False) -> None:
        """Add every sample that is not stored yet; ``loader`` returns its dataframe."""
        for sid in sample_ids:
            if overwrite or not self.path(sid).exists():
                self.add(sid, loader(sid))

    def get(self, sample_id: str) -> pd.DataFrame:
        return pd.read_pickle(self.path(sample_id))

    def __contains__(self, sample_id: str) -> bool:
        return self.path(sample_id).exists()

    def matrix(self, sample_ids: Iterable[str], keys: np.ndarray,
               min_cpgs: int = 1) -> pd.DataFrame:
        """Samples x regions matrix for the given keys; missing values are NaN."""
        keys = np.asarray(keys, dtype=np.int64)
        rows = {}
        for sid in sample_ids:
            table = self.get(sid)
            table = table[table["n_cpgs"] >= min_cpgs]
            rows[sid] = table["mean"].reindex(keys).to_numpy()
        return pd.DataFrame.from_dict(rows, orient="index", columns=keys)


class BinStore(FeatureStore):
    """Bins of ``bin_size`` bases; the key encodes chromosome and bin start."""

    def __init__(self, folder: str | Path, bin_size: int = 1000):
        super().__init__(folder)
        self.bin_size = bin_size

    def compute(self, df):
        means = bin_means(df, self.bin_size, min_cpgs=1)
        return pd.DataFrame({"mean": means["mean"].to_numpy(),
                             "n_cpgs": means["n_cpgs"].to_numpy(dtype=np.int32)},
                            index=region_key(means["chr"], means["start"]))

    def region_names(self, keys):
        return region_id(decode_keys(np.asarray(keys, dtype=np.int64), self.bin_size)).tolist()


class RegionStore(FeatureStore):
    """A fixed list of regions; the key is the row number in ``regions``.

    The regions are saved in the store folder, so a store cannot be reused by
    mistake with a different list of regions.
    """

    def __init__(self, folder: str | Path, regions: pd.DataFrame):
        super().__init__(folder)
        self.regions = regions[["chr", "start", "end"] + (
            ["name"] if "name" in regions.columns else [])].reset_index(drop=True)
        saved = self.folder / "regions.tsv"
        if saved.exists():
            previous = pd.read_csv(saved, sep="\t")
            same = (len(previous) == len(self.regions) and
                    previous[["chr", "start", "end"]].equals(self.regions[["chr", "start", "end"]]))
            if not same:
                raise ValueError(f"{self.folder} was built for other regions; use another folder")
        else:
            self.regions.to_csv(saved, sep="\t", index=False)

    def compute(self, df):
        stats = region_stats(df, self.regions)
        return pd.DataFrame({"mean": stats["mean"].to_numpy(),
                             "n_cpgs": stats["n_cpgs"].to_numpy(dtype=np.int32)},
                            index=stats.index.to_numpy(dtype=np.int64))

    def region_names(self, keys):
        rows = self.regions.loc[np.asarray(keys, dtype=np.int64)]
        return region_id(rows).tolist()
