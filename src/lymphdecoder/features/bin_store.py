"""Cache of per-sample bin means.

The mean methylation of each bin of a sample depends neither on the class labels
nor on the cross-validation folds, so it is computed once per sample and saved
to disk. DMR identification and feature extraction then only read these small
files, which keeps memory use low even for hundreds of samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .regions import bin_means, region_key


class BinStore:
    """Per-sample bin means stored as one pickle file per sample.

    Each stored series is indexed by the region key of the bin and has two
    columns: ``mean`` (mean methylation) and ``n_cpgs`` (number of CpGs measured).
    """

    def __init__(self, folder: str | Path, bin_size: int = 1000):
        self.folder = Path(folder)
        self.bin_size = bin_size
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, sample_id: str) -> Path:
        safe = sample_id.replace("/", "__").replace("\\", "__")
        return self.folder / f"{safe}.pkl"

    def add(self, sample_id: str, df: pd.DataFrame) -> None:
        """Compute and save the bin means of one sample in the common format."""
        means = bin_means(df, self.bin_size, min_cpgs=1)
        table = pd.DataFrame({"mean": means["mean"].to_numpy(),
                              "n_cpgs": means["n_cpgs"].to_numpy(dtype=np.int32)},
                             index=region_key(means["chr"], means["start"]))
        table.to_pickle(self.path(sample_id))

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
        """Samples x bins matrix for the given bin keys; missing values are NaN."""
        keys = np.asarray(keys, dtype=np.int64)
        rows = {}
        for sid in sample_ids:
            table = self.get(sid)
            table = table[table["n_cpgs"] >= min_cpgs]
            rows[sid] = table["mean"].reindex(keys).to_numpy()
        return pd.DataFrame.from_dict(rows, orient="index", columns=keys)
