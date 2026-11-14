import numpy as np
import pandas as pd
import pytest

from lymphdecoder.features.bin_store import BinStore

N_BINS = 40
CPGS_PER_BIN = 12
SIGNAL_BINS = {"A": range(0, 5), "B": range(5, 10), "C": range(10, 15)}


def synthetic_sample(label: str, rng: np.random.Generator) -> pd.DataFrame:
    """One sample in the common format: 40 bins of 12 CpGs on chr1; each class is
    40 points more methylated in its own 5 bins."""
    rows = []
    for b in range(N_BINS):
        level = 30.0 + (40.0 if b in SIGNAL_BINS[label] else 0.0) + rng.normal(0, 5)
        for k in range(CPGS_PER_BIN):
            value = float(np.clip(level + rng.normal(0, 3), 0, 100))
            rows.append(["chr1", b * 1000 + 1 + 50 * k, 30, value])
    return pd.DataFrame(rows, columns=["chr", "base", "coverage", "freqC"])


@pytest.fixture
def synthetic(tmp_path):
    """BinStore and labels for 3 classes with 8 samples each."""
    rng = np.random.default_rng(0)
    store = BinStore(tmp_path / "bins")
    labels = {}
    for label in SIGNAL_BINS:
        for i in range(8):
            sid = f"{label}{i}"
            store.add(sid, synthetic_sample(label, rng))
            labels[sid] = label
    return store, pd.Series(labels)
