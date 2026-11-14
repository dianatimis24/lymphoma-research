"""Sample metadata.

The metadata file is a CSV with one row per sample and the columns:

    sample_id    unique identifier of the sample
    file         path of the sample file in the common format (relative to the file)
    class        lymphoma subtype or ``Normal``
    platform     array_450k, array_epic, rrbs, wgbs or emseq
    dataset      source of the sample (e.g. a GEO accession)
    sample_type  tissue or plasma

An optional boolean column ``exclude`` marks samples left out of all analyses
(for example mixed or diluted samples).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["sample_id", "file", "class", "platform", "dataset", "sample_type"]


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read and validate the metadata file; excluded samples are removed.

    File paths are resolved relative to the folder of the metadata file.
    """
    path = Path(path)
    meta = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")
    if meta["sample_id"].duplicated().any():
        dup = meta.loc[meta["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample ids: {dup}")
    if meta[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Metadata contains empty values")
    if "exclude" in meta.columns:
        meta = meta[~meta["exclude"].fillna(False).astype(bool)]
    meta = meta.copy()
    meta["file"] = [str((path.parent / f).resolve()) for f in meta["file"]]
    return meta.set_index("sample_id")


def merge_small_classes(labels: pd.Series, min_size: int,
                        merged_name: str = "Other lymphoma",
                        keep: tuple[str, ...] = ("Normal",)) -> pd.Series:
    """Merge the classes with fewer than ``min_size`` samples into one class.

    Classes listed in ``keep`` are never merged.
    """
    counts = labels.value_counts()
    small = [c for c in counts.index if counts[c] < min_size and c not in keep]
    return labels.where(~labels.isin(small), merged_name)


def metadata_template(root: str | Path, pattern: str = "*.tsv") -> pd.DataFrame:
    """Build a metadata draft from a folder with one subfolder per class.

    ``platform``, ``dataset`` and ``sample_type`` are left empty and must be
    filled in by hand before the file is used.
    """
    root = Path(root)
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(class_dir.glob(pattern)):
            rows.append({"sample_id": f"{class_dir.name}/{f.stem}",
                         "file": str(f.relative_to(root)),
                         "class": class_dir.name,
                         "platform": "", "dataset": "", "sample_type": ""})
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
