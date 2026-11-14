"""Uniform processing of all datasets, driven by a configuration file.

The configuration (YAML) lists the datasets, where their files are, which reader
understands them and which class each sample belongs to. Processing converts
every sample to the common format, applies the Twist panel and coverage filters,
writes one file per sample and builds the metadata file ``samples.csv``.

Example::

    output: processed
    panel: reference/twist_panel.bed
    min_coverage: 10
    datasets:
      - name: GSE66329
        reader: rrbs_methylkit
        folder: raw/DLBCL
        pattern: "*.txt"
        class: DLBCL
        platform: rrbs
        sample_type: tissue
      - name: GSE171424
        reader: matrix
        reader_params: {chr_col: chr, base_col: pos, sep: ",", scale: fraction}
        file: raw/beta_EPIC.csv
        platform: array_epic
        sample_type: tissue
        samples: {FL002: FL, MZL001: MZL}

Paths are relative to the configuration file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd
import yaml

from .data import REQUIRED_COLUMNS
from .io.formats import write
from .io.readers import READERS, MultiSampleReader, Reader, make_reader
from .preprocessing.qc import filter_by_coverage, filter_by_panel, read_panel

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    """Description of one dataset in the configuration file."""

    name: str
    reader: str
    platform: str
    sample_type: str
    folder: Path | None = None
    pattern: str = "*"
    file: Path | None = None
    sample_class: str | None = None
    samples: dict[str, str] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)
    reader_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, base: Path) -> "DatasetSpec":
        known = {"name", "reader", "platform", "sample_type", "folder", "pattern", "file",
                 "class", "samples", "exclude", "reader_params"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Dataset {d.get('name')!r}: unknown keys {sorted(unknown)}")
        for key in ("name", "reader", "platform", "sample_type"):
            if key not in d:
                raise ValueError(f"Dataset {d.get('name')!r}: missing {key!r}")
        if d["reader"] not in READERS:
            raise ValueError(f"Dataset {d['name']!r}: unknown reader {d['reader']!r}")
        if ("folder" in d) == ("file" in d):
            raise ValueError(f"Dataset {d['name']!r}: give exactly one of 'folder' and 'file'")
        if "folder" in d and "class" not in d:
            raise ValueError(f"Dataset {d['name']!r}: a folder dataset needs a 'class'")
        if "file" in d and not d.get("samples"):
            raise ValueError(f"Dataset {d['name']!r}: a matrix dataset needs 'samples'")
        return cls(name=d["name"], reader=d["reader"], platform=d["platform"],
                   sample_type=d["sample_type"],
                   folder=base / d["folder"] if "folder" in d else None,
                   pattern=d.get("pattern", "*"),
                   file=base / d["file"] if "file" in d else None,
                   sample_class=d.get("class"), samples=dict(d.get("samples") or {}),
                   exclude=list(d.get("exclude") or []),
                   reader_params=dict(d.get("reader_params") or {}))

    def missing_paths(self) -> list[Path]:
        """Input paths of this dataset that do not exist."""
        path = self.folder if self.folder is not None else self.file
        return [] if path.exists() else [path]

    def iter_samples(self) -> Iterator[tuple[str, str, pd.DataFrame]]:
        """Yield ``(sample name, class, dataframe)`` for every sample of the dataset."""
        reader = make_reader(self.reader, **self.reader_params)
        if self.folder is not None:
            if not isinstance(reader, Reader):
                raise ValueError(f"Reader {self.reader!r} cannot read single-sample files")
            for path in sorted(p for p in self.folder.glob(self.pattern) if p.is_file()):
                if path.stem not in self.exclude:
                    yield path.stem, self.sample_class, reader.read(path)
        else:
            if not isinstance(reader, MultiSampleReader):
                raise ValueError(f"Reader {self.reader!r} cannot read multi-sample files")
            columns = [c for c in self.samples if c not in self.exclude]
            for col, df in reader.read_samples(self.file, columns).items():
                yield col, self.samples[col], df


@dataclass
class ProcessingConfig:
    output: Path
    datasets: list[DatasetSpec]
    panel: Path | None = None
    min_coverage: int = 10

    @classmethod
    def load(cls, path: str | Path) -> "ProcessingConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        base = path.parent
        names = [d.get("name") for d in raw.get("datasets", [])]
        if len(names) != len(set(names)):
            raise ValueError("Dataset names must be unique")
        return cls(output=base / raw.get("output", "processed"),
                   datasets=[DatasetSpec.from_dict(d, base) for d in raw.get("datasets", [])],
                   panel=base / raw["panel"] if raw.get("panel") else None,
                   min_coverage=int(raw.get("min_coverage", 10)))

    def check_paths(self) -> None:
        """Raise ``FileNotFoundError`` listing every input path that does not exist."""
        missing = [p for spec in self.datasets for p in spec.missing_paths()]
        if self.panel is not None and not self.panel.exists():
            missing.insert(0, self.panel)
        if missing:
            lines = "\n".join(f"  - {p.resolve()}" for p in missing)
            raise FileNotFoundError(f"These paths from the configuration do not exist:\n{lines}")


def process(config: ProcessingConfig, overwrite: bool = False) -> pd.DataFrame:
    """Convert and filter all samples; write them and ``samples.csv`` to ``config.output``.

    Samples whose output file already exists are not processed again unless
    ``overwrite`` is set. Returns the metadata table.
    """
    config.check_paths()
    panel = read_panel(config.panel) if config.panel else None
    rows = []
    for spec in config.datasets:
        out_dir = config.output / spec.name
        for sample, label, df in spec.iter_samples():
            out_file = out_dir / f"{sample}.tsv"
            if overwrite or not out_file.exists():
                n_raw = len(df)
                if panel is not None:
                    df = filter_by_panel(df, panel)
                df = filter_by_coverage(df, config.min_coverage)
                write(df, out_file)
                logger.info("%s/%s: %d -> %d CpGs", spec.name, sample, n_raw, len(df))
            rows.append({"sample_id": f"{spec.name}/{sample}",
                         "file": str(out_file.relative_to(config.output)).replace("\\", "/"),
                         "class": label, "platform": spec.platform, "dataset": spec.name,
                         "sample_type": spec.sample_type})
    meta = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    config.output.mkdir(parents=True, exist_ok=True)
    meta.to_csv(config.output / "samples.csv", index=False)
    return meta
