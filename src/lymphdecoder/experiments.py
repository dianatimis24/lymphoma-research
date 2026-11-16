"""Experiments of the case study (E0-E5), driven by one configuration file.

    E0  reproduction of the baseline on a fixed train/test split
    E1  checks of the converted data (per-sample statistics, suspicious samples)
    E2  overlap of the regions measured by the different platforms
    E3  natural grouping of the samples: by class or by platform / sample type?
    E4  comparison of feature extraction strategies (DMRs on bins, CpG islands, promoters)
    E5  leakage-free versus leaky evaluation

Every experiment writes its tables, figures and a ``run.json`` (configuration,
git commit, package versions) to ``<results>/<experiment>/``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import load_metadata, merge_small_classes
from .evaluation import cv, exploration, plots
from .features import annotations
from .features.bin_store import BinStore, FeatureStore, RegionStore
from .features.dmr import make_dmr_finder
from .io import formats
from .utils.reproducibility import save_run, set_seed

logger = logging.getLogger(__name__)

EXPERIMENTS = ("e0", "e1", "e2", "e3", "e4", "e5")


class Experiments:
    """Shared state of the experiments: samples, labels, region stores and settings."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        base = self.config_path.parent
        self.raw = yaml.safe_load(self.config_path.read_text())
        self.base = base
        self.results = base / self.raw.get("results", "results")
        self.cache = base / self.raw.get("cache", "cache")
        self.meta = load_metadata(base / self.raw["metadata"])
        classes = self.raw.get("classes")
        if classes:
            self.meta = self.meta[self.meta["class"].isin(classes)]
        labels = self.meta["class"]
        min_size = self.raw.get("min_class_size")
        if min_size:
            labels = merge_small_classes(labels, int(min_size))
        self.labels = labels
        self.platforms = self.meta["platform"]
        self.pipeline = cv.PipelineConfig.from_dict(self.raw.get("pipeline"))
        self.cv = dict(self.raw.get("cv") or {})
        self.region_specs = dict(self.raw.get("regions") or {"bins": {"type": "bins"}})
        self._stores: dict[str, FeatureStore] = {}
        set_seed(self.pipeline.seed)

    def load_sample(self, sample_id: str) -> pd.DataFrame:
        return formats.read(self.meta.loc[sample_id, "file"])

    def store(self, name: str) -> FeatureStore:
        """Region store ``name`` from the configuration, built on first use."""
        if name in self._stores:
            return self._stores[name]
        if name not in self.region_specs:
            raise ValueError(f"Unknown region set {name!r}, expected one of {list(self.region_specs)}")
        spec = dict(self.region_specs[name])
        spec.pop("min_cpgs", None)
        kind = spec.pop("type", "bins")
        folder = self.cache / name
        if kind == "bins":
            store = BinStore(folder, int(spec.get("bin_size", self.pipeline.bin_size)))
        elif kind == "cpg_islands":
            store = RegionStore(folder, annotations.read_cpg_islands(self.base / spec["file"]))
        elif kind == "promoters":
            regions = annotations.read_promoters(self.base / spec["file"],
                                                 int(spec.get("upstream", 1500)),
                                                 int(spec.get("downstream", 500)))
            store = RegionStore(folder, regions)
        elif kind == "bed":
            store = RegionStore(folder, annotations.read_bed_regions(self.base / spec["file"]))
        else:
            raise ValueError(f"Unknown region type {kind!r}")
        logger.info("Building region store %r for %d samples", name, len(self.meta))
        store.build(self.meta.index, self.load_sample)
        self._stores[name] = store
        return store

    def min_cpgs(self, regions: str):
        """Density threshold for a region set: its own ``min_cpgs`` or the pipeline's."""
        spec = self.region_specs.get(regions) or {}
        return spec.get("min_cpgs", self.pipeline.min_cpgs)

    def section(self, name: str) -> dict:
        return dict(self.raw.get(name) or {})

    def out(self, name: str) -> Path:
        folder = self.results / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def crossval(self, store, cfg, leaky=False):
        return cv.cross_validate(store, self.labels, cfg,
                                 n_splits=int(self.cv.get("n_splits", 5)),
                                 n_repeats=int(self.cv.get("n_repeats", 1)),
                                 grid=self.cv.get("grid"),
                                 inner_splits=int(self.cv.get("inner_splits", 3)),
                                 leaky=leaky, platforms=self.platforms)


def match_test_samples(entries: list[str], sample_ids) -> list[str]:
    """Map entries of a test-set list (file names or ids) to sample ids."""
    by_stem = {}
    for sid in sample_ids:
        by_stem.setdefault(sid.split("/")[-1], []).append(sid)
    matched = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        if entry in sample_ids:
            matched.append(entry)
            continue
        stem = Path(entry).stem
        matched.extend(by_stem.get(stem, []))
    return list(dict.fromkeys(matched))


def _save_cv(exp: Experiments, name: str, result, cfg, title: str, extra=None) -> dict:
    folder = exp.out(name)
    save_run(folder, cfg.to_dict(), result, extra)
    plots.plot_confusion(result.confusion(), folder / "confusion_matrix.png", title)
    return result.summary()


def run_e0(exp: Experiments) -> dict:
    """E0: the pipeline on the fixed train/test split of the baseline."""
    section = exp.section("e0")
    entries = (exp.base / section["test_set"]).read_text().splitlines()
    test_ids = match_test_samples(entries, list(exp.labels.index))
    logger.info("E0: %d of %d test entries matched", len(test_ids), len([e for e in entries if e.strip()]))
    regions = section.get("regions", "bins")
    cfg = replace(exp.pipeline, min_cpgs=exp.min_cpgs(regions))
    result = cv.fixed_split(exp.store(regions), exp.labels, test_ids, cfg, exp.platforms)
    baseline = section.get("baseline") or {}
    summary = _save_cv(exp, "e0", result, cfg, "E0: fixed split",
                       {"baseline": baseline, "n_test": len(test_ids)})
    if baseline:
        rows = [{"metric": k, "baseline": v, "this_work": summary.get(f"{k}_mean")}
                for k, v in baseline.items()]
        pd.DataFrame(rows).to_csv(exp.out("e0") / "comparison_with_baseline.csv", index=False)
    return summary


def run_e1(exp: Experiments) -> pd.DataFrame:
    """E1: statistics of every converted sample and the samples that look suspicious."""
    folder = exp.out("e1")
    summary = exploration.sample_summary(exp.meta.index, exp.load_sample)
    summary = summary.join(exp.meta[["class", "platform", "dataset", "sample_type"]])
    summary.to_csv(folder / "sample_summary.csv")
    by_platform = summary.groupby("platform")[["n_cpgs", "mean_methylation", "median_coverage"]].median()
    by_platform.to_csv(folder / "summary_by_platform.csv")
    flags = exploration.flag_samples(summary, int(exp.section("e1").get("min_cpgs", 1000)))
    flags.to_csv(folder / "flagged_samples.csv")
    save_run(folder, {"min_cpgs": int(exp.section("e1").get("min_cpgs", 1000))},
             extra={"n_samples": len(summary), "n_flagged": len(flags)})
    return summary


def run_e2(exp: Experiments) -> pd.DataFrame:
    """E2: regions measured by each platform and shared between platforms."""
    folder = exp.out("e2")
    section = exp.section("e2")
    rows = []
    for name in section.get("regions", list(exp.region_specs)):
        table = exploration.platform_overlap(exp.store(name), exp.platforms,
                                             exp.min_cpgs(name),
                                             exp.pipeline.min_sample_fraction)
        table.insert(0, "regions", name)
        rows.append(table)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(folder / "platform_overlap.csv", index=False)
    save_run(folder, {"min_cpgs": {n: exp.min_cpgs(n) for n in section.get("regions", list(exp.region_specs))},
                      "min_sample_fraction": exp.pipeline.min_sample_fraction})
    return result


def run_e3(exp: Experiments) -> pd.DataFrame:
    """E3: do the samples group by class, or by platform and sample type?"""
    folder = exp.out("e3")
    section = exp.section("e3")
    store = exp.store(section.get("regions", "bins"))
    finder = make_dmr_finder("all", min_cpgs=exp.min_cpgs(section.get("regions", "bins")),
                             min_sample_fraction=float(section.get("min_sample_fraction", 0.8)))
    keys = finder.find(store, exp.meta.index, exp.labels, exp.platforms).index.to_numpy()
    if len(keys) == 0:
        raise ValueError("No region is measured in enough samples; lower min_sample_fraction")
    X = exploration.top_variance(store.matrix(exp.meta.index, keys),
                                 int(section.get("top_variance", 5000)))
    groupings = {"class": exp.labels, "platform": exp.platforms,
                 "sample_type": exp.meta["sample_type"], "dataset": exp.meta["dataset"]}
    clusters, ari = exploration.grouping_agreement(X, groupings, exp.labels.nunique(),
                                                   seed=exp.pipeline.seed)
    ari.to_csv(folder / "ari.csv", index=False)
    method = section.get("method", "umap")
    coords = exploration.embed(X, method, exp.pipeline.seed)
    axis = "UMAP" if method == "umap" else "PC"
    for name in ("class", "platform", "sample_type"):
        plots.plot_embedding(coords, groupings[name].loc[X.index], folder / f"embedding_{name}.png",
                             f"Samples coloured by {name.replace('_', ' ')}",
                             f"{axis} 1", f"{axis} 2")
    pd.DataFrame(coords, index=X.index, columns=["x", "y"]).join(clusters).to_csv(
        folder / "embedding.csv")
    save_run(folder, {"regions": section.get("regions", "bins"), "n_regions": X.shape[1],
                      "method": method}, extra={"ari": ari.to_dict(orient="records")})
    return ari


def run_e4(exp: Experiments) -> pd.DataFrame:
    """E4: compare feature extraction strategies under the same evaluation protocol."""
    section = exp.section("e4")
    strategies = section.get("strategies") or [{"name": "DMR bins", "regions": "bins"}]
    rows = []
    for strategy in strategies:
        strategy = dict(strategy)
        name = strategy.pop("name")
        regions = strategy.pop("regions", "bins")
        strategy.setdefault("min_cpgs", exp.min_cpgs(regions))
        cfg = replace(exp.pipeline, **strategy)
        result = exp.crossval(exp.store(regions), cfg)
        slug = "e4/" + "".join(c if c.isalnum() else "_" for c in name.lower())
        summary = _save_cv(exp, slug, result, cfg, f"E4: {name}")
        rows.append({"strategy": name, "regions": regions, "dmr_method": cfg.dmr_method,
                     **summary})
    table = pd.DataFrame(rows)
    table.to_csv(exp.out("e4") / "comparison.csv", index=False)
    return table


def run_e5(exp: Experiments) -> pd.DataFrame:
    """E5: the same pipeline evaluated with and without information leakage."""
    section = exp.section("e5")
    regions = section.get("regions", "bins")
    store = exp.store(regions)
    cfg = replace(exp.pipeline, min_cpgs=exp.min_cpgs(regions))
    rows = []
    for leaky, name in ((False, "leakage_free"), (True, "leaky")):
        result = exp.crossval(store, cfg, leaky=leaky)
        summary = _save_cv(exp, f"e5/{name}", result, cfg,
                           "E5: " + name.replace("_", "-"))
        rows.append({"protocol": name, **summary})
    table = pd.DataFrame(rows)
    for metric in ("macro_f1_mean", "mcc_mean", "accuracy_mean"):
        table.loc[len(table) - 1, f"{metric}_overestimation"] = (
            table.loc[1, metric] - table.loc[0, metric])
    table.to_csv(exp.out("e5") / "comparison.csv", index=False)
    return table


RUNNERS = {"e0": run_e0, "e1": run_e1, "e2": run_e2, "e3": run_e3, "e4": run_e4, "e5": run_e5}


def run(config_path: str | Path, names: list[str]) -> dict:
    """Run the given experiments (``all`` runs E1-E5 and E0 if a test set is configured)."""
    exp = Experiments(config_path)
    if names == ["all"]:
        names = [n for n in EXPERIMENTS if n != "e0" or exp.section("e0").get("test_set")]
    out = {}
    for name in names:
        if name not in RUNNERS:
            raise ValueError(f"Unknown experiment {name!r}, expected one of {list(RUNNERS)}")
        logger.info("Running %s", name.upper())
        out[name] = RUNNERS[name](exp)
    return out
