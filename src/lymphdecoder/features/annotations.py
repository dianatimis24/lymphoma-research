"""Annotated genomic regions: CpG islands and gene promoters.

Supported inputs are the UCSC tables ``cpgIslandExt.txt.gz`` and
``ncbiRefSeq.txt.gz`` / ``refGene.txt.gz`` (hg38), and plain BED files. All
returned regions are 1-based and closed, with columns ``chr``, ``start``, ``end``
and ``name``, sorted and without overlaps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lymphdecoder.io.formats import CHROMOSOME_ORDER

UCSC_CPG_ISLAND_COLUMNS = ["bin", "chrom", "chromStart", "chromEnd", "name", "length",
                           "cpgNum", "gcNum", "perCpg", "perGc", "obsExp"]
UCSC_GENE_COLUMNS = ["bin", "name", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd",
                     "exonCount", "exonStarts", "exonEnds", "score", "name2", "cdsStartStat",
                     "cdsEndStat", "exonFrames"]


def _standard(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["chr"].isin(CHROMOSOME_ORDER)].copy()
    df["chr"] = pd.Categorical(df["chr"], categories=CHROMOSOME_ORDER, ordered=True)
    df = df.sort_values(["chr", "start", "end"])
    df["chr"] = df["chr"].astype(str)
    return df.reset_index(drop=True)


def merge_overlapping(regions: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping 1-based closed regions; names of merged regions are joined."""
    regions = _standard(regions)
    merged = []
    for chrom, group in regions.groupby("chr", sort=False):
        cur = None
        for start, end, name in zip(group["start"], group["end"], group["name"]):
            if cur is not None and start <= cur[1]:
                cur[1] = max(cur[1], end)
                if name not in cur[2]:
                    cur[2].append(name)
            else:
                if cur is not None:
                    merged.append((chrom, cur[0], cur[1], ";".join(cur[2])))
                cur = [start, end, [name]]
        if cur is not None:
            merged.append((chrom, cur[0], cur[1], ";".join(cur[2])))
    return _standard(pd.DataFrame(merged, columns=["chr", "start", "end", "name"]))


def read_bed_regions(path: str | Path) -> pd.DataFrame:
    """Plain BED file (0-based start); the 4th column, if present, is the name."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    names = df[3].astype(str) if df.shape[1] > 3 else (
        df[0].astype(str) + ":" + df[1].astype(str))
    out = pd.DataFrame({"chr": df[0], "start": df[1] + 1, "end": df[2], "name": names})
    return merge_overlapping(out)


def read_cpg_islands(path: str | Path) -> pd.DataFrame:
    """CpG islands from the UCSC ``cpgIslandExt`` table or from a BED file."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if df.shape[1] != len(UCSC_CPG_ISLAND_COLUMNS):
        return read_bed_regions(path)
    df.columns = UCSC_CPG_ISLAND_COLUMNS
    out = pd.DataFrame({"chr": df["chrom"], "start": df["chromStart"] + 1,
                        "end": df["chromEnd"],
                        "name": df["chrom"] + ":" + (df["chromStart"] + 1).astype(str)})
    return merge_overlapping(out)


def read_genes(path: str | Path) -> pd.DataFrame:
    """Transcripts from the UCSC ``ncbiRefSeq`` or ``refGene`` table.

    Returns columns ``chr``, ``strand``, ``tss`` (1-based) and ``gene``.
    """
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    if df.shape[1] != len(UCSC_GENE_COLUMNS):
        raise ValueError(f"Expected the {len(UCSC_GENE_COLUMNS)} columns of a UCSC gene table")
    df.columns = UCSC_GENE_COLUMNS
    tss = df["txStart"].where(df["strand"] == "-", df["txStart"] + 1)
    tss = tss.where(df["strand"] == "+", df["txEnd"])
    return pd.DataFrame({"chr": df["chrom"], "strand": df["strand"], "tss": tss,
                         "gene": df["name2"]})


def promoters(genes: pd.DataFrame, upstream: int = 1500, downstream: int = 500) -> pd.DataFrame:
    """Promoter regions around each transcription start site, strand-aware.

    On the + strand the promoter is ``[tss - upstream, tss + downstream - 1]``; on
    the - strand it is mirrored. Overlapping promoters are merged.
    """
    plus = genes["strand"] == "+"
    start = (genes["tss"] - upstream).where(plus, genes["tss"] - downstream + 1)
    end = (genes["tss"] + downstream - 1).where(plus, genes["tss"] + upstream)
    out = pd.DataFrame({"chr": genes["chr"], "start": start.clip(lower=1), "end": end,
                        "name": genes["gene"]})
    return merge_overlapping(out)


def read_promoters(path: str | Path, upstream: int = 1500, downstream: int = 500) -> pd.DataFrame:
    """Promoters from a UCSC gene table, or regions from a BED file."""
    try:
        return promoters(read_genes(path), upstream, downstream)
    except ValueError:
        return read_bed_regions(path)
