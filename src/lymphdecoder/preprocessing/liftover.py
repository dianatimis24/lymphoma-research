"""Conversion of CpG coordinates between genome builds (e.g. hg19 -> hg38).

Uses the UCSC chain files (``hg19ToHg38.over.chain.gz``). A chain is a list of
aligned blocks between the source and the target genome; a position inside a
block is moved by the offset of the block. Positions outside every block, or
in blocks that map to the reverse strand or to another chromosome, are dropped,
as are positions that end up on the same target CpG as another position.
"""

from __future__ import annotations

import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from lymphdecoder.io.formats import finalize


class Chain:
    """Aligned blocks of a chain file, indexed for fast lookups per chromosome.

    Only blocks that map to the same chromosome on the + strand are kept. Where
    blocks of several chains overlap, the block of the chain with the higher
    score wins.
    """

    def __init__(self, blocks: pd.DataFrame):
        self.blocks = {}
        for chrom, group in blocks.groupby("chr", sort=False):
            k_start = np.empty(0, dtype=np.int64)
            k_end = np.empty(0, dtype=np.int64)
            k_off = np.empty(0, dtype=np.int64)
            chains = group.groupby("chain", sort=False)
            order = group.groupby("chain")["score"].first().sort_values(ascending=False).index
            for cid in order:
                c = chains.get_group(cid)
                s = c["start"].to_numpy(dtype=np.int64)
                e = c["end"].to_numpy(dtype=np.int64)
                if len(k_start):
                    i = np.searchsorted(k_start, e, side="left") - 1
                    clash = (i >= 0) & (k_end[np.maximum(i, 0)] > s)
                    s, e = s[~clash], e[~clash]
                    off = c["offset"].to_numpy(dtype=np.int64)[~clash]
                else:
                    off = c["offset"].to_numpy(dtype=np.int64)
                if len(s) == 0:
                    continue
                k_start = np.concatenate([k_start, s])
                k_end = np.concatenate([k_end, e])
                k_off = np.concatenate([k_off, off])
                idx = np.argsort(k_start, kind="stable")
                k_start, k_end, k_off = k_start[idx], k_end[idx], k_off[idx]
            self.blocks[chrom] = (k_start, k_end, k_off)

    @classmethod
    def load(cls, path: str | Path) -> "Chain":
        return _load_chain(str(Path(path).resolve()))

    def map_positions(self, chrom: str, positions: np.ndarray) -> np.ndarray:
        """Map 1-based positions; unmapped positions become -1."""
        out = np.full(len(positions), -1, dtype=np.int64)
        if chrom not in self.blocks:
            return out
        starts, ends, offsets = self.blocks[chrom]
        zero = np.asarray(positions, dtype=np.int64) - 1
        idx = np.searchsorted(starts, zero, side="right") - 1
        valid = idx >= 0
        inside = np.zeros(len(zero), dtype=bool)
        inside[valid] = zero[valid] < ends[idx[valid]]
        out[inside] = zero[inside] + offsets[idx[inside]] + 1
        return out


def parse_chain(lines) -> pd.DataFrame:
    """Blocks of a chain file as a dataframe (chr, start, end, offset, score, chain), 0-based."""
    rows = []
    chain_id = -1
    t_name = q_name = None
    t_pos = q_pos = 0
    score = 0.0
    usable = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if fields[0] == "chain":
            score = float(fields[1])
            chain_id += 1
            t_name, t_strand, t_pos = fields[2], fields[4], int(fields[5])
            q_name, q_strand, q_pos = fields[7], fields[9], int(fields[10])
            usable = t_strand == "+" and q_strand == "+" and t_name == q_name
            continue
        size = int(fields[0])
        if usable:
            rows.append((t_name, t_pos, t_pos + size, q_pos - t_pos, score, chain_id))
        if len(fields) == 3:
            t_pos += size + int(fields[1])
            q_pos += size + int(fields[2])
    return pd.DataFrame(rows, columns=["chr", "start", "end", "offset", "score", "chain"])


@lru_cache(maxsize=4)
def _load_chain(path: str) -> Chain:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return Chain(parse_chain(f))


def lift(df: pd.DataFrame, chain: Chain) -> pd.DataFrame:
    """Move a common-format sample to the target build of ``chain``."""
    parts = []
    for chrom, group in df.groupby("chr", sort=False):
        mapped = chain.map_positions(chrom, group["base"].to_numpy())
        keep = mapped > 0
        part = group[keep].copy()
        part["base"] = mapped[keep]
        parts.append(part)
    if not parts:
        return finalize(df.iloc[0:0])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["chr", "base", "strand"], keep=False)
    return finalize(out)
