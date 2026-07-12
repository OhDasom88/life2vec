#!/usr/bin/env python3
"""Build non-destructive shadow probe folds (farm / time) without altering train corpus."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def bucket(key: str, seed: int = 2023) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def main() -> None:
    out = Path("outputs/online2/v2_build")
    seq = pd.read_parquet(
        out / "sequences_v2.parquet",
        columns=["sequence_id", "farm_ids", "source_window_signature", "split_group_id"],
    )
    seq["farms"] = seq["farm_ids"].map(json.loads)
    # Connected components over farms that co-occur in sequences
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for farms in seq["farms"]:
        farms = [str(f) for f in farms]
        for a in farms:
            find(a)
        for a, b in zip(farms, farms[1:]):
            union(a, b)

    seq["farm_component"] = seq["farms"].map(lambda fs: find(str(fs[0])) if fs else "NONE")
    seq["farm_held_out_fold"] = seq["farm_component"].map(bucket)
    seq["time_window_held_out_fold"] = seq["source_window_signature"].map(bucket)

    # Event overlap within farm_held_out probe only (diagnostic; may be >0 by design of reuse)
    report = {
        "note": "Probe folds only. Final transductive train corpus includes all public sequences.",
        "farm_held_out_counts": seq["farm_held_out_fold"].value_counts().to_dict(),
        "time_window_held_out_counts": seq["time_window_held_out_fold"].value_counts().to_dict(),
        "n_farm_components": int(seq["farm_component"].nunique()),
        "sequence_count": int(len(seq)),
    }
    seq[
        [
            "sequence_id",
            "farm_component",
            "farm_held_out_fold",
            "time_window_held_out_fold",
        ]
    ].to_parquet(out / "shadow_probe_folds_v2.parquet", index=False)
    (out / "shadow_probe_folds_v2.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
