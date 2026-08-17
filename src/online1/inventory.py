from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import Online1Paths, sha256_file, write_json
from .time_keys import parse_dat_time, assign_zone_ids


CUBE_NAME_RE = re.compile(r"(?P<zone>\d+)_(?P<farm>DAT\d+)_(?P<time>\d{6})")


def inventory_env_csv(path: Path, split: str, cfg: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_csv(path)
    meta = parse_dat_time(df["time"])
    df2 = pd.concat([df.reset_index(drop=True), meta], axis=1)
    df2 = assign_zone_ids(df2, cfg, split=split)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dat_min": int(df2["dat"].min()),
        "dat_max": int(df2["dat"].max()),
        "n_dat": int(df2["dat"].nunique()),
        "zone_counts": {str(k): int(v) for k, v in df2["zone_id"].value_counts().sort_index().items()},
        "duplicate_times": int(df["time"].duplicated().sum()),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
    }


def inventory_cubes(ms_root: Path) -> dict[str, Any]:
    hdrs = sorted(ms_root.rglob("cube.hdr"))
    records = []
    malformed = []
    zone_counts: dict[str, int] = {}
    for hdr in hdrs:
        sample = hdr.parent
        m = CUBE_NAME_RE.match(sample.name)
        raw = sample / "cube.raw"
        rec = {
            "cube_id": sample.name,
            "path": str(sample),
            "hdr": str(hdr),
            "raw": str(raw),
            "dat_dir": sample.parent.name,
            "has_raw": raw.exists(),
            "hdr_sha256": sha256_file(hdr) if hdr.exists() else None,
            "raw_bytes": raw.stat().st_size if raw.exists() else 0,
        }
        if not m or not raw.exists():
            malformed.append(rec)
            continue
        zone = m.group("zone")
        farm = m.group("farm")
        t = m.group("time")
        rec.update(
            {
                "cube_zone": int(zone),
                "farm": farm,
                "capture_hhmmss": t,
                "capture_time": f"{t[:2]}:{t[2:4]}:{t[4:6]}",
                "dat": int(farm[3:]),
            }
        )
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        records.append(rec)
    return {
        "root": str(ms_root),
        "n_hdr": len(hdrs),
        "n_valid": len(records),
        "n_malformed": len(malformed),
        "zone_counts": zone_counts,
        "records": records,
        "malformed": malformed,
    }


def build_raw_inventory(paths: Online1Paths, cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    train_x = inventory_env_csv(paths.train_x, "train", cfg)
    train_y = inventory_env_csv(paths.train_y, "train", cfg)
    test_x = inventory_env_csv(paths.test_x, "test", cfg)
    train_ms = inventory_cubes(paths.train_ms)
    test_ms = inventory_cubes(paths.test_ms)
    inv = {
        "train_X": {k: v for k, v in train_x.items() if k != "null_counts"} | {"null_counts": train_x["null_counts"]},
        "train_y": train_y,
        "test_X": test_x,
        "train_ms": {
            "root": train_ms["root"],
            "n_hdr": train_ms["n_hdr"],
            "n_valid": train_ms["n_valid"],
            "n_malformed": train_ms["n_malformed"],
            "zone_counts": train_ms["zone_counts"],
        },
        "test_ms": {
            "root": test_ms["root"],
            "n_hdr": test_ms["n_hdr"],
            "n_valid": test_ms["n_valid"],
            "n_malformed": test_ms["n_malformed"],
            "zone_counts": test_ms["zone_counts"],
        },
        "cube_record_index": {
            "train": [
                {
                    "cube_id": r["cube_id"],
                    "dat": r["dat"],
                    "cube_zone": r["cube_zone"],
                    "capture_time": r["capture_time"],
                    "path": r["path"],
                    "raw_bytes": r["raw_bytes"],
                }
                for r in train_ms["records"]
            ],
            "test": [
                {
                    "cube_id": r["cube_id"],
                    "dat": r["dat"],
                    "cube_zone": r["cube_zone"],
                    "capture_time": r["capture_time"],
                    "path": r["path"],
                    "raw_bytes": r["raw_bytes"],
                }
                for r in test_ms["records"]
            ],
        },
    }
    write_json(out_dir / "raw_inventory.json", inv)
    # keep full cube records separately for later phases
    write_json(out_dir / "cube_inventory_full.json", {"train": train_ms, "test": test_ms})
    return inv
