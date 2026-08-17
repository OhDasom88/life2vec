from __future__ import annotations

from typing import Any

import pandas as pd

from .paths import sha256_json, write_json
from .time_keys import parse_dat_time, assign_zone_ids


def embargo_minutes(cfg: dict[str, Any]) -> int:
    e = cfg.get("embargo", {})
    return int(
        max(
            e.get("regression_lookback", 60),
            e.get("narrative_maximum_lookback", 180),
            e.get("cube_event_window", 30),
            e.get("derived_feature_dependency_horizon", 60),
        )
    )


def build_development_split(cfg: dict[str, Any], out_path) -> dict[str, Any]:
    """DAT-block forward split with embargo metadata.

    Membership is assigned at DAT level first; all derivatives inherit split_id.
    """
    scfg = cfg["splits"]["development"]
    train = [int(x) for x in scfg["train_dats"]]
    valid = [int(x) for x in scfg["valid_dats"]]
    holdout = [int(x) for x in scfg["holdout_dats"]]
    emb = embargo_minutes(cfg)
    # embargo applies at boundary between train→valid and valid→holdout
    boundary_tv = min(valid) if valid else None
    boundary_vh = min(holdout) if holdout else None
    manifest = {
        "split_name": "development_forward_dat",
        "order": "split_first_raw_temporal",
        "train_dats": train,
        "valid_dats": valid,
        "holdout_dats": holdout,
        "embargo_minutes": emb,
        "boundaries": {
            "train_valid_dat": boundary_tv,
            "valid_holdout_dat": boundary_vh,
        },
        "membership_by_dat": {
            **{str(d): "train" for d in train},
            **{str(d): "valid" for d in valid},
            **{str(d): "holdout" for d in holdout},
        },
        "rules": {
            "raw_row_overlap": 0,
            "cube_overlap": 0,
            "source_window_overlap": 0,
            "timestamp_interval_overlap": 0,
            "train_anchor_in_embargo_before_valid": "exclude",
        },
    }
    manifest["checksum"] = sha256_json(manifest)
    write_json(out_path, manifest)
    return manifest


def attach_split_id(df: pd.DataFrame, split_manifest: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    mem = {int(k): v for k, v in split_manifest["membership_by_dat"].items()}
    out["split_id"] = out["dat"].map(mem)
    if out["split_id"].isna().any():
        missing = sorted(out.loc[out["split_id"].isna(), "dat"].unique().tolist())
        raise ValueError(f"DAT not in split membership: {missing}")
    return out


def verify_no_overlap(train_ids, valid_ids, name: str) -> dict[str, Any]:
    inter = set(train_ids) & set(valid_ids)
    return {
        "name": name,
        "train_n": len(set(train_ids)),
        "valid_n": len(set(valid_ids)),
        "overlap_n": len(inter),
        "overlap_examples": sorted(list(inter))[:20],
        "pass": len(inter) == 0,
    }


def load_env_with_split(
    csv_path,
    cfg: dict[str, Any],
    *,
    split_name: str,
    split_manifest: dict[str, Any],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    meta = parse_dat_time(df["time"])
    df = pd.concat([df.reset_index(drop=True), meta], axis=1)
    df = assign_zone_ids(df, cfg, split=split_name)
    df["raw_row_id"] = [
        f"{split_name}:{dat}:{mod}:{i}"
        for i, (dat, mod) in enumerate(zip(df["dat"], df["minutes_of_day"]))
    ]
    df["source_window_id"] = [
        f"{split_name}:DAT{dat}:Z{z}"
        for dat, z in zip(df["dat"], df["zone_id"])
    ]
    df = attach_split_id(df, split_manifest)
    return df
