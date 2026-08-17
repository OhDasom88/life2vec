from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .lineage import build_env_lineage, encoder_columns, load_deny_config
from .time_keys import parse_dat_time, assign_zone_ids, format_dat_time


def normalize_actuators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c].astype(float) > 0).astype(float)
    return out


def load_clean_env(
    path,
    cfg: dict[str, Any],
    *,
    split: str,
) -> tuple[pd.DataFrame, list[str], list[dict]]:
    deny = load_deny_config()
    raw = pd.read_csv(path)
    meta = parse_dat_time(raw["time"])
    df = pd.concat([raw.reset_index(drop=True), meta], axis=1)
    df = assign_zone_ids(df, cfg, split=split)
    df = normalize_actuators(df, cfg.get("binary_actuator_cols", []))
    drop = set(cfg.get("drop_cols", []))
    lineages = build_env_lineage(list(raw.columns), deny)
    keep = [c for c in encoder_columns(lineages) if c not in drop]
    # large gap detection: expected 1-minute grid within DAT
    df = df.sort_values(["dat", "minutes_of_day"]).reset_index(drop=True)
    df["raw_row_id"] = [f"{split}:{r.dat}:{r.minutes_of_day}:{i}" for i, r in df.iterrows()]
    df["source_window_id"] = [f"{split}:DAT{r.dat}:Z{r.zone_id}" for _, r in df.iterrows()]
    gaps = []
    for (dat, zone), g in df.groupby(["dat", "zone_id"], sort=False):
        mods = g["minutes_of_day"].to_numpy()
        d = np.diff(mods)
        bad = np.where(d > 1)[0]
        for idx in bad:
            gaps.append(
                {
                    "dat": int(dat),
                    "zone_id": int(zone),
                    "gap_after_minute": int(mods[idx]),
                    "gap_size": int(d[idx]),
                }
            )
    df["time_std"] = [format_dat_time(d, m) for d, m in zip(df["dat"], df["minutes_of_day"])]
    return df, keep, [l.to_dict() for l in lineages]


def aggregate_to_5min(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df.copy()
    x["bucket"] = (x["minutes_of_day"] // 5) * 5
    agg = {c: "mean" for c in feature_cols if c in x.columns}
    for c in ["fcu_fan", "fcu_pump", "circ_fan", "co2_supply", "fogging", "rainfall"]:
        if c in x.columns:
            agg[c] = "max"
    g = (
        x.groupby(["dat", "zone_id", "bucket"], sort=True)
        .agg(agg)
        .reset_index()
        .rename(columns={"bucket": "minutes_of_day"})
    )
    g["t_ord"] = g["dat"] * 1440 + g["minutes_of_day"]
    g["time"] = [format_dat_time(d, m) for d, m in zip(g["dat"], g["minutes_of_day"])]
    g["source_window_id"] = [f"agg5:DAT{d}:Z{z}" for d, z in zip(g["dat"], g["zone_id"])]
    return g.sort_values("t_ord").reset_index(drop=True)
