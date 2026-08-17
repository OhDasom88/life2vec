from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .paths import load_yaml, CONFIG_DIR, write_json
from .time_keys import parse_dat_time, assign_zone_ids
from .vocab import value_to_token, encode_tokens


def regression_example_id(scope: str, dat: int, zone: int, target_timestamp: str) -> str:
    raw = f"{scope}|{dat}|{zone}|{target_timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def build_regression_anchors(
    train_y_path,
    env_5min: pd.DataFrame,
    feature_cols: list[str],
    vocab: dict[str, Any],
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
    *,
    cube_index: pd.DataFrame | None = None,
    scope: str = "inductive",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    align = load_yaml(CONFIG_DIR / "cube_env_alignment.yaml")["cube_env_alignment"]
    lookback = int(cfg["regression"]["lookback_minutes"])
    max_cubes = int(cfg["regression"]["max_cube_events"])
    emb = int(
        max(
            cfg["embargo"]["regression_lookback"],
            cfg["embargo"]["narrative_maximum_lookback"],
            cfg["embargo"]["cube_event_window"],
            cfg["embargo"]["derived_feature_dependency_horizon"],
        )
    )
    y = pd.read_csv(train_y_path)
    meta = parse_dat_time(y["time"])
    y = pd.concat([y.reset_index(drop=True), meta], axis=1)
    y = assign_zone_ids(y, cfg, split="train")
    y = y.drop_duplicates(subset=["dat", "zone_id", "minutes_of_day"], keep="first")

    # exclude train anchors that fall in embargo before valid/holdout boundary
    valid_dats = split_manifest.get("valid_dats", [])
    holdout_dats = split_manifest.get("holdout_dats", [])
    boundary = min(valid_dats) if valid_dats else None

    rows = []
    skipped = {"no_x": 0, "all_missing_y": 0, "embargo": 0, "no_split": 0}
    targets = cfg["target_columns"]
    for _, yr in y.iterrows():
        dat = int(yr["dat"])
        zone = int(yr["zone_id"])
        minute = int(yr["minutes_of_day"])
        split_id = split_manifest["membership_by_dat"].get(str(dat))
        if split_id is None:
            skipped["no_split"] += 1
            continue
        # embargo: if train DAT is immediately before boundary day, exclude last emb minutes
        if (
            split_id == "train"
            and boundary is not None
            and dat == boundary - 1
            and minute >= 1440 - emb
        ):
            skipped["embargo"] += 1
            continue
        yvals = {t: yr[t] for t in targets}
        if all(pd.isna(v) for v in yvals.values()):
            skipped["all_missing_y"] += 1
            continue
        g = env_5min[(env_5min["dat"] == dat) & (env_5min["zone_id"] == zone)]
        ctx = g[(g["minutes_of_day"] >= minute - lookback) & (g["minutes_of_day"] <= minute)]
        if ctx.empty:
            skipped["no_x"] += 1
            continue
        tokens = ["[CLS]", f"ZONE_{'ABCD'[zone]}", "[VIEW_EVENT_WINDOW]", "[SEP]"]
        event_ids = []
        for _, r in ctx.iterrows():
            tokens.append("[EVENT_START]")
            for c in feature_cols:
                tokens.append(value_to_token(c, float(r[c]), vocab))
            tokens.append("[EVENT_SEP]")
            event_ids.append(f"ENV:{dat}:{zone}:{int(r['minutes_of_day'])}")
        cube_ids = []
        if cube_index is not None and len(cube_index):
            # STRICT_PAST_ONLY: capture_end_minute < target minute (same DAT), same zone
            cands = cube_index[
                (cube_index["dat"] == dat)
                & (cube_index["env_zone_id"] == zone)
                & (cube_index["capture_end_minute"] < minute)
            ]
            if len(cands):
                cands = cands.assign(dist=minute - cands["capture_end_minute"]).sort_values("dist")
                # also enforce max backward distance for env alignment quality when attaching
                cands = cands[cands["dist"] <= max(lookback, int(align["max_backward_distance_minutes"]))]
                for _, c in cands.head(max_cubes).iterrows():
                    cube_ids.append(c["cube_id"])
                    for i in range(10):
                        tokens.append(f"CUBE_BAND_{i}")
                    tokens.append("CUBE_POOL")
                    event_ids.append(f"CUBE:{c['cube_id']}")
        tokens.append("[EVENT_END]")
        ids = encode_tokens(tokens, vocab)
        ts = str(yr["time"])
        rows.append(
            {
                "regression_example_id": regression_example_id(scope, dat, zone, ts),
                "dat": dat,
                "zone_id": zone,
                "target_timestamp": ts,
                "target_minute": minute,
                "t_ord": int(yr["t_ord"]),
                "split_id": split_id,
                "token_ids": ids,
                "n_tokens": len(ids),
                "event_ids": event_ids,
                "cube_ids": cube_ids,
                "has_cube": bool(cube_ids),
                "soil_moisture": yvals["soil_moisture"],
                "soil_ec": yvals["soil_ec"],
                "soil_temp": yvals["soil_temp"],
                "target_mask": [
                    int(not pd.isna(yvals["soil_moisture"])),
                    int(not pd.isna(yvals["soil_ec"])),
                    int(not pd.isna(yvals["soil_temp"])),
                ],
                "source_window_id": f"train:DAT{dat}:Z{zone}",
                "raw_context_row_ids": ctx["raw_row_id"].tolist() if "raw_row_id" in ctx.columns else [],
            }
        )
    df = pd.DataFrame(rows)
    report = {
        "n_examples": int(len(df)),
        "skipped": skipped,
        "by_split": df["split_id"].value_counts().to_dict() if len(df) else {},
        "cube_coverage": float(df["has_cube"].mean()) if len(df) else 0.0,
        "past_policy": "STRICT_PAST_ONLY",
        "lookback_minutes": lookback,
    }
    return df, report
