from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .paths import load_yaml, CONFIG_DIR, write_json, sha256_json
from .vocab import value_to_token, encode_tokens, narrative_vocab_view


def _seq_id(parts: list[str]) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def _phase(minute: int) -> str:
    return "PHASE_DAY" if 480 <= minute <= 1020 else "PHASE_NIGHT"


def snapshot_tokens(row: pd.Series, feature_cols: list[str], vocab: dict, zone_id: int) -> list[str]:
    zlabel = "ABCD"[int(zone_id)]
    toks = ["[CLS]", f"ZONE_{zlabel}", "[VIEW_EVENT_WINDOW]", "[SEP]"]
    for c in feature_cols:
        toks.append(value_to_token(c, float(row[c]), vocab))
    toks.append("[EVENT_END]")
    return toks


def materialize_pretrain_sequences(
    env_5min: pd.DataFrame,
    feature_cols: list[str],
    vocab: dict[str, Any],
    split_manifest: dict[str, Any],
    *,
    max_per_narrative: int | None = None,
    cube_index: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """max_per_narrative=None (default) generates every matching window for every
    narrative — production scale. Pass an int to cap generation (used by the dev/smoke
    scripts to keep iteration fast)."""

    def _cap_hit(produced: int) -> bool:
        return max_per_narrative is not None and produced >= max_per_narrative

    catalog = load_yaml(CONFIG_DIR / "narrative_catalog.yaml")
    rows = []
    views = {}
    for narr in catalog["narratives"]:
        if narr["temporal_direction"] == "past_only" and narr["narrative_id"] == "N_REG_PAST_WINDOW":
            continue  # regression-only
        nid = narr["narrative_id"]
        views[nid] = narrative_vocab_view(vocab, narr["allowed_token_families"])
        if nid == "N_SNAP_W8":
            lookback = int(narr["lookback"])
            stride = int(narr["stride"])
            produced = 0
            for (dat, zone), g in env_5min.groupby(["dat", "zone_id"]):
                g = g.sort_values("t_ord")
                minutes = g["minutes_of_day"].to_numpy()
                for i, m in enumerate(minutes):
                    if m % stride != 0:
                        continue
                    # window of lookback minutes ending at m
                    start_m = m - lookback
                    win = g[(g["minutes_of_day"] >= start_m) & (g["minutes_of_day"] <= m)]
                    if len(win) < max(2, lookback // 10):
                        continue
                    tokens = ["[CLS]", f"ZONE_{'ABCD'[int(zone)]}", "[VIEW_EVENT_WINDOW]", "[SEP]"]
                    event_ids = []
                    for _, r in win.iterrows():
                        tokens.append("[EVENT_START]")
                        tokens.append(_phase(int(r["minutes_of_day"])))
                        for c in feature_cols:
                            tokens.append(value_to_token(c, float(r[c]), vocab))
                        tokens.append("[EVENT_SEP]")
                        event_ids.append(f"ENV:{int(dat)}:{int(zone)}:{int(r['minutes_of_day'])}")
                    tokens.append("[EVENT_END]")
                    ids = encode_tokens(tokens, vocab)
                    split_id = split_manifest["membership_by_dat"].get(str(int(dat)))
                    if split_id is None:
                        continue
                    sid = _seq_id([nid, dat, zone, m, split_id])
                    rows.append(
                        {
                            "sequence_id": sid,
                            "narrative_id": nid,
                            "family": narr["family"],
                            "dat": int(dat),
                            "zone_id": int(zone),
                            "anchor_minute": int(m),
                            "split_id": split_id,
                            "sop_eligible": bool(narr["sop_eligible"]),
                            "n_tokens": len(ids),
                            "token_ids": ids,
                            "event_ids": event_ids,
                            "source_window_id": f"train:DAT{int(dat)}:Z{int(zone)}",
                            "cube_ids": [],
                            "arm": "C0",
                        }
                    )
                    produced += 1
                    if _cap_hit(produced):
                        break
                if _cap_hit(produced):
                    break
        elif nid == "N_ACT_FAN_RESPONSE":
            produced = 0
            for (dat, zone), g in env_5min.groupby(["dat", "zone_id"]):
                g = g.sort_values("t_ord").reset_index(drop=True)
                if "circ_fan" not in g.columns and "fcu_fan" not in g.columns:
                    continue
                fan = g["circ_fan"].to_numpy() if "circ_fan" in g.columns else g["fcu_fan"].to_numpy()
                changes = np.where(np.diff(fan) != 0)[0] + 1
                for idx in changes:
                    a = max(0, idx - 3)
                    b = min(len(g), idx + 4)
                    win = g.iloc[a:b]
                    tokens = ["[CLS]", f"ZONE_{'ABCD'[int(zone)]}", "[VIEW_EVENT_WINDOW]", "[SEP]"]
                    event_ids = []
                    for _, r in win.iterrows():
                        tokens.append("[EVENT_START]")
                        for c in feature_cols:
                            tokens.append(value_to_token(c, float(r[c]), vocab))
                        tokens.append("[EVENT_SEP]")
                        event_ids.append(f"ENV:{int(dat)}:{int(zone)}:{int(r['minutes_of_day'])}")
                    tokens.append("[EVENT_END]")
                    ids = encode_tokens(tokens, vocab)
                    split_id = split_manifest["membership_by_dat"].get(str(int(dat)))
                    if split_id is None:
                        continue
                    rows.append(
                        {
                            "sequence_id": _seq_id([nid, dat, zone, int(g.iloc[idx]["minutes_of_day"])]),
                            "narrative_id": nid,
                            "family": narr["family"],
                            "dat": int(dat),
                            "zone_id": int(zone),
                            "anchor_minute": int(g.iloc[idx]["minutes_of_day"]),
                            "split_id": split_id,
                            "sop_eligible": True,
                            "n_tokens": len(ids),
                            "token_ids": ids,
                            "event_ids": event_ids,
                            "source_window_id": f"train:DAT{int(dat)}:Z{int(zone)}",
                            "cube_ids": [],
                            "arm": "C0",
                            "claim_language": "ACTUATOR_RESPONSE_CANDIDATE",
                        }
                    )
                    produced += 1
                    if _cap_hit(produced):
                        break
                if _cap_hit(produced):
                    break
        elif nid == "N_CUBE_ENV_ALIGN" and cube_index is not None and len(cube_index):
            produced = 0
            align_cfg = load_yaml(CONFIG_DIR / "cube_env_alignment.yaml")["cube_env_alignment"]
            max_back = int(align_cfg["max_backward_distance_minutes"])
            for _, cube in cube_index.iterrows():
                dat = int(cube["dat"])
                zone = int(cube["env_zone_id"])
                split_id = split_manifest["membership_by_dat"].get(str(dat))
                if split_id is None:
                    continue
                capt = int(cube["capture_end_minute"])
                g = env_5min[(env_5min["dat"] == dat) & (env_5min["zone_id"] == zone)]
                if g.empty:
                    continue
                # nearest past env minute
                past = g[g["minutes_of_day"] <= capt]
                if past.empty:
                    continue
                env_row = past.iloc[(past["minutes_of_day"] - capt).abs().argmin()]
                dist = abs(int(env_row["minutes_of_day"]) - capt)
                if dist > max_back and int(env_row["minutes_of_day"]) != capt:
                    # still allow if within expanded window of lookback narrative
                    if dist > int(narr["lookback"]):
                        continue
                tokens = ["[CLS]", f"ZONE_{'ABCD'[zone]}", "[VIEW_CUBE_ENV]", "[SEP]"]
                tokens.append("[EVENT_START]")
                for c in feature_cols:
                    tokens.append(value_to_token(c, float(env_row[c]), vocab))
                tokens.append("[EVENT_SEP]")
                # cube band slots (ids only; continuous emb separate)
                for i in range(10):
                    tokens.append(f"CUBE_BAND_{i}")
                tokens.append("CUBE_POOL")
                tokens.append("[EVENT_END]")
                ids = encode_tokens(tokens, vocab)
                rows.append(
                    {
                        "sequence_id": _seq_id([nid, cube["cube_id"]]),
                        "narrative_id": nid,
                        "family": narr["family"],
                        "dat": dat,
                        "zone_id": zone,
                        "anchor_minute": capt,
                        "split_id": split_id,
                        "sop_eligible": True,
                        "n_tokens": len(ids),
                        "token_ids": ids,
                        "event_ids": [
                            f"ENV:{dat}:{zone}:{int(env_row['minutes_of_day'])}",
                            f"CUBE:{cube['cube_id']}",
                        ],
                        "source_window_id": f"train:DAT{dat}:Z{zone}",
                        "cube_ids": [cube["cube_id"]],
                        "cube_env_distance_min": int(dist),
                        "arm": "C1",
                    }
                )
                produced += 1
                if _cap_hit(produced):
                    break
        elif nid == "N_EXT_TEMP_DAY":
            ext_fields = [c for c in ("temperature_outside", "solar_radiation", "rainfall") if c in feature_cols]
            head_tail_k = int(
                load_yaml(CONFIG_DIR / "pipeline.yaml").get("materialize", {}).get("ext_temp_day_head_tail_k", 8)
            )
            produced = 0
            for (dat, zone), g in env_5min.groupby(["dat", "zone_id"]):
                g = g.sort_values("t_ord")
                trace = g[g["minutes_of_day"] % 15 == 0]
                coverage = len(trace) / 96.0  # 96 fifteen-minute slots in a day
                if coverage < float(narr["minimum_coverage"]):
                    continue
                # head_tail chunk_policy: keep the day's opening/closing stretch instead of
                # every 15-min point, bounding token count on a lookback=1439min narrative.
                if len(trace) > 2 * head_tail_k:
                    chunk = pd.concat([trace.iloc[:head_tail_k], trace.iloc[-head_tail_k:]])
                else:
                    chunk = trace
                tokens = ["[CLS]", f"ZONE_{'ABCD'[int(zone)]}", "[VIEW_EVENT_WINDOW]", "[SEP]"]
                event_ids = []
                for _, r in chunk.iterrows():
                    tokens.append("[EVENT_START]")
                    for c in ext_fields:
                        tokens.append(value_to_token(c, float(r[c]), vocab))
                    tokens.append("[EVENT_SEP]")
                    event_ids.append(f"ENV:{int(dat)}:{int(zone)}:{int(r['minutes_of_day'])}")
                tokens.append("[EVENT_END]")
                ids = encode_tokens(tokens, vocab)
                split_id = split_manifest["membership_by_dat"].get(str(int(dat)))
                if split_id is None:
                    continue
                rows.append(
                    {
                        "sequence_id": _seq_id([nid, dat, zone]),
                        "narrative_id": nid,
                        "family": narr["family"],
                        "dat": int(dat),
                        "zone_id": int(zone),
                        "anchor_minute": 1439,
                        "split_id": split_id,
                        "sop_eligible": bool(narr["sop_eligible"]),
                        "n_tokens": len(ids),
                        "token_ids": ids,
                        "event_ids": event_ids,
                        "source_window_id": f"train:DAT{int(dat)}:Z{int(zone)}",
                        "cube_ids": [],
                        "arm": "C0",
                    }
                )
                produced += 1
                if _cap_hit(produced):
                    break
        elif nid == "N_PARALLEL_NOON":
            noon_minute = 720
            par_fields = [c for c in ("temperature", "humidity", "shading_curtain") if c in feature_cols]
            noon_rows = env_5min[env_5min["minutes_of_day"] == noon_minute]
            zone_dat_min = env_5min.groupby("zone_id")["dat"].min().to_dict()
            n_zones_total = env_5min["zone_id"].nunique()
            by_rel_day: dict[int, dict[int, pd.Series]] = {}
            for _, r in noon_rows.iterrows():
                zone = int(r["zone_id"])
                rel_day = int(r["dat"]) - int(zone_dat_min[zone])
                by_rel_day.setdefault(rel_day, {})[zone] = r
            produced = 0
            for rel_day, zone_map in sorted(by_rel_day.items()):
                if len(zone_map) < n_zones_total:  # exclude_incomplete_set (minimum_coverage=1.0)
                    continue
                dats_in_set = [int(r["dat"]) for r in zone_map.values()]
                # a zone-set can straddle several absolute DATs (one per zone); the DAT-based
                # train/valid/holdout partition only covers this narrative when every member
                # DAT lands in the same split, otherwise skip rather than risk cross-fold leakage.
                split_ids = {split_manifest["membership_by_dat"].get(str(d)) for d in dats_in_set}
                if len(split_ids) != 1 or None in split_ids:
                    continue
                split_id = split_ids.pop()
                tokens = ["[CLS]", "[VIEW_EVENT_WINDOW]", "[SEP]"]
                event_ids = []
                for zone in sorted(zone_map):
                    r = zone_map[zone]
                    tokens.append("[EVENT_START]")
                    tokens.append(f"ZONE_{'ABCD'[zone]}")
                    for c in par_fields:
                        tokens.append(value_to_token(c, float(r[c]), vocab))
                    tokens.append("[EVENT_SEP]")
                    event_ids.append(f"ENV:{int(r['dat'])}:{zone}:{noon_minute}")
                tokens.append("[EVENT_END]")
                ids = encode_tokens(tokens, vocab)
                rows.append(
                    {
                        "sequence_id": _seq_id([nid, rel_day]),
                        "narrative_id": nid,
                        "family": narr["family"],
                        "dat": dats_in_set[0],
                        "zone_id": -1,
                        "anchor_minute": noon_minute,
                        "split_id": split_id,
                        "sop_eligible": bool(narr["sop_eligible"]),
                        "n_tokens": len(ids),
                        "token_ids": ids,
                        "event_ids": event_ids,
                        "source_window_id": f"train:RELDAY{rel_day}:ALLZ",
                        "cube_ids": [],
                        "arm": "C0",
                    }
                )
                produced += 1
                if _cap_hit(produced):
                    break

    seq_df = pd.DataFrame(rows)
    report = {
        "n_sequences": int(len(seq_df)),
        "by_narrative": seq_df["narrative_id"].value_counts().to_dict() if len(seq_df) else {},
        "by_split": seq_df["split_id"].value_counts().to_dict() if len(seq_df) else {},
        "narrative_vocab_views": views,
        "checksum": sha256_json(rows[:50]) if rows else None,
    }
    return seq_df, report
