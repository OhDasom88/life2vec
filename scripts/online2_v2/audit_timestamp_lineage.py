#!/usr/bin/env python3
"""Audit cell_occurrences timestamps vs raw online2 CSV (offset 0 / ±9h)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


META_COLS = {"farm_id", "zone_id", "timestamp", "observation_date"}
OFFSETS = (0, 9, -9)


def _close(a: float, b: float, atol: float = 1e-3, rtol: float = 1e-6) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def _load_zone_csv(path: Path, csv_tz: str = "Asia/Seoul") -> Optional[pd.DataFrame]:
    """Load zone CSV. Naive timestamps are interpreted as ``csv_tz`` (default KST)."""
    if not path.exists():
        return None
    from zoneinfo import ZoneInfo

    df = pd.read_csv(path)
    tcol = "timestamp" if "timestamp" in df.columns else None
    if tcol is None:
        for c in df.columns:
            if "time" in c.lower() or c.lower() in {"datetime", "date"}:
                tcol = c
                break
    if tcol is None:
        return None
    ts = pd.to_datetime(df[tcol], errors="coerce")
    # Policy: naive online2 CSV wall clocks are Asia/Seoul.
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(ZoneInfo(csv_tz))
    else:
        ts = ts.dt.tz_convert("UTC")
    df[tcol] = ts.dt.tz_convert("UTC")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _load_root(path: Path, farm: str, csv_tz: str = "Asia/Seoul") -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    from zoneinfo import ZoneInfo

    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(ZoneInfo(csv_tz))
    df["timestamp"] = ts.dt.tz_convert("UTC")
    df = df[df["farm_id"].astype(str) == farm].dropna(subset=["timestamp"])
    out: dict[str, pd.DataFrame] = {}
    for z, g in df.groupby(df["zone_id"].astype(str)):
        g = g.set_index("timestamp").sort_index()
        g = g[~g.index.duplicated(keep="first")]
        out[str(z)] = g
    return out


def _load_growth(path: Path, farm: str, csv_tz: str = "Asia/Seoul") -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    from zoneinfo import ZoneInfo

    df = pd.read_csv(path)
    dcol = "observation_date" if "observation_date" in df.columns else None
    if dcol is None:
        return {}
    df = df[df["farm_id"].astype(str) == farm].copy()
    ts = pd.to_datetime(df[dcol], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(ZoneInfo(csv_tz))
    df[dcol] = ts.dt.tz_convert("UTC")
    df = df.dropna(subset=[dcol])
    out: dict[str, pd.DataFrame] = {}
    for z, g in df.groupby(df["zone_id"].astype(str)):
        g = g.set_index(dcol).sort_index()
        g = g[~g.index.duplicated(keep="first")]
        out[str(z)] = g
    return out


def audit_farm(
    farm: str,
    cells: pd.DataFrame,
    data_root: Path,
    atol: float,
    rtol: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env: dict[str, pd.DataFrame] = {}
    act: dict[str, pd.DataFrame] = {}
    for z in sorted({str(x) for x in cells["zone_id"].unique()}):
        e = _load_zone_csv(data_root / "E_environment" / f"{farm}_z{z}.csv")
        a = _load_zone_csv(data_root / "A_actuator" / f"{farm}_z{z}.csv")
        if e is not None:
            env[z] = e
        if a is not None:
            act[z] = a
    root = _load_root(data_root / "R_rootzone" / "root_data.csv", farm)
    growth = _load_growth(data_root / "G_growth" / "growth_data.csv", farm)

    modality_map = {
        "E_environment": env,
        "A_actuator": act,
        "R_rootzone": root,
        "G_growth": growth,
    }

    rows: list[dict[str, Any]] = []
    # aggregates: (modality, feature) -> counters
    agg = defaultdict(lambda: {o: {"match": 0, "total": 0} for o in OFFSETS})
    # per modality best-offset histogram for rows that match exactly one offset best
    mod_best = defaultdict(lambda: defaultdict(int))
    mod_view_pair = defaultdict(lambda: defaultdict(int))  # for STG-ish: offset agreement

    for rec in cells.itertuples(index=False):
        col = str(rec.column_name)
        if col in META_COLS:
            continue
        modality = str(rec.modality)
        sources = modality_map.get(modality)
        if not sources:
            continue
        z = str(rec.zone_id)
        base = sources.get(z)
        if base is None or col not in base.columns:
            continue
        try:
            cell_v = float(rec.raw_display)
        except (TypeError, ValueError):
            continue
        if pd.isna(cell_v):
            continue
        ts = pd.Timestamp(rec.ts)
        if pd.isna(ts):
            continue

        matches = {}
        csv_vals = {}
        for o in OFFSETS:
            t = ts + pd.Timedelta(hours=o)
            if t not in base.index:
                matches[o] = None
                csv_vals[o] = None
                continue
            try:
                sv = float(base.loc[t, col])
            except Exception:
                matches[o] = None
                csv_vals[o] = None
                continue
            if pd.isna(sv):
                matches[o] = None
                csv_vals[o] = None
                continue
            ok = _close(cell_v, sv, atol=atol, rtol=rtol)
            matches[o] = ok
            csv_vals[o] = sv
            key = (modality, col)
            agg[key][o]["total"] += 1
            if ok:
                agg[key][o]["match"] += 1

        # best offset among those with data
        scored = [(o, matches[o]) for o in OFFSETS if matches[o] is not None]
        if not scored:
            continue
        # prefer matching offsets; if multiple, record all; if none match, best=-999
        matching = [o for o, m in scored if m]
        if len(matching) == 1:
            best = matching[0]
        elif len(matching) > 1:
            best = 999  # ambiguous
        else:
            best = -999  # no match
        mod_best[modality][best] += 1

        rows.append(
            {
                "farm_id": farm,
                "zone_id": z,
                "source_view": modality,
                "feature": col,
                "cell_timestamp_utc": ts.isoformat(),
                "cell_value": cell_v,
                "csv_value_at_T": csv_vals.get(0),
                "csv_value_at_T_minus_9h": csv_vals.get(-9),
                "csv_value_at_T_plus_9h": csv_vals.get(9),
                "match_at_0h": matches.get(0),
                "match_at_minus_9h": matches.get(-9),
                "match_at_plus_9h": matches.get(9),
                "best_offset_hours": best,
            }
        )

    # modality co-offset: for same farm/zone/ts, do E and A share best?
    # Use sampled unique ts from rows
    by_ts: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r["best_offset_hours"] in OFFSETS:
            by_ts[(r["zone_id"], r["cell_timestamp_utc"])][r["source_view"]].add(
                int(r["best_offset_hours"])
            )
    stg_stats = {"comparable_slots": 0, "same_offset": 0, "diff_offset": 0}
    for (_z, _t), views in by_ts.items():
        if "E_environment" in views and "A_actuator" in views:
            stg_stats["comparable_slots"] += 1
            e_off = views["E_environment"]
            a_off = views["A_actuator"]
            if e_off & a_off:
                stg_stats["same_offset"] += 1
            else:
                stg_stats["diff_offset"] += 1

    summary = {
        "farm_id": farm,
        "n_compared_rows": len(rows),
        "modality_best_offset_counts": {m: dict(c) for m, c in mod_best.items()},
        "feature_match_rates": {},
        "stg_env_act_offset_agreement": stg_stats,
    }
    for (mod, feat), od in agg.items():
        summary["feature_match_rates"][f"{mod}|{feat}"] = {
            str(o): {
                "match_rate": (v["match"] / v["total"] if v["total"] else None),
                "match": v["match"],
                "total": v["total"],
            }
            for o, v in od.items()
        }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells",
        type=Path,
        default=Path("outputs/online2/build-v8-active80-r3/cell_occurrences.parquet"),
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/agrichallenge/online2/data"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/online2/v2_finetune/timestamp_lineage_audit_kst"),
    )
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--max-detail-rows", type=int, default=200_000)
    ap.add_argument("--farms", type=str, default="")  # comma list optional
    args = ap.parse_args()

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("loading cells…", flush=True)
    table = pq.read_table(
        args.cells,
        columns=[
            "farm_id",
            "zone_id",
            "observation_timestamp",
            "column_name",
            "raw_display",
            "modality",
            "is_null",
        ],
    )
    cells = table.to_pandas()
    cells = cells[~cells["is_null"].fillna(False)]
    cells = cells[~cells["column_name"].isin(META_COLS)]
    cells["farm_id"] = cells["farm_id"].astype(str)
    cells["zone_id"] = cells["zone_id"].astype(str)
    cells["ts"] = pd.to_datetime(cells["observation_timestamp"], utc=True, errors="coerce")
    cells = cells.dropna(subset=["ts"])

    farms = sorted(cells["farm_id"].unique().tolist())
    if args.farms.strip():
        want = {x.strip() for x in args.farms.split(",") if x.strip()}
        farms = [f for f in farms if f in want]
    print(f"farms={len(farms)} cells={len(cells)}", flush=True)

    all_summaries: list[dict[str, Any]] = []
    detail_chunks: list[pd.DataFrame] = []
    detail_budget = args.max_detail_rows
    global_mod_best: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    global_stg = {"comparable_slots": 0, "same_offset": 0, "diff_offset": 0}
    farm_dominant: list[dict[str, Any]] = []

    for i, farm in enumerate(farms, 1):
        sub = cells[cells["farm_id"] == farm]
        rows, summary = audit_farm(farm, sub, args.data_root, args.atol, args.rtol)
        all_summaries.append(summary)
        for m, counts in summary["modality_best_offset_counts"].items():
            for k, v in counts.items():
                global_mod_best[m][str(k)] += v
        stg = summary["stg_env_act_offset_agreement"]
        for k in global_stg:
            global_stg[k] += stg.get(k, 0)

        # dominant offset per modality for this farm
        dom = {"farm_id": farm}
        for m, counts in summary["modality_best_offset_counts"].items():
            if not counts:
                continue
            # ignore ambiguous/no-match for dominance among 0/9/-9
            usable = {k: v for k, v in counts.items() if k in OFFSETS}
            if usable:
                best_o = max(usable, key=usable.get)
                total = sum(counts.values())
                dom[f"{m}_dominant_offset"] = best_o
                dom[f"{m}_dominant_frac"] = usable[best_o] / total if total else None
                dom[f"{m}_n"] = total
        farm_dominant.append(dom)

        if detail_budget > 0 and rows:
            take = min(len(rows), max(500, detail_budget // max(1, len(farms) - i + 1)))
            # stratified: prefer mismatches at 0h
            rdf = pd.DataFrame(rows)
            mis = rdf[rdf["match_at_0h"] == False]
            ok = rdf[rdf["match_at_0h"] == True]
            part = pd.concat(
                [mis.head(take // 2), ok.head(take - take // 2)], ignore_index=True
            ).head(take)
            detail_chunks.append(part)
            detail_budget -= len(part)

        if i % 5 == 0 or i == len(farms):
            print(f"  farm {i}/{len(farms)} {farm} rows={len(rows)}", flush=True)

    detail = pd.concat(detail_chunks, ignore_index=True) if detail_chunks else pd.DataFrame()
    detail_path = out / "timestamp_lineage_audit_sample.csv"
    detail.to_csv(detail_path, index=False)

    farm_dom_df = pd.DataFrame(farm_dominant)
    farm_dom_df.to_csv(out / "farm_modality_dominant_offset.csv", index=False)

    # overall match rates by modality folding feature rates
    overall_mod = defaultdict(lambda: {str(o): {"match": 0, "total": 0} for o in OFFSETS})
    for s in all_summaries:
        for key, od in s["feature_match_rates"].items():
            mod = key.split("|", 1)[0]
            for o, st in od.items():
                overall_mod[mod][o]["match"] += st["match"]
                overall_mod[mod][o]["total"] += st["total"]

    overall_rates = {
        mod: {
            o: {
                "match_rate": (v["match"] / v["total"] if v["total"] else None),
                "match": v["match"],
                "total": v["total"],
            }
            for o, v in od.items()
        }
        for mod, od in overall_mod.items()
    }

    # classify farms by E+A dominant
    def farm_class(row: pd.Series) -> str:
        e = row.get("E_environment_dominant_offset")
        a = row.get("A_actuator_dominant_offset")
        if pd.isna(e) and pd.isna(a):
            return "no_EA"
        if e == 9 and a == 9:
            return "uniform_plus_9h"
        if e == 0 and a == 0:
            return "uniform_0h"
        if e == -9 and a == -9:
            return "uniform_minus_9h"
        if pd.notna(e) and pd.notna(a) and e != a:
            return "EA_disagree"
        return "mixed_or_partial"

    if len(farm_dom_df):
        farm_dom_df["ea_class"] = farm_dom_df.apply(farm_class, axis=1)
        class_counts = farm_dom_df["ea_class"].value_counts().to_dict()
    else:
        class_counts = {}

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cells_path": str(args.cells),
        "data_root": str(args.data_root),
        "csv_tz_policy": "Asia/Seoul",
        "n_farms": len(farms),
        "atol": args.atol,
        "rtol": args.rtol,
        "overall_match_rates_by_modality": overall_rates,
        "global_modality_best_offset_counts": {m: dict(c) for m, c in global_mod_best.items()},
        "stg_env_act_offset_agreement": global_stg,
        "farm_ea_class_counts": class_counts,
        "detail_sample_rows": int(len(detail)),
        "detail_sample_path": str(detail_path),
        "verdict_hints": {
            "expected_under_kst_policy": "match@0h ≈ 1 across modalities; E/A uniform_0h",
            "if_mostly_uniform_plus_9h": "CSV likely still loaded as UTC (utc=True on naive) — check loader",
            "if_EA_disagree": "SameTimeGroup cross-view corruption — full rebuild required",
        },
    }
    (out / "timestamp_lineage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # human summary md
    lines = [
        "# Timestamp lineage audit report",
        "",
        f"- cells: `{args.cells}`",
        f"- csv: `{args.data_root}`",
        f"- farms: {len(farms)}",
        "- csv_tz_policy: **Asia/Seoul (KST)** — naive CSV wall clocks localized then converted to UTC",
        "- policy: `outputs/online2/v2_finetune/TIMEZONE_POLICY_KST.md`",
        "",
        "## Overall match rates (cell(T) vs csv(T+offset))",
        "",
    ]
    for mod, od in sorted(overall_rates.items()):
        lines.append(f"### {mod}")
        for o in ["0", "9", "-9"]:
            st = od.get(o, {})
            lines.append(
                f"- offset {o}h: match_rate={st.get('match_rate')} "
                f"({st.get('match')}/{st.get('total')})"
            )
        lines.append("")
    lines.append("## Farm E/A class counts")
    lines.append("")
    for k, v in sorted(class_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Env–Act same-timestamp offset agreement")
    lines.append("")
    lines.append(f"- comparable zone×ts slots: {global_stg['comparable_slots']}")
    lines.append(f"- same offset: {global_stg['same_offset']}")
    lines.append(f"- different offset: {global_stg['diff_offset']}")
    if global_stg["comparable_slots"]:
        frac = global_stg["same_offset"] / global_stg["comparable_slots"]
        lines.append(f"- agreement rate: {frac:.4f}")
    lines.append("")
    (out / "TIMESTAMP_LINEAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False)[:2000])
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
