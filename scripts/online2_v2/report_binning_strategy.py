#!/usr/bin/env python3
"""Fit adaptive V2 binning and export feature/farm strategy tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.online2.v2.binning import BinningPolicy, BinningRegistryV2, fit_binning_v2
from src.online2.v2.feature_schema import FeatureSchema, build_feature_schema_from_audit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = ROOT / "outputs/online2/build-v8-active80-r3"
DEFAULT_AUDIT = ROOT / "outputs/online2/v2_audit/feature_semantics_and_units.csv"
DEFAULT_OUT = ROOT / "outputs/online2/v2_build"
DEFAULT_POLICY = DEFAULT_OUT / "binning_policy_v2.yaml"
DATA = ROOT / "datasets/agrichallenge/online2"


def case_sets() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in ("example_set", "problem_set"):
        path = DATA / name / "case_list.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for farm in frame["farm_id"].astype(str):
            mapping[farm] = name
    return mapping


def load_fit_rows(legacy_build: Path) -> tuple[list[tuple[str, Optional[str], Optional[float], str]], int, int]:
    farms = case_sets()
    cells = pd.read_parquet(
        legacy_build / "cell_occurrences.parquet",
        columns=["cell_id", "column_name", "raw_display", "farm_id", "zone_id", "is_null"],
    )
    cells = cells.sort_values(["column_name", "farm_id", "zone_id", "cell_id"], kind="mergesort")
    rows: list[tuple[str, Optional[str], Optional[float], str]] = []
    example_n = problem_n = 0
    for rec in cells.itertuples(index=False):
        source_set = farms.get(str(rec.farm_id), "unknown")
        if source_set == "example_set":
            example_n += 1
        elif source_set == "problem_set":
            problem_n += 1
        raw = "" if bool(getattr(rec, "is_null", False)) else str(rec.raw_display)
        try:
            value = float(raw) if raw not in {"", "nan", "None"} else None
        except ValueError:
            value = None
        rows.append((str(rec.column_name), str(rec.farm_id), value, source_set))
    return rows, example_n, problem_n


def _edges_str(edges: tuple[float, ...], limit: int = 14) -> str:
    if len(edges) <= limit:
        return json.dumps(list(edges))
    head = list(edges[: limit // 2])
    tail = list(edges[-(limit // 2) :])
    return json.dumps(head + ["..."] + tail)


def _threshold_placement(edges: tuple[float, ...], thresholds: list[tuple[str, float]]) -> str:
    if len(edges) < 2:
        return ""
    parts = []
    import bisect

    for name, value in thresholds:
        pos = bisect.bisect_left(edges[1:-1], value)
        lo, hi = edges[pos], edges[pos + 1]
        width = hi - lo
        parts.append(f"{name}→B{pos:02d}[{lo:.4g},{hi:.4g}] w={width:.4g}")
    return "; ".join(parts)


def feature_table(schema: FeatureSchema, binning: BinningRegistryV2, policy: BinningPolicy) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, spec in sorted(schema.features.items()):
        if not (spec.absolute_encoding or spec.global_relative_encoding or spec.farm_relative_encoding):
            continue
        abs_rule = binning.rules.get((name, "ABS", None))
        glob_rule = binning.rules.get((name, "GLOBAL_REL", None))
        farm_rules = [r for (f, k, s), r in binning.rules.items() if f == name and k == "FARM_REL"]
        feat_pol = policy.features.get(name, {})
        notes = []
        if feat_pol.get("rationale_ko"):
            notes.append(str(feat_pol["rationale_ko"]))
        if abs_rule and abs_rule.strategy.startswith("zero_inflated"):
            notes.append(f"영점질량 {abs_rule.zero_mass_frac:.1%} → B00 예약 + 양수 균등빈도")
        if farm_rules:
            const_n = sum(1 for r in farm_rules if r.constant)
            if const_n:
                notes.append(f"농장 {const_n}곳 상수")
        thr = ""
        if name == "inside_humidity_pct" and abs_rule:
            thr = _threshold_placement(abs_rule.edges, [("RH90", 90.0), ("RH65", 65.0)])
        elif name == "inside_temp_c" and abs_rule:
            thr = _threshold_placement(abs_rule.edges, [("T25", 25.0), ("T6", 6.0)])
        elif name == "substrate_ec_ds_m" and abs_rule:
            thr = _threshold_placement(abs_rule.edges, [("EC1.8", 1.8), ("EC1.2", 1.2)])
        rows.append(
            {
                "feature": name,
                "event_view": spec.event_view,
                "type": spec.type,
                "unit_confidence": spec.unit_confidence,
                "tier": feat_pol.get("tier", ""),
                "encoding_abs": spec.absolute_encoding,
                "encoding_global_rel": spec.global_relative_encoding,
                "encoding_farm_rel": spec.farm_relative_encoding,
                "abs_strategy": None if abs_rule is None else abs_rule.strategy,
                "abs_n_bins_requested": None if abs_rule is None else abs_rule.n_bins_requested,
                "abs_n_bins": None if abs_rule is None else abs_rule.n_bins_effective,
                "abs_zero_mass_frac": None if abs_rule is None else abs_rule.zero_mass_frac,
                "abs_occupancy_cv": None if abs_rule is None else abs_rule.occupancy_cv,
                "abs_edges": None if abs_rule is None else _edges_str(abs_rule.edges),
                "global_strategy": None if glob_rule is None else glob_rule.strategy,
                "global_n_bins_requested": None if glob_rule is None else glob_rule.n_bins_requested,
                "global_n_bins": None if glob_rule is None else glob_rule.n_bins_effective,
                "global_zero_mass_frac": None if glob_rule is None else glob_rule.zero_mass_frac,
                "global_occupancy_cv": None if glob_rule is None else glob_rule.occupancy_cv,
                "global_edges": None if glob_rule is None else _edges_str(glob_rule.edges),
                "farm_rule_count": len(farm_rules),
                "farm_median_n_bins": float(np.median([r.n_bins_effective for r in farm_rules])) if farm_rules else None,
                "farm_constant_count": sum(1 for r in farm_rules if r.constant),
                "farm_zero_inflated_count": sum(1 for r in farm_rules if "zero_inflated" in r.strategy),
                "threshold_bin_placement": thr,
                "notes_ko": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def farm_table(binning: BinningRegistryV2) -> pd.DataFrame:
    rows = []
    for (feature, kind, farm), rule in sorted(binning.rules.items(), key=lambda x: (x[0][0], x[0][2] or "")):
        if kind != "FARM_REL":
            continue
        rows.append(
            {
                "feature": feature,
                "farm_id": farm,
                "strategy": rule.strategy,
                "n_bins_requested": rule.n_bins_requested,
                "n_bins_effective": rule.n_bins_effective,
                "constant": rule.constant,
                "zero_mass_frac": rule.zero_mass_frac,
                "occupancy_cv": rule.occupancy_cv,
                "n_values": (rule.diagnostics or {}).get("n_values"),
                "n_unique": (rule.diagnostics or {}).get("n_unique"),
                "edges": _edges_str(rule.edges, limit=10),
            }
        )
    return pd.DataFrame(rows)


def write_strategy_md(out: Path, feature_df: pd.DataFrame, binning: BinningRegistryV2) -> None:
    crit = feature_df[feature_df["tier"] == "disease_critical"] if "tier" in feature_df.columns else feature_df.head(0)
    lines = [
        "# Online2 V2 Binning Strategy",
        "",
        f"- Policy: `{binning.meta.get('policy_version')}` / `{binning.meta.get('edge_policy')}`",
        f"- Registry hash: `{binning.meta.get('registry_hash')}`",
        f"- Fit values: {binning.meta.get('n_fit_values')}",
        f"- Farm rules: {binning.meta.get('farm_rule_count')}",
        "",
        "## Principles",
        "",
        "1. **Equal-frequency (quantile) edges** so each bin gets similar sample mass (not equal-width).",
        "2. **Per-feature `n_bins`** — disease/causal critical features get denser bins (≤100), counters stay coarse.",
        "3. **Zero-inflation** when near-zero mass ≥ 15%: reserve `B00` for OFF/zero, quantile the positive tail.",
        "4. **Domain anchors** (RH90, T25, EC1.8, …) inserted into ABS/GLOBAL edges for critical features.",
        "5. **Farm-relative** bins are fit per `(feature, farm)` with fewer bins (`farm_bins_scale≈0.5`) and sample floors.",
        "6. Encoding-off actuators/flows keep policy ready but do not emit value bins until codebook unlock.",
        "",
        "## Channel roles",
        "",
        "| Channel | Scope | Role |",
        "|---------|-------|------|",
        "| ABS | global feature | physical magnitude + threshold anchors |",
        "| GLOBAL_REL | global feature | corpus-relative rank (same algorithm) |",
        "| FARM_REL | feature × farm | within-farm rank / local operating point |",
        "",
        "## Disease-critical placement",
        "",
    ]
    if not crit.empty:
        lines.append("| feature | abs_n_bins | occupancy_cv | threshold placement |")
        lines.append("|---|---:|---:|---|")
        for _, row in crit.iterrows():
            lines.append(
                f"| {row['feature']} | {row.get('abs_n_bins')} | {row.get('abs_occupancy_cv'):.4f} | {row.get('threshold_bin_placement','')} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            "- `binning_policy_v2.yaml` — declarative tier/feature policy",
            "- `binning_registry_v2_transductive.json` — fitted edges",
            "- `binning_strategy_by_feature.csv` — feature summary",
            "- `binning_strategy_by_feature_farm.csv` — farm summary",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/online2_v2/report_binning_strategy.py --refit",
            "# or full registries:",
            "python scripts/online2_v2/build_v2.py  # uses binning_policy_v2.yaml",
            "```",
            "",
        ]
    )
    (out / "BINNING_STRATEGY_V2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--refit", action="store_true", help="Refit registry and overwrite JSON")
    parser.add_argument("--from-registry", action="store_true", help="Only report from existing registry")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    schema = build_feature_schema_from_audit(args.audit)
    schema.save(args.out / "feature_schema_v2.yaml")
    policy = BinningPolicy.load(args.policy if args.policy.exists() else None)

    if args.from_registry and (args.out / "binning_registry_v2_transductive.json").exists() and not args.refit:
        binning = BinningRegistryV2.load(args.out / "binning_registry_v2_transductive.json")
    else:
        rows, example_n, problem_n = load_fit_rows(args.legacy_build)
        source_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.legacy_build.glob("*.json"))
        }
        binning = fit_binning_v2(
            schema,
            rows,
            source_file_hashes=source_hashes,
            example_count=example_n,
            problem_count=problem_n,
            policy=policy,
        )
        if args.refit or not (args.out / "binning_registry_v2_transductive.json").exists():
            binning.save(args.out / "binning_registry_v2_transductive.json")
            (args.out / "binning_policy_v2.json").write_text(
                json.dumps(
                    {
                        "edge_policy": binning.meta.get("edge_policy"),
                        "abs_policy": binning.meta.get("abs_policy"),
                        "global_rel_policy": binning.meta.get("global_rel_policy"),
                        "farm_rel_policy": binning.meta.get("farm_rel_policy"),
                        "zero_mass_threshold": binning.meta.get("zero_mass_threshold"),
                        "policy_version": binning.meta.get("policy_version"),
                        "registry_hash": binning.meta.get("registry_hash"),
                        "feature_count": len(binning.meta.get("feature_summaries") or {}),
                        "farm_rule_count": binning.meta.get("farm_rule_count"),
                        "policy_yaml": str(args.policy),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    feat_df = feature_table(schema, binning, policy)
    farm_df = farm_table(binning)
    feat_df.to_csv(args.out / "binning_strategy_by_feature.csv", index=False)
    farm_df.to_csv(args.out / "binning_strategy_by_feature_farm.csv", index=False)
    write_strategy_md(args.out, feat_df, binning)
    print(
        json.dumps(
            {
                "features": len(feat_df),
                "farm_rows": len(farm_df),
                "registry_hash": binning.meta.get("registry_hash"),
                "policy_version": binning.meta.get("policy_version"),
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
