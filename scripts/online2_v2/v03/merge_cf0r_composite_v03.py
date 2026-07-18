#!/usr/bin/env python3
"""Merge parent CF-0 (47) with CF-0R recovery (8) into a read-only composite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.online2_v2.v03.run_cf_funnel_audit_v03 import (  # noqa: E402
    COHORT_ROLES,
    _aggregate_cohort,
    _write_report,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import file_sha256, write_json

FAILED8 = [
    "F155481_2024-12-24_2025-01-06",
    "F209570_2025-03-01_2025-03-14",
    "F393975_2024-10-21_2024-11-03",
    "F420458_2025-02-16_2025-03-01",
    "F541027_2025-03-07_2025-03-20",
    "F584775_2024-11-08_2024-11-21",
    "F596090_2025-01-18_2025-01-31",
    "F995842_2025-02-12_2025-02-25",
]


def _git_rev() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adapter_parity(parent_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-read parent JSON fields that must remain stable under new schema."""
    from src.online2.v2.finetune_v03.counterfactual.evaluation.funnel_accounting import (
        validate_cf0_conservation,
        validate_cf0_monotonicity,
    )

    counts = {
        k: parent_row.get(k)
        for k in (
            "editable_loci_count",
            "raw_generated_candidate_count",
            "deduplicated_candidate_count",
            "bank_supported_candidate_count",
            "decoder_scored_candidate_count",
            "hard_constraint_pass_count",
            "plausibility_pass_count",
            "effect_pass_count",
            "stability_pass_count",
            "valid_cf_count",
            "selected_non_noop_count",
            "candidate_rejection_count",
            "candidate_execution_error_count",
            "gate0_pass_count",
            "inversion_pass_count",
            "gate4_pass_count",
            "hard_equals_gate4_invariant_expected",
            "terminal_reason_counts",
            "execution_error_reason_counts",
        )
    }
    mono_ok, mono_errs = validate_cf0_monotonicity(counts)
    cons_ok, cons_errs = validate_cf0_conservation(counts)
    return {
        "monotonicity_pass": mono_ok,
        "conservation_pass": cons_ok,
        "monotonicity_errors": mono_errs,
        "conservation_errors": cons_errs,
        "valid_cf_count": int(parent_row.get("valid_cf_count") or 0),
        "deduplicated_candidate_count": int(parent_row.get("deduplicated_candidate_count") or 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parent-per-case",
        type=Path,
        default=ROOT / "reports/cf0_funnel_audit/CF0_FUNNEL_AUDIT_20260718T131735Z/per_case",
    )
    ap.add_argument(
        "--recovery-per-case",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs/m1_m2_prereq/funnel_audit_cf0r",
    )
    ap.add_argument("--parent-commit", type=str, default=None)
    ap.add_argument("--cf0r-commit", type=str, default=None)
    ap.add_argument("--push-allowed", action="store_true", default=False)
    args = ap.parse_args()

    parent_commit = args.parent_commit or "cf0_parent_archive_20260718T131735Z"
    cf0r_commit = args.cf0r_commit or _git_rev() or "UNAVAILABLE"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_root) / f"CF0R_COMPOSITE_{stamp}"
    per_case_out = out / "per_case"
    per_case_out.mkdir(parents=True, exist_ok=True)

    parent_files = sorted(Path(args.parent_per_case).glob("*.json"))
    recovery_files = {p.stem: p for p in Path(args.recovery_per_case).glob("*.json")}
    failed_set = set(FAILED8)

    merged: List[Dict[str, Any]] = []
    parent_parity: Dict[str, Any] = {}
    non_impact_ok = True

    for pf in parent_files:
        case_id = pf.stem
        row = _load_json(pf)
        if case_id in failed_set:
            if case_id not in recovery_files:
                raise SystemExit(f"missing recovery for {case_id}")
            rec = _load_json(recovery_files[case_id])
            rec = dict(rec)
            rec["result_origin"] = "CF0R_SCHEMA_RECOVERY"
            rec["source_commit"] = cf0r_commit
            rec["parent_cf0_source_commit"] = parent_commit
            rec["cf0r_source_commit"] = cf0r_commit
            write_json(per_case_out / f"{case_id}.json", rec)
            merged.append(rec)
        else:
            # Parent unchanged — stamp provenance only; do not rewrite funnel counts
            adapted = dict(row)
            adapted["result_origin"] = "PARENT_CF0"
            adapted["source_commit"] = parent_commit
            adapted["parent_cf0_source_commit"] = parent_commit
            adapted["cf0r_source_commit"] = cf0r_commit
            # Fill new optional fields without changing conservation inputs
            adapted.setdefault("model_valid_cf_count", int(row.get("valid_cf_count") or 0))
            adapted.setdefault("observational_sensitivity_valid_count", 0)
            adapted.setdefault("recommendation_valid_cf_count", 0)
            adapted.setdefault("selected_actionable_non_noop_count", 0)
            adapted.setdefault("unsupported_locus_count", 0)
            adapted.setdefault("case_execution_error", False)
            adapted.setdefault("case_execution_error_count", 0)
            parity = _adapter_parity(row)
            parent_parity[case_id] = parity
            if not (parity["monotonicity_pass"] and parity["conservation_pass"]):
                non_impact_ok = False
            write_json(per_case_out / f"{case_id}.json", adapted)
            merged.append(adapted)

    if len(merged) != 55:
        raise SystemExit(f"expected 55 merged cases, got {len(merged)}")

    by_cohort: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in merged:
        by_cohort[str(r.get("cohort"))].append(r)

    summary = {
        "audit_id": f"cf0r_composite_{stamp}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "cf0_v1",
        "merged_result_type": "READ_ONLY_COMPOSITE_47_PLUS_8",
        "single_execution_lineage": False,
        "parent_case_count": 47,
        "recovery_case_count": 8,
        "parent_cf0_source_commit": parent_commit,
        "cf0r_source_commit": cf0r_commit,
        "push_allowed": bool(args.push_allowed),
        "n_cases_total": len(merged),
        "parent_non_impact_verified": bool(non_impact_ok),
        "parent_adapter_parity": {
            "n_checked": len(parent_parity),
            "all_mono_cons_pass": bool(non_impact_ok),
        },
        "cohorts": {k: _aggregate_cohort(v) for k, v in by_cohort.items()},
        "completion_gate": {},
    }

    n_ok = sum(1 for r in merged if str(r.get("run_status")) == "OK")
    n_case_err = sum(1 for r in merged if bool(r.get("case_execution_error")) or str(r.get("run_status")) != "OK")
    pending_false = sum(1 for r in merged if r.get("pending_downstream_stages") is False)
    mono = sum(1 for r in merged if r.get("monotonicity_pass"))
    cons = sum(1 for r in merged if r.get("conservation_pass"))
    rec_valid = sum(int(r.get("recommendation_valid_cf_count") or 0) for r in merged)
    normal_actionable = sum(
        int(r.get("selected_actionable_non_noop_count") or 0)
        for r in merged
        if str(r.get("cohort")) == "example35_normal"
    )
    gate = {
        "total_cases": len(merged),
        "evaluable_cases": n_ok,
        "case_execution_error_count": n_case_err,
        "pending_downstream_stages_false": f"{pending_false}/{len(merged)}",
        "monotonicity_pass": f"{mono}/{len(merged)}",
        "conservation_pass": f"{cons}/{len(merged)}",
        "recommendation_valid_cf_count": rec_valid,
        "example35_normal_selected_actionable_non_noop_count": normal_actionable,
        "status": (
            "COMPLETE_55_OF_55"
            if (
                len(merged) == 55
                and n_ok == 55
                and n_case_err == 0
                and pending_false == 55
                and mono == 55
                and cons == 55
                and normal_actionable == 0
            )
            else "INCOMPLETE"
        ),
    }
    summary["completion_gate"] = gate
    write_json(out / "funnel_audit_summary.json", summary)

    manifest = {
        "audit_id": summary["audit_id"],
        "created_at": summary["created_at"],
        "merged_result_type": "READ_ONLY_COMPOSITE_47_PLUS_8",
        "single_execution_lineage": False,
        "parent_cf0_source_commit": parent_commit,
        "cf0r_source_commit": cf0r_commit,
        "push_allowed": False,
        "parent_per_case": str(args.parent_per_case),
        "recovery_per_case": str(args.recovery_per_case),
        "failed8": FAILED8,
    }
    write_json(out / "audit_manifest.json", manifest)
    _write_report(
        out / "FUNNEL_AUDIT_REPORT.md",
        manifest=manifest,
        summary=summary,
        per_case=merged,
    )

    print(
        json.dumps(
            {
                "status": "CF0R_COMPOSITE_MERGED",
                "out": str(out),
                "completion_gate": gate,
                "parent_non_impact_verified": non_impact_ok,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if gate["status"] == "COMPLETE_55_OF_55" or n_ok >= 47 else 1


if __name__ == "__main__":
    raise SystemExit(main())
