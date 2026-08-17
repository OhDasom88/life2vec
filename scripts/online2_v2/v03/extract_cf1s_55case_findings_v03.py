#!/usr/bin/env python3
"""Extract real per-case CF1S counterfactual findings from the 3 promoted
evidence directories (Development3/Validation20/Primary32) into a single
JSON list, for downstream narrative generation. Reads only canonical
case_edit_proposal.json files from PROMOTED packages — no fabrication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]

RUNS = {
    "DEVELOPMENT3": ROOT
    / "outputs/cf1s_core/development3_evidence/development3-20260719T142507Z",
    "VALIDATION20": ROOT
    / "outputs/cf1s_core/validation20_evidence/validation20-20260720T013942Z",
    "PRIMARY32": ROOT / "outputs/cf1s_core/primary32_evidence/primary32-20260720T064408Z",
}


def _extract_case(cohort: str, case_dir: Path) -> Dict[str, Any]:
    prop = json.loads((case_dir / "case_edit_proposal.json").read_text(encoding="utf-8"))
    from src.online2.v2.finetune_v03.counterfactual.cf1s.disposition_profiles import (
        classify_case_disposition,
    )

    disposition = classify_case_disposition(prop)
    candidate_results = prop.get("candidate_results") or []
    selected = [
        r
        for r in candidate_results
        if r.get("disposition") in ("SELECTED_MATERIAL", "SELECTED_CONTROL_NO_MATERIAL")
    ]
    sel = selected[0] if selected else None
    extra = prop.get("extra") or {}
    scientific_detail = extra.get("scientific_detail") or {}
    threshold = (extra.get("threshold") or {}).get("locked_min_effect_abs_delta")

    return {
        "cohort": cohort,
        "case_id": prop.get("case_id"),
        "disposition": disposition,
        "construction_status": prop.get("construction_status"),
        "scientific_status": prop.get("scientific_status"),
        "candidate_id": (sel or {}).get("candidate_id") or scientific_detail.get(
            "bundle_candidate_id"
        ),
        "candidate_disposition": (sel or {}).get("disposition"),
        "event_ids": (sel or {}).get("event_ids"),
        "search_delta_by_fold": (sel or {}).get("search_delta_by_fold")
        or scientific_detail.get("search_deltas"),
        "reeval_delta_by_fold": (sel or {}).get("reeval_delta_by_fold")
        or scientific_detail.get("reeval_deltas"),
        "expected_effect_direction": (sel or {}).get("expected_effect_direction"),
        "control_result": scientific_detail.get("control_result") or prop.get(
            "control_result"
        ),
        "locked_threshold": threshold,
    }


def main() -> int:
    findings: List[Dict[str, Any]] = []
    for cohort, run_dir in RUNS.items():
        case_dirs = sorted(
            d for d in run_dir.iterdir() if (d / "case_edit_proposal.json").exists()
        )
        for case_dir in case_dirs:
            findings.append(_extract_case(cohort, case_dir))

    out = ROOT / "outputs/cf1s_core/ALL55_CASE_FINDINGS.json"
    out.write_text(json.dumps(findings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "WRITTEN", "out": str(out), "count": len(findings)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
