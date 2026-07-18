#!/usr/bin/env python3
"""Post-hoc bidirectional epsilon sensitivity (diagnostic only; does not retune)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json

DEFAULT_GRID = [0.01, 0.001, 0.0005, 0.0001]


def _noise_floor(calib_path: Path) -> float:
    if not calib_path.exists():
        return 8.374e-7
    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    for key in ("noise_floor", "noop_noise_floor", "delta_r_noise_floor"):
        if payload.get(key) is not None:
            return float(payload[key])
    # nested
    for nest in ("stats", "summary", "calibration"):
        block = payload.get(nest) or {}
        if isinstance(block, dict) and block.get("noise_floor") is not None:
            return float(block["noise_floor"])
    return float(payload.get("effective_noise") or 8.374e-7)


def _iter_case_candidates(case_run_root: Path, case_id: str) -> List[Dict[str, Any]]:
    # Prefer CF0R recovery runs, else parent archive layout
    candidates = []
    for base in sorted(case_run_root.glob("**/path_a_m2_cf_results.jsonl")):
        if case_id not in str(base):
            continue
        for line in base.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("is_noop")):
                continue
            candidates.append(row)
        break
    return candidates


def _fold_agree_decrease(row: Mapping[str, Any]) -> bool:
    folds = row.get("delta_r_folds_search") or row.get("delta_r_folds")
    if not folds:
        return False
    try:
        return all(float(x) < 0.0 for x in list(folds))
    except Exception:
        return False


def _fold_agree_increase(row: Mapping[str, Any]) -> bool:
    folds = row.get("delta_r_folds_search") or row.get("delta_r_folds")
    if not folds:
        return False
    try:
        return all(float(x) > 0.0 for x in list(folds))
    except Exception:
        return False


def analyze_cases(
    *,
    per_case_dir: Path,
    case_run_roots: Sequence[Path],
    eps_grid: Sequence[float],
    noise: float,
) -> Dict[str, Any]:
    cases = sorted(per_case_dir.glob("*.json"))
    by_eps: Dict[str, Any] = {}
    for eps in eps_grid:
        dec_cands = 0
        inc_cands = 0
        dec_cases = set()
        inc_cases = set()
        by_cohort_dec = defaultdict(int)
        by_cohort_inc = defaultdict(int)
        for cf in cases:
            case_id = cf.stem
            meta = json.loads(cf.read_text(encoding="utf-8"))
            cohort = str(meta.get("cohort") or "")
            rows: List[Dict[str, Any]] = []
            for root in case_run_roots:
                rows = _iter_case_candidates(root, case_id)
                if rows:
                    break
            seen = set()
            for row in rows:
                cid = str(row.get("dedup_candidate_id") or row.get("candidate_id") or "")
                if cid in seen:
                    continue
                seen.add(cid)
                # Observational sensitivity never recommendation-eligible
                if str(row.get("operational_eligibility")) == "OBSERVATIONAL_SENSITIVITY_ONLY":
                    continue
                if str(row.get("source")) == "schema_categorical_adjacent":
                    continue
                dr = row.get("delta_r_search")
                if dr is None:
                    continue
                drf = float(dr)
                if drf <= -float(eps) and _fold_agree_decrease(row):
                    dec_cands += 1
                    dec_cases.add(case_id)
                    by_cohort_dec[cohort] += 1
                if drf >= float(eps) and _fold_agree_increase(row):
                    inc_cands += 1
                    inc_cases.add(case_id)
                    by_cohort_inc[cohort] += 1
        by_eps[str(eps)] = {
            "epsilon": float(eps),
            "risk_decrease_candidate_count": dec_cands,
            "risk_decrease_case_count": len(dec_cases),
            "risk_increase_candidate_count": inc_cands,
            "risk_increase_case_count": len(inc_cases),
            "recommendation_eligible_count": 0,
            "by_cohort_decrease_candidates": dict(by_cohort_dec),
            "by_cohort_increase_candidates": dict(by_cohort_inc),
            "absolute_effect": float(eps),
            "noise_multiple": float(eps) / noise if noise > 0 else None,
            "operational_materiality": False,
            "causal_interpretation_allowed": False,
        }

    return {
        "current_operational_epsilon": 0.01,
        "epsilon_policy_status": "UNCHANGED_NOT_RETUNED",
        "sensitivity_grid_role": "POST_HOC_DIAGNOSTIC_ONLY",
        "noise_floor": noise,
        "grid": by_eps,
        "provisional_note": (
            "Counts exclude observational categorical candidates from recommendation "
            "eligibility; threshold is not selected from this grid."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case-dir", type=Path, required=True)
    ap.add_argument("--case-run-root", type=Path, action="append", default=[])
    ap.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "outputs/cf_calibration/noop_noise.json",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    noise = _noise_floor(Path(args.calibration))
    grid = list(DEFAULT_GRID) + [noise]
    report = analyze_cases(
        per_case_dir=Path(args.per_case_dir),
        case_run_roots=list(args.case_run_root or []),
        eps_grid=grid,
        noise=noise,
    )
    write_json(Path(args.out), report)
    print(json.dumps({"status": "EPS_SENSITIVITY_WRITTEN", "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
