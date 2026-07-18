#!/usr/bin/env python3
"""CF-0 Candidate Funnel Audit driver.

Runs Natural Path A per case into an isolated namespace, writes cf0_v1 per-case
funnels, then aggregates cohort-separated summary + markdown report.

Does NOT change thresholds, action bundles, or retrain models.
Does NOT overwrite Locked P1 outputs under the smoke output_root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.counterfactual.io_utils import file_sha256, write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths
from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import (
    stage_m2_compute_attribution,
    stage_m2_path_a_smoke,
    stage_m2_preflight,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.funnel_accounting import (
    classify_case_execution_error_stage,
)


NORMAL_LABELS = {"정상_운영", "정상", "normal", "NORMAL"}

COHORT_ROLES = {
    "example35_abnormal": "PRIMARY_FUNNEL_AUDIT",
    "example35_normal": "OVER_EDIT_CONTROL",
    "problem20_unlabeled": "BLIND_DIAGNOSTIC_ONLY",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_seed(case_id: str, base_seed: int = 20260718) -> int:
    h = hashlib.sha256(f"{base_seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_case_manifest(labels_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    df = pd.read_csv(labels_path)
    cohorts: Dict[str, List[Dict[str, Any]]] = {
        "example35_abnormal": [],
        "example35_normal": [],
        "problem20_unlabeled": [],
    }
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        set_name = str(row.get("set") or "")
        label = str(row.get("diagnosis_normalized") or row.get("label") or "")
        if set_name == "problem_set":
            # Labels on Problem20 are ignored for CF-0 (blind diagnostic only).
            cohorts["problem20_unlabeled"].append(
                {
                    "case_id": case_id,
                    "cohort": "problem20_unlabeled",
                    "role": COHORT_ROLES["problem20_unlabeled"],
                    "used_for_tuning": False,
                    "label_used": False,
                }
            )
            continue
        if set_name != "example_set":
            continue
        is_normal = label in NORMAL_LABELS
        key = "example35_normal" if is_normal else "example35_abnormal"
        cohorts[key].append(
            {
                "case_id": case_id,
                "cohort": key,
                "role": COHORT_ROLES[key],
                "used_for_tuning": False,
                "label_used": True,
                "label": label,
            }
        )
    return cohorts


def _select_cases(
    cohorts: Dict[str, List[Dict[str, Any]]],
    *,
    include_problem20: bool,
    max_cases: Optional[int],
    case_ids: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    ordered.extend(cohorts["example35_abnormal"])
    ordered.extend(cohorts["example35_normal"])
    if include_problem20:
        ordered.extend(cohorts["problem20_unlabeled"])
    if case_ids:
        want = {str(c) for c in case_ids}
        ordered = [c for c in ordered if c["case_id"] in want]
    if max_cases is not None and max_cases >= 0:
        ordered = ordered[: int(max_cases)]
    return ordered


def _run_one_case(
    *,
    base_cfg: Dict[str, Any],
    case_rec: Dict[str, Any],
    case_out_root: Path,
) -> Dict[str, Any]:
    case_id = case_rec["case_id"]
    cohort = case_rec["cohort"]
    seed = _stable_seed(case_id)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed % (2**32 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed % (2**32 - 1))
    except Exception:
        pass

    cfg = dict(base_cfg)
    cfg["case_id"] = case_id
    # Evaluation cohort for fold context: example vs problem
    if cohort.startswith("example35"):
        cfg["cohort"] = "example_set"
    else:
        cfg["cohort"] = "problem_set"
    # Isolate outputs; never write into Locked smoke root directly.
    shared_output_root = Path(base_cfg.get("output_root") or case_out_root)
    cfg["output_root"] = str(case_out_root)
    # Reuse immutable persisted MLM bundle bank from the Locked/shared root.
    mlm_cfg = dict(cfg.get("mlm") or {})
    if not mlm_cfg.get("bundle_bank_dir"):
        shared_bank = shared_output_root / "artifacts" / "mlm_bundle_bank"
        if shared_bank.exists():
            mlm_cfg["bundle_bank_dir"] = str(shared_bank)
    cfg["mlm"] = mlm_cfg
    # Disable GT-normal nested audit inside path-a smoke for CF-0 driver.
    cfg.pop("normal_case_id", None)

    paths = M1Paths(case_out_root)
    for sub in (paths.artifacts, paths.results, paths.reports, paths.manifests):
        sub.mkdir(parents=True, exist_ok=True)

    try:
        stage_m2_preflight(cfg, paths)
        stage_m2_compute_attribution(cfg, paths)
        path_a = stage_m2_path_a_smoke(cfg, paths)
        funnel_path = paths.artifacts / "cf0_case_funnel.json"
        if funnel_path.exists():
            funnel = json.loads(funnel_path.read_text(encoding="utf-8"))
        else:
            funnel = dict(path_a.get("cf0_funnel") or {})
        funnel["case_id"] = case_id
        funnel["cohort"] = cohort
        funnel["cohort_role"] = case_rec["role"]
        funnel["used_for_tuning"] = bool(case_rec.get("used_for_tuning", False))
        funnel["deterministic_seed"] = seed
        funnel["case_output_root"] = str(case_out_root)
        funnel["run_status"] = "OK"
        return funnel
    except Exception as exc:  # noqa: BLE001 — isolate case failures
        stage = classify_case_execution_error_stage(exc)
        reason = "UNSUPPORTED_CONTINUOUS_BIN_ASSUMPTION" if stage in {
            "FEATURE_BIN_LOOKUP",
            "FEATURE_SCHEMA_DISPATCH",
        } else "UNHANDLED_EXCEPTION"
        return {
            "schema_version": "cf0_v1",
            "case_id": case_id,
            "cohort": cohort,
            "cohort_role": case_rec["role"],
            "used_for_tuning": bool(case_rec.get("used_for_tuning", False)),
            "deterministic_seed": seed,
            "case_output_root": str(case_out_root),
            "run_status": "FAIL_EXECUTION_ERROR",
            "funnel_audit_status": "FAIL_EXECUTION_ERROR",
            "pending_downstream_stages": False,
            "case_execution_error": True,
            "case_execution_error_count": 1,
            "first_execution_error_stage": stage,
            "execution_error_reason": reason,
            "execution_error_reason_counts": {reason: 1},
            # Case-level failure is NOT a candidate terminal
            "candidate_execution_error_count": 0,
            "candidate_rejection_count": 0,
            "deduplicated_candidate_count": 0,
            "valid_cf_count": 0,
            "model_valid_cf_count": 0,
            "observational_sensitivity_valid_count": 0,
            "recommendation_valid_cf_count": 0,
            "selected_non_noop_count": 0,
            "selected_actionable_non_noop_count": 0,
            "first_zero_stage": None,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }


def _aggregate_cohort(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n_all = len(rows)
    ok_rows = [r for r in rows if str(r.get("run_status")) == "OK"]
    n_eval = len(ok_rows)
    exec_fail = [
        r
        for r in rows
        if str(r.get("funnel_audit_status")) == "FAIL_EXECUTION_ERROR"
        or bool(r.get("case_execution_error"))
    ]
    first_zero = Counter(str(r.get("first_zero_stage") or "NONE") for r in ok_rows)
    term_reasons: Counter = Counter()
    exec_reasons: Counter = Counter()
    for r in ok_rows:
        for k, v in (r.get("terminal_reason_counts") or {}).items():
            term_reasons[str(k)] += int(v)
        for k, v in (r.get("execution_error_reason_counts") or {}).items():
            exec_reasons[str(k)] += int(v)
    for r in exec_fail:
        for k, v in (r.get("execution_error_reason_counts") or {}).items():
            exec_reasons[str(k)] += int(v)

    def _mean(key: str, population: Sequence[Mapping[str, Any]]) -> Optional[float]:
        vals = [float(r[key]) for r in population if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    def _coverage(population: Sequence[Mapping[str, Any]], denom: int) -> float:
        if denom <= 0:
            return 0.0
        return (
            sum(1 for r in population if int(r.get("raw_generated_candidate_count") or 0) > 0)
            / denom
        )

    def _valid_case_rate(population: Sequence[Mapping[str, Any]], denom: int, key: str) -> float:
        if denom <= 0:
            return 0.0
        return sum(1 for r in population if int(r.get(key) or 0) >= 1) / denom

    hard_pass_rates = []
    for r in ok_rows:
        dedup = int(r.get("deduplicated_candidate_count") or 0)
        hard = int(r.get("hard_constraint_pass_count") or 0)
        if dedup > 0:
            hard_pass_rates.append(hard / dedup)

    # Dominant bottleneck among OK cases only (never mix execution errors)
    dominant = None
    if first_zero:
        dominant = first_zero.most_common(1)[0][0]

    return {
        "n_cases": n_all,
        "n_ok": n_eval,
        "n_evaluable_cases": n_eval,
        "n_execution_error_cases": len(exec_fail),
        "candidate_generation_coverage_all_cases": _coverage(ok_rows, n_all),
        "candidate_generation_coverage_evaluable_cases": _coverage(ok_rows, n_eval),
        # Back-compat alias (all-cases denominator historically)
        "candidate_generation_coverage": _coverage(ok_rows, n_all),
        "mean_raw_generated_candidate_count": _mean("raw_generated_candidate_count", ok_rows),
        "mean_deduplicated_candidates_per_all_case": (
            float(
                sum(int(r.get("deduplicated_candidate_count") or 0) for r in rows) / n_all
            )
            if n_all
            else None
        ),
        "mean_deduplicated_candidates_per_evaluable_case": _mean(
            "deduplicated_candidate_count", ok_rows
        ),
        "mean_deduplicated_candidate_count": _mean("deduplicated_candidate_count", ok_rows),
        "mean_hard_constraint_pass_rate": (
            float(sum(hard_pass_rates) / len(hard_pass_rates)) if hard_pass_rates else None
        ),
        "valid_cf_case_rate_all_cases": _valid_case_rate(ok_rows, n_all, "valid_cf_count"),
        "valid_cf_case_rate_evaluable_cases": _valid_case_rate(
            ok_rows, n_eval, "valid_cf_count"
        ),
        "valid_cf_case_rate": _valid_case_rate(ok_rows, n_all, "valid_cf_count"),
        "recommendation_valid_cf_case_rate_all_cases": _valid_case_rate(
            ok_rows, n_all, "recommendation_valid_cf_count"
        ),
        "recommendation_valid_cf_case_rate_evaluable_cases": _valid_case_rate(
            ok_rows, n_eval, "recommendation_valid_cf_count"
        ),
        "sum_model_valid_cf_count": sum(int(r.get("model_valid_cf_count") or r.get("valid_cf_count") or 0) for r in ok_rows),
        "sum_observational_sensitivity_valid_count": sum(
            int(r.get("observational_sensitivity_valid_count") or 0) for r in ok_rows
        ),
        "sum_recommendation_valid_cf_count": sum(
            int(r.get("recommendation_valid_cf_count") or 0) for r in ok_rows
        ),
        "sum_selected_actionable_non_noop_count": sum(
            int(r.get("selected_actionable_non_noop_count") or 0) for r in ok_rows
        ),
        "first_zero_stage_counts": dict(first_zero),
        "terminal_reason_counts": dict(term_reasons),
        "execution_error_reason_counts": dict(exec_reasons),
        "dominant_bottleneck": dominant,
        "dominant_bottleneck_status": (
            "PENDING_NO_OK_CASES" if not ok_rows else "OBSERVED"
        ),
        "mean_n_mlm_raw_proposals": _mean("n_mlm_raw_proposals", ok_rows),
        "mean_n_mlm_origin_candidates": _mean("n_mlm_origin_candidates", ok_rows),
        "mean_n_valid_mlm_cf": _mean("n_valid_mlm_cf", ok_rows),
        "monotonicity_pass_rate": (
            sum(1 for r in ok_rows if r.get("monotonicity_pass")) / len(ok_rows)
            if ok_rows
            else None
        ),
        "conservation_pass_rate": (
            sum(1 for r in ok_rows if r.get("conservation_pass")) / len(ok_rows)
            if ok_rows
            else None
        ),
    }


def _write_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    per_case: Sequence[Mapping[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# CF-0 Candidate Funnel Audit Report")
    lines.append("")
    lines.append(f"- created_at: `{manifest.get('created_at')}`")
    lines.append(f"- audit_id: `{manifest.get('audit_id')}`")
    lines.append(f"- config_sha256: `{manifest.get('config_sha256')}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Diagnostic only: no threshold relaxation, no action-bundle change, no retrain.")
    lines.append("- Single smoke-case facts are NOT generalized as cohort dominant bottleneck.")
    lines.append("")
    primary_dom = ((summary.get("cohorts") or {}).get("example35_abnormal") or {}).get(
        "dominant_bottleneck"
    )
    lines.append("### Smoke-case vs cohort bottleneck")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "single_case_confirmed": {
                    "provenance_attribution_bug": True,
                    "materiality_bottleneck": True,
                },
                "cohort_dominant_bottleneck_example35_abnormal_ok_cases": primary_dom,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("## Cohort summaries")
    lines.append("")
    for cohort, role in COHORT_ROLES.items():
        block = (summary.get("cohorts") or {}).get(cohort) or {}
        if not block:
            continue
        lines.append(f"### `{cohort}` ({role})")
        lines.append("")
        lines.append(f"- n_cases: {block.get('n_cases')}")
        lines.append(f"- n_ok: {block.get('n_ok')}")
        lines.append(f"- n_execution_error_cases: {block.get('n_execution_error_cases')}")
        lines.append(
            f"- dominant_bottleneck: `{block.get('dominant_bottleneck')}` "
            f"({block.get('dominant_bottleneck_status')})"
        )
        lines.append(
            f"- candidate_generation_coverage: {block.get('candidate_generation_coverage')}"
        )
        lines.append(
            f"- mean_deduplicated_candidate_count: {block.get('mean_deduplicated_candidate_count')}"
        )
        lines.append(
            f"- mean_hard_constraint_pass_rate: {block.get('mean_hard_constraint_pass_rate')}"
        )
        lines.append(f"- valid_cf_case_rate: {block.get('valid_cf_case_rate')}")
        lines.append(
            f"- first_zero_stage_counts: `{json.dumps(block.get('first_zero_stage_counts') or {}, ensure_ascii=False)}`"
        )
        lines.append(
            f"- terminal_reason_counts: `{json.dumps(block.get('terminal_reason_counts') or {}, ensure_ascii=False)}`"
        )
        lines.append(
            f"- execution_error_reason_counts: `{json.dumps(block.get('execution_error_reason_counts') or {}, ensure_ascii=False)}`"
        )
        lines.append("")
    lines.append("## Initial target metrics (§14) vs observed (primary cohort)")
    lines.append("")
    primary = (summary.get("cohorts") or {}).get("example35_abnormal") or {}
    lines.append("| Metric | Initial target | Observed (example35_abnormal) |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Candidate generation coverage | ≥80% | "
        f"{(primary.get('candidate_generation_coverage') or 0)*100:.1f}% |"
    )
    lines.append(
        f"| Mean pre-candidates / case | ≥20 | "
        f"{primary.get('mean_deduplicated_candidate_count')} |"
    )
    mhr = primary.get("mean_hard_constraint_pass_rate")
    lines.append(
        f"| Hard constraint pass rate | ≥20% | "
        f"{(mhr*100 if mhr is not None else float('nan')):.1f}% |"
    )
    lines.append(
        f"| Cases with ≥1 Valid CF | ≥30% | "
        f"{(primary.get('valid_cf_case_rate') or 0)*100:.1f}% |"
    )
    lines.append("")
    lines.append("## Per-case first-zero stage")
    lines.append("")
    lines.append("| case_id | cohort | first_zero_stage | valid_cf | selected_non_noop | status |")
    lines.append("|---|---|---|---:|---:|---|")
    for r in per_case:
        lines.append(
            f"| `{r.get('case_id')}` | {r.get('cohort')} | "
            f"`{r.get('first_zero_stage')}` | {r.get('valid_cf_count', 0)} | "
            f"{r.get('selected_non_noop_count', 0)} | {r.get('funnel_audit_status') or r.get('run_status')} |"
        )
    lines.append("")
    lines.append("## Execution errors (separated from candidate rejection)")
    lines.append("")
    exec_rows = [
        r
        for r in per_case
        if str(r.get("funnel_audit_status")) == "FAIL_EXECUTION_ERROR"
        or str(r.get("run_status")) == "FAIL_EXECUTION_ERROR"
    ]
    if not exec_rows:
        lines.append("- None.")
    else:
        lines.append("| case_id | cohort | error |")
        lines.append("|---|---|---|")
        for r in exec_rows:
            err = str(r.get("error") or "")
            lines.append(f"| `{r.get('case_id')}` | {r.get('cohort')} | `{err}` |")
        lines.append("")
        lines.append(
            "- These are schema/runtime failures (often non-binned features such as "
            "`wind_direction_deg` circular encoding or `rain_detected`), not normal "
            "candidate rejections. They are excluded from dominant-bottleneck counting."
        )
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    lines.append(
        "- `CIRCULAR_FEATURE_LINEAR_BIN_UNAVAILABLE` (`wind_direction_deg`): not fixed in CF-0; "
        "when locus selection lands on this feature, Path A raises and the case is "
        "`FAIL_EXECUTION_ERROR` (blocking for that case)."
    )
    lines.append(
        "- Example35 uses OOF evaluation (`cf_valid` is not independently evaluable); "
        "primary cohort `valid_cf_case_rate=0` therefore mixes evaluation-mode limits with "
        "materiality failures. Problem20 (crossfit) is the independent Valid-CF check."
    )
    lines.append(
        "- Problem20 results are `BLIND_DIAGNOSTIC_ONLY` and must not be used for tuning."
    )
    lines.append("")
    lines.append("## Conclusion for next phase")
    lines.append("")
    primary = (summary.get("cohorts") or {}).get("example35_abnormal") or {}
    p20 = (summary.get("cohorts") or {}).get("problem20_unlabeled") or {}
    lines.append(
        f"- Primary cohort dominant bottleneck (OK cases): "
        f"`{primary.get('dominant_bottleneck')}`."
    )
    lines.append(
        f"- Blind Problem20 dominant bottleneck (OK cases): "
        f"`{p20.get('dominant_bottleneck')}`."
    )
    lines.append(
        "- MLM-origin candidates exist after provenance separation, but "
        "`n_valid_mlm_cf` remains 0 under current materiality/ε and (for Example35) OOF rules."
    )
    lines.append(
        "- CF-0 stops here. Threshold relaxation / action-bundle work require separate approval."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="CF-0 Candidate Funnel Audit")
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml",
    )
    ap.add_argument(
        "--audit-root",
        type=Path,
        default=ROOT / "outputs/m1_m2_prereq/funnel_audit",
    )
    ap.add_argument("--include-problem20", action="store_true", default=True)
    ap.add_argument("--no-problem20", action="store_true", default=False)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--case-id", action="append", default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg_text = args.config.read_text(encoding="utf-8")
    base_cfg = yaml.safe_load(cfg_text) or {}
    if args.device:
        base_cfg["device"] = args.device
    config_sha = _sha_text(cfg_text)

    labels_path = Path(base_cfg["labels_path"])
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    cohorts = build_case_manifest(labels_path)
    include_p20 = bool(args.include_problem20) and not bool(args.no_problem20)
    selected = _select_cases(
        cohorts,
        include_problem20=include_p20,
        max_cases=args.max_cases,
        case_ids=args.case_id,
    )
    if not selected:
        print("No cases selected", file=sys.stderr)
        return 2

    audit_id = f"cf0_funnel_{_utc_now()}"
    audit_root = Path(args.audit_root)
    per_case_dir = audit_root / "per_case"
    runs_dir = audit_root / "case_runs" / audit_id
    per_case_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        "config_sha256": config_sha,
        "labels_sha256": file_sha256(labels_path) if labels_path.exists() else None,
        "vocabulary_sha256": file_sha256(Path(base_cfg["vocabulary_path"]))
        if Path(base_cfg["vocabulary_path"]).exists()
        else None,
        "binning_registry_sha256": file_sha256(Path(base_cfg["binning_registry_path"]))
        if Path(base_cfg["binning_registry_path"]).exists()
        else None,
        "stage_a_ckpt_sha256": file_sha256(Path(base_cfg["stage_a_ckpt"]))
        if Path(base_cfg["stage_a_ckpt"]).exists()
        else None,
    }

    manifest = {
        "audit_id": audit_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "cf0_v1",
        "config_path": str(args.config),
        "config_sha256": config_sha,
        "labels_path": str(labels_path),
        "input_hashes": input_hashes,
        "cohorts": {
            k: {
                "role": COHORT_ROLES[k],
                "n": len(v),
                "used_for_tuning": False,
                "case_ids": [c["case_id"] for c in v],
            }
            for k, v in cohorts.items()
        },
        "selected_cases": selected,
        "rules": {
            "immutable_artifacts_loaded_per_case_via_shared_paths": True,
            "case_independent_output_namespace": True,
            "deterministic_seed": True,
            "locked_output_overwrite_forbidden": True,
            "case_failure_isolation": True,
            "summary_readonly_after_per_case": True,
            "threshold_relaxation": False,
            "action_bundle_change": False,
            "model_retraining": False,
        },
    }
    write_json(audit_root / "audit_manifest.json", manifest)

    per_case_results: List[Dict[str, Any]] = []
    for i, case_rec in enumerate(selected, 1):
        case_id = case_rec["case_id"]
        print(f"[{i}/{len(selected)}] CF-0 funnel audit: {case_id} ({case_rec['cohort']})", flush=True)
        case_out = runs_dir / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        result = _run_one_case(base_cfg=base_cfg, case_rec=case_rec, case_out_root=case_out)
        out_path = per_case_dir / f"{case_id}.json"
        write_json(out_path, result)
        result["per_case_json"] = str(out_path)
        result["per_case_sha256"] = file_sha256(out_path)
        # rewrite with sha
        write_json(out_path, result)
        per_case_results.append(result)

    # Read-only aggregate from written JSON files
    loaded: List[Dict[str, Any]] = []
    for case_rec in selected:
        p = per_case_dir / f"{case_rec['case_id']}.json"
        loaded.append(json.loads(p.read_text(encoding="utf-8")))

    by_cohort: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in loaded:
        by_cohort[str(r.get("cohort"))].append(r)

    summary = {
        "audit_id": audit_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "cf0_v1",
        "n_cases_total": len(loaded),
        "cohorts": {k: _aggregate_cohort(v) for k, v in by_cohort.items()},
        "smoke_case_facts": {
            "provenance_attribution_bug": True,
            "materiality_bottleneck": True,
            "note": "Confirmed on prior single smoke case only; see cohort dominant_bottleneck.",
        },
        "initial_targets": {
            "candidate_generation_coverage": 0.80,
            "mean_pre_candidates": 20,
            "hard_constraint_pass_rate": 0.20,
            "valid_cf_case_rate": 0.30,
        },
    }
    write_json(audit_root / "funnel_audit_summary.json", summary)
    _write_report(
        audit_root / "FUNNEL_AUDIT_REPORT.md",
        manifest=manifest,
        summary=summary,
        per_case=loaded,
    )
    print(
        json.dumps(
            {
                "status": "CF0_FUNNEL_AUDIT_COMPLETE",
                "audit_id": audit_id,
                "n_cases": len(loaded),
                "summary": str(audit_root / "funnel_audit_summary.json"),
                "report": str(audit_root / "FUNNEL_AUDIT_REPORT.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
