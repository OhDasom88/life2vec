"""CF-1S Core pipeline — SALIENCY_TOP_K + EDITABLE_CONTROL_LIKE only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .cf1s.core_acceptance import (
    evaluate_candidate_status,
    evaluate_case_multi_event_status,
    evaluate_cohort_status,
)
from .cf1s.core_candidate_family import build_complete_families, validate_family_membership
from .cf1s.core_contract import CoreContractError, derive_thresholds_from_noise, sha256_json
from .cf1s.core_identity import (
    check_noise_ceilings,
    deltas_from_baseline,
    evaluate_forced_identity,
    forced_identity_tolerances,
    run_pure_repeat_baseline,
)
from .cf1s.core_manifest import (
    FoldForwardTrace,
    assert_holdout_matches_closure,
    freeze_evaluation_closure,
)


def _synthetic_control_atomics(feature_id: str = "circulation_fan") -> List[Dict[str, Any]]:
    """Deterministic atomics for contract tests / dry-run."""
    base = []
    for i, t in enumerate([0, 1800, 7200, 10800]):
        base.append(
            {
                "event_id": f"E_{feature_id}_{i}",
                "feature_id": feature_id,
                "edit_direction": "DOWN",
                "target_raw": 0.0,
                "schema_target": "bin0",
                "sequence_position": i,
                "event_time_epoch": float(t),
                "to_tokens": [f"TOK_{feature_id}_LOW"],
                "sentence_tokens": [f"TOK_{feature_id}_LOW"],
            }
        )
    return base


def run_cf1s_core_case_contract(
    *,
    case_id: str,
    search_fold_ids: Sequence[int],
    holdout_fold_ids: Sequence[int],
    atomics: Optional[Sequence[Mapping[str, Any]]] = None,
    search_bundle_deltas: Optional[Sequence[float]] = None,
    holdout_bundle_deltas: Optional[Sequence[float]] = None,
    search_parent_deltas: Optional[Sequence[Mapping[str, float]]] = None,
    holdout_parent_deltas: Optional[Sequence[Mapping[str, float]]] = None,
    baseline_risk_repeats_by_fold: Optional[Mapping[str, Sequence[float]]] = None,
    effect_threshold: float = 0.001,
    incremental_threshold: float = 0.001,
) -> Dict[str, Any]:
    """Pure contract-level case run used by unit tests and dry pipeline."""
    atomics = list(atomics or _synthetic_control_atomics())
    families = build_complete_families(atomics)
    trace = FoldForwardTrace()

    # Identity: requested search folds only first
    def _forward_factory(risks_by_fold: Mapping[str, Sequence[float]]):
        state = {k: 0 for k in risks_by_fold}

        def _fwd(fold_id: int):
            key = str(int(fold_id))
            seq = list(risks_by_fold[key])
            idx = state[key] % len(seq)
            state[key] += 1
            r = float(seq[idx])
            return {"risk": r, "logit": [0.0, r]}

        return _fwd

    risks = baseline_risk_repeats_by_fold or {
        str(f): [0.20, 0.20, 0.20] for f in list(search_fold_ids) + list(holdout_fold_ids)
    }
    search_base = run_pure_repeat_baseline(
        requested_fold_ids=search_fold_ids,
        forward_fn=_forward_factory({str(f): risks[str(f)] for f in search_fold_ids}),
    )
    trace.record(scope="search", fold_ids=search_fold_ids)
    # freeze closure from family candidates
    cand_list = []
    for fam_i, fam in enumerate(families["two_event_families"]):
        for label, body in fam["candidates"].items():
            cand_list.append(
                {
                    "candidate_id": f"{case_id}:2:{fam_i}:{label}",
                    "case_id": case_id,
                    "family_id": f"2:{fam_i}",
                    "label": label,
                    "event_ids": body.get("keep_event_ids"),
                    "atomic_payload_hashes": [body.get("parent_atomic_payload_hash")],
                    "raw_transaction_hash": body.get("parent_raw_transaction_hash"),
                    "parent_ids": [],
                    "retokenization_hash": body.get("parent_raw_transaction_hash"),
                    "n_events": len(body.get("keep_event_ids") or []),
                }
            )
    closure = freeze_evaluation_closure(
        cohort="development", case_id=case_id, candidates=cand_list
    )
    trace.mark_closure_frozen()
    holdout_base = run_pure_repeat_baseline(
        requested_fold_ids=holdout_fold_ids,
        forward_fn=_forward_factory({str(f): risks[str(f)] for f in holdout_fold_ids}),
    )
    trace.record(scope="holdout", fold_ids=holdout_fold_ids)
    assert_holdout_matches_closure(closure=closure, holdout_candidates=cand_list)

    case_risk_noise = max(
        search_base["case_pure_repeat_risk_noise"],
        holdout_base["case_pure_repeat_risk_noise"],
    )
    case_logit_noise = max(
        search_base["case_pure_repeat_logit_noise"],
        holdout_base["case_pure_repeat_logit_noise"],
    )
    noise = check_noise_ceilings(
        case_pure_repeat_risk_noise=case_risk_noise,
        case_pure_repeat_logit_noise=case_logit_noise,
    )
    tols = forced_identity_tolerances(
        case_pure_repeat_risk_noise=case_risk_noise,
        case_pure_repeat_logit_noise=case_logit_noise,
    )
    # Forced identity: identical risks
    identity_by_fold = {}
    for base in (search_base, holdout_base):
        for fid, row in base["by_fold"].items():
            identity_by_fold[fid] = {
                "risk": row["baseline_reference_risk"],
                "logit": row["baseline_reference_logit"],
            }
    merged_base = {
        "by_fold": {**search_base["by_fold"], **holdout_base["by_fold"]},
    }
    identity = evaluate_forced_identity(
        baseline=merged_base,
        identity_by_fold=identity_by_fold,
        risk_tolerance=tols["case_forced_identity_risk_tolerance"],
        logit_tolerance=tols["case_forced_identity_logit_tolerance"],
    )

    # Default synthetic deltas for one AB bundle if not provided
    s_folds = list(search_fold_ids)
    h_folds = list(holdout_fold_ids)
    if search_bundle_deltas is None:
        search_bundle_deltas = [0.02] * len(s_folds)
    if holdout_bundle_deltas is None:
        holdout_bundle_deltas = [0.02] * len(h_folds)
    if search_parent_deltas is None:
        search_parent_deltas = [{"A": 0.0, "B": 0.0} for _ in s_folds]
    if holdout_parent_deltas is None:
        holdout_parent_deltas = [{"A": 0.0, "B": 0.0} for _ in h_folds]

    two_statuses = []
    if families["two_event_family_evaluable"] and noise["noise_ceiling_pass"] and identity[
        "forced_identity_all_requested_folds_pass"
    ]:
        cand = evaluate_candidate_status(
            execution_complete=True,
            parent_complete=True,
            fold_execution_failed=False,
            search_bundle_deltas=search_bundle_deltas,
            holdout_bundle_deltas=holdout_bundle_deltas,
            search_incremental_parents=search_parent_deltas,
            holdout_incremental_parents=holdout_parent_deltas,
            search_single_parents=search_parent_deltas,
            holdout_single_parents=holdout_parent_deltas,
            effect_threshold=effect_threshold,
            incremental_threshold=incremental_threshold,
            n_events=2,
        )
        two_statuses.append(cand["candidate_scientific_status"])
    elif not noise["noise_ceiling_pass"] or not identity["forced_identity_all_requested_folds_pass"]:
        two_statuses.append("NOT_EVALUABLE")
    elif not families["two_event_family_evaluable"]:
        two_statuses.append("NOT_EVALUABLE")

    three_statuses: List[str] = []
    if families["three_event_family_evaluable"]:
        # reported separately; default not evaluated scientifically in contract dry-run
        three_statuses.append("NOT_EVALUABLE")

    case_status = evaluate_case_multi_event_status(
        two_event_candidate_statuses=two_statuses,
        three_event_candidate_statuses=three_statuses,
    )
    exec_status = (
        "PASS"
        if identity["forced_identity_all_requested_folds_pass"] and noise["noise_ceiling_pass"]
        else "NOT_EVALUABLE"
    )
    if identity["case_execution_status"] == "NOT_EVALUABLE":
        exec_status = "NOT_EVALUABLE"

    return {
        "case_id": case_id,
        "selector": "SALIENCY_TOP_K",
        "arm": "EDITABLE_CONTROL_LIKE",
        "reencode_mode": "FULL_SEQUENCE_COLD_REBUILD",
        "execution_status": exec_status,
        "families": {
            "two_event_family_evaluable": families["two_event_family_evaluable"],
            "three_event_family_evaluable": families["three_event_family_evaluable"],
            "n_two_event_families": len(families["two_event_families"]),
            "n_three_event_families": len(families["three_event_families"]),
        },
        "identity": {
            "search_baseline": search_base,
            "holdout_baseline": holdout_base,
            "noise": noise,
            "forced_identity": identity,
            "tolerances": tols,
        },
        "closure": closure,
        "forward_trace": trace.summarize(
            search_fold_ids=search_fold_ids, holdout_fold_ids=holdout_fold_ids
        ),
        "case_status": case_status,
        "primary_multi_event_edit_feasibility_status": case_status[
            "case_overall_scientific_status"
        ],
    }


def run_cf1s_core_cohort_contract(
    *,
    case_ids: Sequence[str],
    search_fold_ids: Sequence[int],
    holdout_fold_ids: Sequence[int],
    locked_cohort_case_count: int,
    minimum_evaluable_case_count: int,
    minimum_supported_case_count: int,
    minimum_supported_fraction: float,
) -> Dict[str, Any]:
    per_case = []
    noises = []
    for cid in case_ids:
        result = run_cf1s_core_case_contract(
            case_id=cid,
            search_fold_ids=search_fold_ids,
            holdout_fold_ids=holdout_fold_ids,
        )
        per_case.append(result)
        noises.append(result["identity"]["noise"]["case_pure_repeat_risk_noise"])
    max_noise = max(noises) if noises else 0.0
    try:
        thresholds = derive_thresholds_from_noise(max_noise)
        noise_ceiling_pass = True
    except CoreContractError:
        thresholds = {}
        noise_ceiling_pass = False
    statuses = [c["case_status"]["case_overall_scientific_status"] for c in per_case]
    cohort = evaluate_cohort_status(
        statuses,
        locked_cohort_case_count=locked_cohort_case_count,
        minimum_evaluable_case_count=minimum_evaluable_case_count,
        minimum_supported_case_count=minimum_supported_case_count,
        minimum_supported_fraction=minimum_supported_fraction,
    )
    return {
        "per_case": per_case,
        "thresholds": thresholds,
        "noise_ceiling_pass": noise_ceiling_pass,
        "cohort": cohort,
        "case_scientific_statuses": statuses,
        "execution_status": (
            "PASS"
            if all(c["execution_status"] == "PASS" for c in per_case) and noise_ceiling_pass
            else "FAIL"
        ),
    }
