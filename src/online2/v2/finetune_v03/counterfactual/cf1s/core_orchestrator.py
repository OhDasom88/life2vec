"""Cohort orchestrator wiring for Development3 production evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .core_authorization import (
    compute_evidence_package_root,
    consume_pre_execution_authorization,
    issue_post_execution_attestation,
    load_trust_root,
    payload_sha256,
    verify_trust_root_env,
)
from .core_canonical import canonical_json_dumps, canonical_json_sha256
from .core_contract import CoreContractError, sha256_file
from .core_development import (
    DevelopmentPhase,
    PhaseStateMachine,
    apply_dependency_budget,
    build_selection_blind_closure,
    development_readiness_axes,
    evaluate_bundle_effects,
)
from .core_edit_proposal import (
    AuthorityContext,
    ConstructionStatus,
    EvidenceKind,
    EvaluationStage,
    ExecutionEvidenceFlags,
    ForwardKind,
    OrderingStatus,
    ScientificStatus,
)
from .core_locks import (
    assert_stable_locks_identical,
    compute_runtime_observation,
    derive_attested_coverage,
    recompute_stable_lock_manifest,
    verify_post_attestation_public_key_only,
)
from .core_output import (
    build_candidate_set_manifest,
    build_case_edit_proposal_envelope,
    write_case_proposal_json,
)
from .core_production import (
    ForwardCounters,
    attribution_policy_lock,
    evaluate_scope_gate,
    lock_thresholds_from_search_baseline_noise,
    make_production_fold_forward_fn,
    run_cold_forward_pipeline,
)
from .core_promotion import (
    PROMOTED,
    PROMOTED_BUT_UNATTESTED_BY_RECEIPT,
    PROMOTED_BUT_UNVERIFIED,
    QUARANTINE_ONLY,
    atomic_promote_directory,
    package_state_after_receipt,
    readback_final_package,
    validate_promotion_receipt,
    write_promotion_receipt,
)
from .core_raw_transaction import (
    ValidatedRawTransaction,
    build_identity_transaction,
    reject_unchecked_token_dict,
    verify_exact_parent_transactions,
)
from .core_runtime import (
    apply_exact_byte_deterministic_runtime,
    case_fold_provenance,
    load_fold_routing_manifest,
)
from .core_scientific import (
    check_risk_logit_contract,
    control_result_from_delta,
    derive_coverage_axes,
    max_search_identity_reconstruction_error,
    runtime_repeat_noise,
    select_observed_median,
)
from .core_selection import (
    SELECTED_CONTROL_NO_MATERIAL,
    SELECTED_MATERIAL,
    case_coverage_eligible,
    classify_control_reeval,
    select_after_search,
)
from .core_trace import GlobalForwardTrace
from .core_verifier import (
    FINAL,
    PRE_PROMOTION,
    compute_verifier_code_sha256,
    verify_cf1s_package,
)


@dataclass
class CaseRuntimeHooks:
    """Injectable hooks so unit tests can mock GPU/data without production_bridge."""

    load_case_events: Callable[[str], list]
    load_sidecar_by_event_id: Callable[[str], Mapping[str, Mapping[str, Any]]]
    build_reencoder: Callable[[], Any]
    list_checkpoint_paths: Callable[[], Sequence[Path]]
    fold_ids: Sequence[int]
    build_candidates: Callable[[str], List[Mapping[str, Any]]]
    # Returns ValidatedRawTransaction for identity/baseline
    score_identity_transaction: Callable[[str], ValidatedRawTransaction]
    bind_global_trace: Optional[Callable[[GlobalForwardTrace], None]] = None


@dataclass
class OrchestratorResult:
    execution_readiness: str
    evaluation_coverage: str
    attested_evaluation_coverage: str
    scientific_counts: Dict[str, int]
    case_results: List[Dict[str, Any]]
    package_dir: Optional[Path]
    package_state: str
    report_lines: List[str] = field(default_factory=list)
    gate_results: Dict[str, str] = field(default_factory=dict)


def _atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    from .core_promotion import fsync_dir, fsync_file
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json_dumps(obj) + "\n", encoding="utf-8")
    fsync_file(tmp)
    os.replace(str(tmp), str(path))
    fsync_file(path)
    fsync_dir(path.parent)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    from .core_promotion import fsync_dir, fsync_file
    import os

    source = Path(source)
    if not source.is_file():
        raise CoreContractError(f"snapshot source missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_bytes(source.read_bytes())
    fsync_file(tmp)
    os.replace(str(tmp), str(destination))
    fsync_file(destination)
    fsync_dir(destination.parent)


def _fold_deltas(scores: Mapping[str, Any], baseline: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for fid, row in (scores.get("by_fold") or {}).items():
        if fid not in baseline:
            continue
        out[str(fid)] = float(row["risk"]) - float(baseline[fid])
    return out


def _fold_risks(scores: Mapping[str, Any]) -> Dict[str, float]:
    return {str(fid): float(row["risk"]) for fid, row in (scores.get("by_fold") or {}).items()}


def _fold_logits(scores: Mapping[str, Any]) -> Dict[str, List[float]]:
    return {
        str(fid): [float(x) for x in (row.get("logit") or [])]
        for fid, row in (scores.get("by_fold") or {}).items()
    }


def _abnormal_logit(row: Mapping[str, Any]) -> float:
    logits = list(row.get("logit") or [])
    if len(logits) != 1:
        raise CoreContractError("abnormal binary logit must contain exactly one value")
    value = float(logits[0])
    check_risk_logit_contract(float(row["risk"]), value)
    return value


def _fold_evidence_rows(scores: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not scores:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    trace_ref = scores.get("trace_invocation_id")
    for fold_id, raw in (scores.get("scores") or {}).get("by_fold", {}).items():
        row = dict(raw)
        if not row.get("checkpoint_sha256"):
            raise CoreContractError(f"fold {fold_id} checkpoint_sha256 missing")
        if not row.get("semantic_critic_input_sha"):
            raise CoreContractError(
                f"fold {fold_id} semantic_critic_input_sha missing"
            )
        if not trace_ref:
            raise CoreContractError(f"fold {fold_id} trace reference missing")
        result[str(fold_id)] = {
            "risk": float(row["risk"]),
            "abnormal_logit": _abnormal_logit(row),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "semantic_critic_input_sha": row.get("semantic_critic_input_sha"),
            "trace_ref": trace_ref,
            "trace_invocation_id": trace_ref,
        }
    return result


def _ledger_row(
    *,
    candidate: Mapping[str, Any],
    search_scores: Optional[Mapping[str, Any]],
    reeval_scores: Optional[Mapping[str, Any]],
    baseline_med: Mapping[str, float],
    reeval_base_risk: Mapping[str, float],
    locked_thr: float,
    direction: Optional[str],
    disposition: Optional[str],
    baseline_observations: Optional[Mapping[str, Mapping[str, Any]]] = None,
    reeval_baseline_scores: Optional[Mapping[str, Any]] = None,
    failure_reason: Optional[str] = None,
    parent_transaction_shas: Optional[Sequence[str]] = None,
    incremental: Optional[Mapping[str, float]] = None,
    residual: Optional[Mapping[str, float]] = None,
    trace_refs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    tx = candidate.get("validated_transaction")
    tx_sha = candidate.get("transaction_sha")
    validated_transaction_payload: Optional[Dict[str, Any]] = None
    canonical_atomic_set_payload: Optional[Dict[str, Any]] = None
    if isinstance(tx, ValidatedRawTransaction):
        tx_sha = tx.canonical_validated_transaction_sha
        canonical_atomics = [
            atomic.to_dict() for atomic in sorted(tx.atomics, key=lambda item: item.sort_key())
        ]
        validated_transaction_payload = {
            "case_id": tx.case_id,
            "atomics": canonical_atomics,
            "allowed_change_set_sha": tx.allowed_change_set.sha256,
            "original_caseevents_sha": tx.original_caseevents_sha,
            "transaction_mode": tx.transaction_mode,
            "identity": bool(tx.identity),
        }
        canonical_atomic_set_payload = {
            "case_id": tx.case_id,
            "atomics": canonical_atomics,
        }
    search_deltas = (
        _fold_deltas(search_scores["scores"], baseline_med) if search_scores else {}
    )
    reeval_deltas = (
        _fold_deltas(reeval_scores["scores"], reeval_base_risk) if reeval_scores else {}
    )
    baseline_observation_rows = {
        str(key): dict(value)
        for key, value in (baseline_observations or {}).items()
    }
    search_candidate_evidence = _fold_evidence_rows(search_scores)
    for fold_id, evidence in search_candidate_evidence.items():
        baseline = baseline_observation_rows.get(fold_id) or {}
        if baseline:
            evidence["delta_risk"] = float(evidence["risk"]) - float(
                baseline["risk"]
            )
            evidence["delta_logit"] = float(evidence["abnormal_logit"]) - float(
                baseline["logit"]
            )
            evidence["baseline_observation_id"] = baseline.get("observation_id")
    reeval_baseline_evidence = _fold_evidence_rows(reeval_baseline_scores)
    reeval_candidate_evidence = _fold_evidence_rows(reeval_scores)
    for fold_id, evidence in reeval_candidate_evidence.items():
        baseline = reeval_baseline_evidence.get(fold_id) or {}
        if baseline:
            evidence["delta_risk"] = float(evidence["risk"]) - float(
                baseline["risk"]
            )
            evidence["delta_logit"] = float(evidence["abnormal_logit"]) - float(
                baseline["abnormal_logit"]
            )
            evidence["baseline_observation_id"] = (
                f"REEVAL_BASELINE::{fold_id}::{baseline.get('trace_invocation_id')}"
            )
    return {
        "candidate_id": candidate["candidate_id"],
        "kind": candidate.get("kind"),
        "event_ids": list(candidate.get("event_ids") or []),
        "transaction_sha": tx_sha,
        "validated_transaction_payload": validated_transaction_payload,
        "canonical_atomic_set_payload": canonical_atomic_set_payload,
        "canonical_atomic_set_sha": (
            tx.canonical_atomic_set_sha
            if isinstance(tx, ValidatedRawTransaction)
            else None
        ),
        "parent_transaction_shas": list(parent_transaction_shas or []),
        "gate0_status": candidate.get("gate0_status"),
        "gate4_status": candidate.get("gate4_status"),
        "outside_mg_unchanged": candidate.get("outside_mg_unchanged"),
        "allowed_change_set_sha": candidate.get("allowed_change_set_sha"),
        "actual_changed_paths_sha": candidate.get("actual_changed_paths_sha"),
        "required_changed_paths_sha": candidate.get("required_changed_paths_sha"),
        "canonical_atomic_evidence_manifest_sha": (
            (candidate.get("atomic_evidence_manifest") or {}).get(
                "canonical_atomic_evidence_manifest_sha"
            )
        ),
        "baseline_risk_by_fold": dict(baseline_med),
        "search_baseline_observation_by_fold": baseline_observation_rows,
        "search_candidate_evidence_by_fold": search_candidate_evidence,
        "search_candidate_risk_by_fold": _fold_risks(search_scores["scores"])
        if search_scores
        else {},
        "search_candidate_logit_by_fold": _fold_logits(search_scores["scores"])
        if search_scores
        else {},
        "search_delta_by_fold": search_deltas,
        "reeval_baseline_risk_by_fold": dict(reeval_base_risk),
        "reeval_baseline_evidence_by_fold": reeval_baseline_evidence,
        "reeval_candidate_evidence_by_fold": reeval_candidate_evidence,
        "reeval_candidate_risk_by_fold": _fold_risks(reeval_scores["scores"])
        if reeval_scores
        else {},
        "reeval_delta_by_fold": reeval_deltas,
        "incremental_by_fold": dict(incremental or {}),
        "residual_by_fold": dict(residual or {}),
        "locked_threshold": float(locked_thr),
        "expected_effect_direction": direction,
        "disposition": disposition,
        "trace_references": list(trace_refs or []),
        "failure_reason": failure_reason,
        "stage_a_output_sha_search": (search_scores or {}).get("stage_a_output_sha"),
        "diagnosis_batch_sha_search": (search_scores or {}).get("diagnosis_batch_sha"),
        "semantic_critic_input_sha_search": (search_scores or {}).get(
            "semantic_critic_input_sha"
        ),
        "stage_a_output_sha_reeval": (reeval_scores or {}).get("stage_a_output_sha"),
        "semantic_critic_input_sha_reeval": (reeval_scores or {}).get(
            "semantic_critic_input_sha"
        ),
    }


def run_development3_orchestrator(
    *,
    config: Mapping[str, Any],
    authorization_artifact: Mapping[str, Any],
    trust_root_path: Path,
    fold_routing_path: Path,
    hooks: CaseRuntimeHooks,
    out_root: Path,
    signing_private_key_hex: Optional[str] = None,
    quarantine: bool = True,
    observed_stable_lock: Optional[Mapping[str, Any]] = None,
    repo_root: Path = Path("/home/dasom/life2vec"),
) -> OrchestratorResult:
    """
    Full Development3 phase machine.
    Quarantine first; promote only after POST attestation + public-key verify + atomic rename.
    """
    gate_results: Dict[str, str] = {f"G{i}": "NOT_RUN" for i in range(1, 13)}
    run_id = str(authorization_artifact.get("run_id") or "")
    if not run_id:
        raise CoreContractError("signed PRE authorization must bind run_id")
    runtime_lock = apply_exact_byte_deterministic_runtime(
        before_cuda_init=not __import__("torch").cuda.is_initialized()
    )
    tr_sha = verify_trust_root_env(trust_root_path)
    trust_root = load_trust_root(trust_root_path)

    if observed_stable_lock is not None:
        raise CoreContractError(
            "caller-supplied observed stable lock forbidden; filesystem recomputation required"
        )
    pre_stable = recompute_stable_lock_manifest(
        authorization_artifact.get("stable_lock_manifest") or {},
        root=repo_root,
    )
    consume_pre_execution_authorization(
        authorization_artifact,
        trust_root=trust_root,
        trust_root_sha256=tr_sha,
        observed_stable_lock=pre_stable,
        expected_run_id=run_id,
    )

    quarantine_dir = out_root / "quarantine" / run_id
    final_dir = out_root / "development3_evidence" / run_id
    sidecar_dir = out_root / "sidecars" / run_id
    if final_dir.exists() or quarantine_dir.exists() or sidecar_dir.exists():
        raise CoreContractError(f"run ID collision; refuse overwrite: {run_id}")
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=False)
    runtime_obs = compute_runtime_observation(
        run_id=run_id, temporary_path=str(quarantine_dir)
    )
    _atomic_write_json(quarantine_dir / "runtime_observation.json", runtime_obs)
    _atomic_write_json(quarantine_dir / "pre_stable_lock.json", pre_stable)

    routing = load_fold_routing_manifest(fold_routing_path)
    man_path = Path(str(config["development_manifest_path"]))
    if not man_path.is_absolute():
        man_path = repo_root / man_path
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    case_ids = list(manifest["ordered_case_ids"])
    if len(case_ids) != 3:
        raise CoreContractError("Development3 requires exactly 3 cases")

    ckpts = list(hooks.list_checkpoint_paths())
    fold_ids = [int(x) for x in hooks.fold_ids]
    global_trace = GlobalForwardTrace()
    fold_forward = make_production_fold_forward_fn(
        normal_class_id=0,
        gpu_fraction=float(config.get("gpu_memory_fraction") or 0.4),
        global_trace=global_trace,
    )
    counters = ForwardCounters()
    if hooks.bind_global_trace is not None:
        hooks.bind_global_trace(global_trace)
    case_results: List[Dict[str, Any]] = []
    scientific_counts = {
        "SUPPORTED": 0,
        "NOT_SUPPORTED": 0,
        "INCONCLUSIVE": 0,
        "NOT_EVALUABLE": 0,
        "NOT_EVALUATED": 0,
    }
    evaluable = 0
    execution_ready = True

    for case_id in case_ids:
        sm = PhaseStateMachine()
        case_entry = (routing.get("cases") or {}).get(case_id) or {}
        prov = case_fold_provenance(case_entry)
        sm.advance(DevelopmentPhase.ATTRIBUTION)
        sm.advance(DevelopmentPhase.CANDIDATE_UNIVERSE)
        candidates = list(hooks.build_candidates(case_id))
        # Reject any bare edits
        for c in candidates:
            if "edits" in c and "validated_transaction" not in c:
                reject_unchecked_token_dict(c.get("edits"))
            if not isinstance(c.get("validated_transaction"), ValidatedRawTransaction):
                raise CoreContractError(
                    f"candidate {c.get('candidate_id')} missing ValidatedRawTransaction"
                )
        sm.advance(DevelopmentPhase.SEARCH_BASELINE_IDENTITY)

        events = hooks.load_case_events(case_id)
        sidecar = hooks.load_sidecar_by_event_id(case_id)
        reencoder = hooks.build_reencoder()
        search_folds = list(prov["search_checkpoint_ids"])
        reeval_folds = list(prov["reevaluation_checkpoint_ids"])
        identity_tx = hooks.score_identity_transaction(case_id)
        reject_unchecked_token_dict(identity_tx)

        baseline_risks_by_fold: Dict[str, List[float]] = {str(f): [] for f in search_folds}
        baseline_observations_by_fold: Dict[str, List[Dict[str, Any]]] = {
            str(f): [] for f in search_folds
        }
        identity_risk_by_fold: Dict[str, float] = {}
        candidate_ledger: List[Dict[str, Any]] = []
        selection_manifest: Dict[str, Any] = {
            "selected_candidate_ids": [],
            "excluded": [],
            "selected": [],
        }
        frozen: Dict[str, Any] = {}
        thr: Dict[str, Any] = {}
        gate: Dict[str, Any] = {"allow_candidate_effect_forward": False}

        try:
            for _repeat in range(3):
                base = run_cold_forward_pipeline(
                    events=events,
                    validated_transaction=identity_tx,
                    reencoder=reencoder,
                    fold_forward_fn=fold_forward,
                    checkpoint_paths=ckpts,
                    fold_ids=fold_ids,
                    requested_fold_ids=search_folds,
                    sidecar_by_event_id=sidecar,
                    scope="search",
                    transaction_mode="FORCED_IDENTITY",
                    forward_kind=ForwardKind.BASELINE_FORWARD,
                    counters=counters,
                    global_trace=global_trace,
                    case_id=case_id,
                    candidate_id="BASELINE",
                    phase="SEARCH_BASELINE_IDENTITY",
                    invocation_nonce=str(_repeat),
                )
                for fid, row in base["scores"]["by_fold"].items():
                    baseline_risks_by_fold[str(fid)].append(float(row["risk"]))
                    baseline_observations_by_fold[str(fid)].append(
                        {
                            "observation_id": f"{case_id}:search:{fid}:{_repeat}",
                            "risk": float(row["risk"]),
                            "logit": _abnormal_logit(row),
                            "trace_ref": base.get("trace_invocation_id"),
                            "critic_input_sha": row.get("semantic_critic_input_sha")
                            or base.get("semantic_critic_input_sha"),
                            "checkpoint_sha256": row.get("checkpoint_sha256"),
                        }
                    )

            ident = run_cold_forward_pipeline(
                events=events,
                validated_transaction=identity_tx,
                reencoder=reencoder,
                fold_forward_fn=fold_forward,
                checkpoint_paths=ckpts,
                fold_ids=fold_ids,
                requested_fold_ids=search_folds,
                sidecar_by_event_id=sidecar,
                scope="search",
                transaction_mode="FORCED_IDENTITY",
                forward_kind=ForwardKind.FORCED_IDENTITY_FORWARD,
                counters=counters,
                global_trace=global_trace,
                case_id=case_id,
                candidate_id="IDENTITY",
                phase="SEARCH_BASELINE_IDENTITY",
            )
            for fid, row in ident["scores"]["by_fold"].items():
                identity_risk_by_fold[str(fid)] = float(row["risk"])

            selected_baseline_observation = {
                fid: select_observed_median(rows)
                for fid, rows in baseline_observations_by_fold.items()
            }
            baseline_med = {
                fid: float(row["risk"])
                for fid, row in selected_baseline_observation.items()
            }
            runtime_noise = max(
                (runtime_repeat_noise(vals) for vals in baseline_risks_by_fold.values()),
                default=0.0,
            )
            identity_err = max_search_identity_reconstruction_error(
                identity_by_fold=identity_risk_by_fold,
                baseline_by_fold=baseline_med,
            )
            sm.advance(DevelopmentPhase.THRESHOLD_LOCK)
            thr = lock_thresholds_from_search_baseline_noise(
                runtime_noise,
                max_search_identity_reconstruction_error=identity_err,
            )
            gate = evaluate_scope_gate(
                scope="search",
                identity_pass=identity_err <= 0.001,
                noise_ceiling_pass=runtime_noise <= 0.001,
            )
        except Exception as exc:
            execution_ready = False
            scientific_counts["NOT_EVALUABLE"] += 1
            case_results.append(
                {
                    "case_id": case_id,
                    "status": "NOT_EVALUABLE",
                    "error": str(exc),
                    "fold_provenance": prov,
                }
            )
            continue

        sm.advance(DevelopmentPhase.SEARCH_EFFECTS)
        # Search ALL candidates first (no pre-selection by budget)
        search_rows: List[Dict[str, Any]] = []
        locked_thr = float(thr["locked_min_effect_abs_delta"])
        if gate["allow_candidate_effect_forward"] and candidates:
            for cand in candidates:
                scored = run_cold_forward_pipeline(
                    events=events,
                    validated_transaction=cand["validated_transaction"],
                    reencoder=reencoder,
                    fold_forward_fn=fold_forward,
                    checkpoint_paths=ckpts,
                    fold_ids=fold_ids,
                    requested_fold_ids=search_folds,
                    sidecar_by_event_id=sidecar,
                    scope="search",
                    transaction_mode="CANDIDATE",
                    forward_kind=ForwardKind.CANDIDATE_EFFECT_FORWARD,
                    counters=counters,
                    global_trace=global_trace,
                    case_id=case_id,
                    candidate_id=str(cand["candidate_id"]),
                    phase="SEARCH_EFFECTS",
                )
                search_rows.append({"candidate": cand, "search_scores": scored})

        sm.advance(DevelopmentPhase.SELECTION)
        # Eligibility only on PAIR bundles; atomics are dependency parents
        pair_search = []
        for r in search_rows:
            if r["candidate"].get("kind") != "PAIR":
                continue
            deltas = _fold_deltas(r["search_scores"]["scores"], baseline_med)
            pair_search.append(
                {
                    "candidate_id": r["candidate"]["candidate_id"],
                    "event_ids": r["candidate"]["event_ids"],
                    "kind": "PAIR",
                    "fold_deltas": deltas,
                    "transaction_sha": r["candidate"].get("transaction_sha"),
                }
            )
        selection_manifest = select_after_search(
            pair_rows=pair_search,
            locked_threshold=locked_thr,
            max_pair_budget=10,
            max_control=1,
        )
        # Expand parents for selected pairs, then apply dependency budget
        selected_pairs = list(selection_manifest["selected"])
        by_cid_search = {r["candidate"]["candidate_id"]: r for r in search_rows}
        parent_shas_by_pair: Dict[str, List[str]] = {}
        expanded: List[Dict[str, Any]] = []
        for sel in selected_pairs:
            pair_candidate = by_cid_search[sel["candidate_id"]]["candidate"]
            parent_ids = list(pair_candidate.get("parent_candidate_ids") or [])
            declared_parent_shas = list(
                pair_candidate.get("parent_transaction_shas") or []
            )
            parent_candidates = [
                by_cid_search[parent_id]["candidate"]
                for parent_id in parent_ids
                if parent_id in by_cid_search
            ]
            if len(parent_candidates) != 2:
                sel["disposition"] = "EXCLUDED_PARENT_GATE_FAILURE"
                continue
            verified_parent_shas = verify_exact_parent_transactions(
                bundle=pair_candidate["validated_transaction"],
                parents=[
                    candidate["validated_transaction"]
                    for candidate in parent_candidates
                ],
                declared_parent_shas=declared_parent_shas,
            )
            parent_shas_by_pair[str(sel["candidate_id"])] = verified_parent_shas
            expanded.append(
                {
                    "candidate_id": sel["candidate_id"],
                    "event_ids": sel["event_ids"],
                    "kind": "PAIR",
                    "disposition": sel["disposition"],
                    "expected_effect_direction": sel.get("expected_effect_direction"),
                }
            )
            for eid in sel["event_ids"]:
                matches = [
                    r
                    for r in search_rows
                    if r["candidate"].get("kind") == "ATOMIC"
                    and r["candidate"]["event_ids"] == [eid]
                ]
                if not matches:
                    sel["disposition"] = "EXCLUDED_PARENT_GATE_FAILURE"
                    continue
                parent = matches[0]["candidate"]
                expanded.append(
                    {
                        "candidate_id": parent["candidate_id"],
                        "event_ids": parent["event_ids"],
                        "kind": "ATOMIC",
                        "disposition": "PARENT_OF_SELECTED",
                        "expected_effect_direction": sel.get("expected_effect_direction"),
                    }
                )
        # Drop pairs that lost parents
        selected_pairs = [
            s
            for s in selected_pairs
            if s["disposition"] in (SELECTED_MATERIAL, SELECTED_CONTROL_NO_MATERIAL)
        ]
        budgeted = apply_dependency_budget(
            [
                {
                    "candidate_id": s["candidate_id"],
                    "event_ids": s["event_ids"],
                    "kind": "PAIR",
                }
                for s in selected_pairs
            ]
        )
        kept_ids = {c["candidate_id"] for c in budgeted["selected_bundles"]}
        for s in list(selected_pairs):
            if s["candidate_id"] not in kept_ids:
                s["disposition"] = "EXCLUDED_DEPENDENCY_BUDGET"
                selection_manifest["excluded"].append(s)
        selected_pairs = [s for s in selected_pairs if s["candidate_id"] in kept_ids]
        selection_manifest["selected"] = selected_pairs
        selection_manifest["selected_candidate_ids"] = [s["candidate_id"] for s in selected_pairs]
        selection_manifest["dependency_budget"] = budgeted
        selection_sha = canonical_json_sha256(
            {
                k: selection_manifest[k]
                for k in (
                    "version",
                    "locked_threshold",
                    "selected",
                    "excluded",
                    "selected_candidate_ids",
                    "execution_scope",
                )
                if k in selection_manifest
            }
        )

        # Assign Search-derived directions onto candidates
        direction_by_cid = {
            s["candidate_id"]: s.get("expected_effect_direction") for s in selected_pairs
        }
        closure_candidates = []
        for s in selected_pairs:
            closure_candidates.append(s["candidate_id"])
            for eid in s["event_ids"]:
                for r in search_rows:
                    if (
                        r["candidate"].get("kind") == "ATOMIC"
                        and r["candidate"]["event_ids"] == [eid]
                    ):
                        closure_candidates.append(r["candidate"]["candidate_id"])
                        if s.get("expected_effect_direction"):
                            direction_by_cid[r["candidate"]["candidate_id"]] = s[
                                "expected_effect_direction"
                            ]

        closure = build_selection_blind_closure(
            case_id=case_id,
            candidate_identity_hashes=sorted(set(closure_candidates)),
            search_result_hash=canonical_json_sha256(
                [
                    {
                        "cid": r["candidate"]["candidate_id"],
                        "sha": r["search_scores"].get("semantic_critic_input_sha"),
                        "tx": r["candidate"].get("transaction_sha"),
                    }
                    for r in search_rows
                ]
            ),
            selection_manifest_sha256=selection_sha,
            candidate_budget_policy_sha256=canonical_json_sha256(attribution_policy_lock()),
            threshold_policy_sha256=thr["threshold_policy_sha256"],
            risk_definition_sha256=canonical_json_sha256({"risk": "SIGMOID_ABNORMAL_LOGIT"}),
            candidate_universe_hash=canonical_json_sha256(
                sorted(c["candidate_id"] for c in candidates)
            ),
            parent_graph_sha256=canonical_json_sha256(budgeted.get("closure") or {}),
            expected_effect_direction_by_candidate={
                k: v for k, v in sorted(direction_by_cid.items()) if v
            },
            required_parent_transaction_shas=sorted(
                {
                    parent_sha
                    for selected in selected_pairs
                    for parent_sha in parent_shas_by_pair.get(
                        str(selected["candidate_id"]), []
                    )
                }
            ),
        )
        frozen = sm.freeze_closure(closure)
        global_trace.mark_closure_frozen(closure_sha=str(frozen["evaluation_closure_hash"]))
        sm.assert_closure_immutable(frozen)

        sm.advance(DevelopmentPhase.REEVAL_BASELINE_IDENTITY)
        reeval_base = run_cold_forward_pipeline(
            events=events,
            validated_transaction=identity_tx,
            reencoder=reencoder,
            fold_forward_fn=fold_forward,
            checkpoint_paths=ckpts,
            fold_ids=fold_ids,
            requested_fold_ids=reeval_folds,
            sidecar_by_event_id=sidecar,
            scope="selection_blind_reevaluation",
            transaction_mode="FORCED_IDENTITY",
            forward_kind=ForwardKind.FORCED_IDENTITY_FORWARD,
            counters=counters,
            global_trace=global_trace,
            case_id=case_id,
            candidate_id="IDENTITY_REEVAL",
            phase="REEVAL_BASELINE_IDENTITY",
        )
        reeval_base_risk = {
            fid: float(row["risk"]) for fid, row in reeval_base["scores"]["by_fold"].items()
        }

        sm.advance(DevelopmentPhase.REEVAL_EFFECTS)
        # Fold2: frozen closure rows only
        closure_set = set(closure_candidates)
        reeval_by_cid: Dict[str, Mapping[str, Any]] = {}
        for r in search_rows:
            cid = r["candidate"]["candidate_id"]
            if cid not in closure_set:
                continue
            reeval_by_cid[cid] = run_cold_forward_pipeline(
                events=events,
                validated_transaction=r["candidate"]["validated_transaction"],
                reencoder=reencoder,
                fold_forward_fn=fold_forward,
                checkpoint_paths=ckpts,
                fold_ids=fold_ids,
                requested_fold_ids=reeval_folds,
                sidecar_by_event_id=sidecar,
                scope="selection_blind_reevaluation",
                transaction_mode="CANDIDATE",
                counters=counters,
                global_trace=global_trace,
                case_id=case_id,
                candidate_id=cid,
                phase="REEVAL_EFFECTS",
            )
        sm.advance(DevelopmentPhase.RESULTS)

        scientific_detail: Dict[str, Any] = {}
        status_name = "NOT_EVALUATED"
        sci_enum = ScientificStatus.NOT_EVALUATED

        # Build candidate ledger for all searched candidates
        ledger_errors: Dict[str, str] = {}
        for r in search_rows:
            cand = r["candidate"]
            cid = cand["candidate_id"]
            disp = None
            direction = None
            for s in selected_pairs:
                if s["candidate_id"] == cid:
                    disp = s["disposition"]
                    direction = s.get("expected_effect_direction")
            if cid in closure_set and disp is None and cand.get("kind") == "ATOMIC":
                # parent of selected
                for s in selected_pairs:
                    if set(cand["event_ids"]).issubset(set(s["event_ids"])):
                        disp = "PARENT_OF_SELECTED"
                        direction = s.get("expected_effect_direction")
            try:
                ledger_row = _ledger_row(
                    candidate=cand,
                    search_scores=r["search_scores"],
                    reeval_scores=reeval_by_cid.get(cid),
                    baseline_med=baseline_med,
                    baseline_observations=selected_baseline_observation,
                    reeval_base_risk=reeval_base_risk,
                    reeval_baseline_scores=reeval_base,
                    locked_thr=locked_thr,
                    direction=direction,
                    disposition=disp,
                    parent_transaction_shas=(
                        parent_shas_by_pair.get(cid, [])
                        if cand.get("kind") == "PAIR"
                        else []
                    ),
                    trace_refs=[r["search_scores"].get("trace_invocation_id")]
                    + (
                        [reeval_by_cid[cid].get("trace_invocation_id")]
                        if cid in reeval_by_cid
                        else []
                    ),
                )
            except CoreContractError as exc:
                ledger_errors[str(cid)] = str(exc)
                ledger_row = {
                    "candidate_id": cid,
                    "kind": cand.get("kind"),
                    "transaction_sha": cand.get("transaction_sha"),
                    "parent_transaction_shas": list(
                        parent_shas_by_pair.get(cid, [])
                    ),
                    "disposition": disp,
                    "failure_reason": str(exc),
                    "scientific_status": "NOT_EVALUABLE",
                }
            candidate_ledger.append(ledger_row)

        if not selected_pairs:
            scientific_counts["NOT_EVALUATED"] += 1
            status_name = "NOT_EVALUATED"
            scientific_detail = {"reason": "NO_SELECTED_BUNDLE"}
        else:
            # Evaluate first selected material, else first control
            materials = [s for s in selected_pairs if s["disposition"] == SELECTED_MATERIAL]
            controls = [
                s for s in selected_pairs if s["disposition"] == SELECTED_CONTROL_NO_MATERIAL
            ]
            target = (materials or controls)[0]
            pair_search_row = by_cid_search[target["candidate_id"]]
            pair_reeval = reeval_by_cid.get(target["candidate_id"])
            if target["candidate_id"] in ledger_errors:
                status_name = "NOT_EVALUABLE"
                sci_enum = ScientificStatus.NOT_EVALUABLE
                scientific_counts["NOT_EVALUABLE"] += 1
                scientific_detail = {
                    "reason": "SELECTED_BUNDLE_LEDGER_INCOMPLETE",
                    "ledger_error": ledger_errors[target["candidate_id"]],
                    "failed_gates": ["G4", "G8", "G9"],
                }
            elif pair_reeval is None:
                status_name = "NOT_EVALUABLE"
                sci_enum = ScientificStatus.NOT_EVALUABLE
                scientific_counts["NOT_EVALUABLE"] += 1
                scientific_detail = {"reason": "REEVAL_MISSING"}
            elif target["disposition"] == SELECTED_CONTROL_NO_MATERIAL:
                reeval_deltas = _fold_deltas(pair_reeval["scores"], reeval_base_risk)
                control_diagnostic = classify_control_reeval(
                    reeval_deltas=reeval_deltas, locked_threshold=locked_thr
                )
                max_abs_delta = max(
                    (abs(float(value)) for value in reeval_deltas.values()),
                    default=None,
                )
                status_name = "NOT_EVALUATED"
                sci_enum = ScientificStatus.NOT_EVALUATED
                scientific_counts["NOT_EVALUATED"] += 1
                scientific_detail = {
                    "development_scientific_result": "NOT_EVALUATED",
                    "reason": "CONTROL_ONLY_NO_MATERIAL_EFFECT_EVALUATION",
                    "control_result": control_result_from_delta(
                        delta_risk_value=max_abs_delta,
                        locked_threshold=locked_thr,
                        evaluable=bool(reeval_deltas),
                    ),
                    "control_diagnostic": control_diagnostic,
                }
                scientific_detail["bundle_candidate_id"] = target["candidate_id"]
                scientific_detail["search_deltas"] = _fold_deltas(
                    pair_search_row["search_scores"]["scores"], baseline_med
                )
                scientific_detail["reeval_deltas"] = reeval_deltas
                scientific_detail["control_branch"] = True
            else:
                parent_ids = []
                for eid in target["event_ids"]:
                    matches = [
                        r
                        for r in search_rows
                        if r["candidate"].get("kind") == "ATOMIC"
                        and r["candidate"]["event_ids"] == [eid]
                    ]
                    if matches:
                        parent_ids.append(matches[0]["candidate"]["candidate_id"])
                if len(parent_ids) < 2:
                    status_name = "NOT_EVALUABLE"
                    sci_enum = ScientificStatus.NOT_EVALUABLE
                    scientific_counts["NOT_EVALUABLE"] += 1
                    scientific_detail = {"reason": "PARENT_ATOMICS_MISSING"}
                else:
                    search_bundle = _fold_deltas(
                        pair_search_row["search_scores"]["scores"], baseline_med
                    )
                    reeval_bundle = _fold_deltas(pair_reeval["scores"], reeval_base_risk)
                    search_parents = {
                        fid: [
                            _fold_deltas(
                                by_cid_search[pid]["search_scores"]["scores"], baseline_med
                            ).get(fid, 0.0)
                            for pid in parent_ids
                        ]
                        for fid in search_bundle
                    }
                    reeval_parents = {
                        fid: [
                            _fold_deltas(
                                reeval_by_cid[pid]["scores"], reeval_base_risk
                            ).get(fid, 0.0)
                            for pid in parent_ids
                            if pid in reeval_by_cid
                        ]
                        for fid in reeval_bundle
                    }
                    parents_complete = all(
                        len(search_parents.get(fold_id) or []) == 2
                        for fold_id in ("0", "1")
                    ) and len(reeval_parents.get("2") or []) == 2
                    parents_complete = parents_complete and all(
                        fold_id
                        in _fold_deltas(
                            by_cid_search[parent_id]["search_scores"]["scores"],
                            baseline_med,
                        )
                        for parent_id in parent_ids
                        for fold_id in ("0", "1")
                    ) and all(
                        parent_id in reeval_by_cid
                        and "2"
                        in _fold_deltas(
                            reeval_by_cid[parent_id]["scores"],
                            reeval_base_risk,
                        )
                        for parent_id in parent_ids
                    )
                    if not parents_complete:
                        status_name = "NOT_EVALUABLE"
                        sci_enum = ScientificStatus.NOT_EVALUABLE
                        scientific_counts["NOT_EVALUABLE"] += 1
                        scientific_detail = {
                            "reason": "PARENT_FOLD_LEDGER_INCOMPLETE"
                        }
                    else:
                        direction = target.get("expected_effect_direction")
                        if not direction:
                            raise CoreContractError(
                                "material bundle missing Search-derived direction"
                            )
                        scientific_detail = evaluate_bundle_effects(
                            search_fold_deltas=search_bundle,
                            reeval_fold_deltas=reeval_bundle,
                            search_parent_deltas_by_fold=search_parents,
                            reeval_parent_deltas_by_fold=reeval_parents,
                            expected_effect_direction=direction,
                            locked_threshold=locked_thr,
                        )
                        status_name = str(
                            scientific_detail["development_scientific_result"]
                        )
                        sci_enum = ScientificStatus[status_name]
                        scientific_counts[status_name] = (
                            scientific_counts.get(status_name, 0) + 1
                        )
                        scientific_detail["bundle_candidate_id"] = target[
                            "candidate_id"
                        ]
                        scientific_detail["search_deltas"] = search_bundle
                        scientific_detail["reeval_deltas"] = reeval_bundle
                        scientific_detail["expected_effect_direction"] = direction

        parent_complete = True
        for s in selected_pairs:
            for eid in s["event_ids"]:
                if not any(
                    r["candidate"].get("kind") == "ATOMIC"
                    and r["candidate"]["event_ids"] == [eid]
                    and r["candidate"]["candidate_id"] in reeval_by_cid
                    for r in search_rows
                ):
                    parent_complete = False
        coverage_ok = case_coverage_eligible(
            baseline_identity_valid=bool(gate["allow_candidate_effect_forward"]),
            selected_bundles=selected_pairs,
            parent_evidence_complete=parent_complete,
            search_folds_complete=bool(search_rows),
            reeval_folds_complete=bool(reeval_by_cid) or not selected_pairs,
            candidate_ledger_complete=bool(candidate_ledger) and all(
                row.get("transaction_sha") for row in candidate_ledger
            ),
            trace_and_lock_valid=True,
        )
        if coverage_ok:
            evaluable += 1

        case_dir = quarantine_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        authority = AuthorityContext(
            evaluation_stage=EvaluationStage.DEVELOPMENT_PRODUCTION,
            execution_authorized=True,
            development_production_execution_authorized=True,
            evidence_output_authorized=True,
            recommendation_authorized=False,
            primary32_execution_authorized=False,
            problem20_execution_authorized=False,
        )
        evidence = ExecutionEvidenceFlags(
            production_input_verified=True,
            pipeline_integrity_verified=True,
            attribution_forward_executed=bool(
                global_trace.summarize().get("attribution_forward_count")
            ),
            baseline_forward_complete=True,
            identity_forward_complete=True,
            full_sequence_cold_rebuild=True,
            identity_pass=bool(gate["allow_candidate_effect_forward"]),
            noise_ceiling_pass=bool(gate["allow_candidate_effect_forward"]),
            candidate_effect_forward_complete=bool(search_rows),
            complete_family_executed=bool(selected_pairs) and status_name != "NOT_EVALUABLE",
            all_required_folds_complete=bool(selected_pairs),
        )
        sets = build_candidate_set_manifest(
            constructed_candidate_ids=[c["candidate_id"] for c in candidates],
            evaluable_candidate_ids=[r["candidate"]["candidate_id"] for r in search_rows],
            selected_candidate_ids=[s["candidate_id"] for s in selected_pairs],
            budget_excluded_candidate_ids=[
                e["candidate_id"] for e in selection_manifest.get("excluded") or []
            ],
        )
        forward_counts = global_trace.to_forward_counts()
        envelope = build_case_edit_proposal_envelope(
            case_id=case_id,
            authority=authority,
            evidence=evidence,
            evidence_kind=EvidenceKind.ACTUAL_MODEL_FORWARD,
            construction_status=ConstructionStatus.COMPLETE
            if candidates
            else ConstructionStatus.NOT_CONSTRUCTIBLE,
            scientific_status=sci_enum,
            ordering_status=OrderingStatus.NOT_ORDERABLE,
            candidate_sets=sets,
            candidate_results=candidate_ledger,
            scientific_status_reason=str(scientific_detail.get("reason") or status_name),
            forward_counts=forward_counts,
            extra={
                "interpretation_label": "Development3 selection-blind contract verification evidence",
                "execution_scope": "TWO_EVENT_ONLY",
                "three_event_execution_status": "OUT_OF_SCOPE",
                "scientifically_independent_holdout": False,
                "training_seen": True,
                "fold_provenance": prov,
                "closure": frozen,
                "threshold": thr,
                "runtime_lock_sha256": runtime_lock.sha256(),
                "scientific_detail": scientific_detail,
                "selection_manifest": selection_manifest,
                "coverage_case_eligible": coverage_ok,
                # Package stores computed only — never attested
                "computed_evaluation_coverage_contribution": coverage_ok,
                "final_attestation_status": "PENDING",
            },
        )
        write_case_proposal_json(envelope, case_dir / "case_edit_proposal.json")
        case_results.append(
            {
                "case_id": case_id,
                "status": status_name,
                "execution_complete": bool(coverage_ok),
                "selection_disposition": (
                    target.get("disposition") if selected_pairs else None
                ),
                "scientifically_evaluable": bool(
                    selected_pairs
                    and target.get("disposition") == SELECTED_MATERIAL
                    and status_name
                    in ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE")
                ),
                "control_result": scientific_detail.get("control_result"),
                "fold_provenance": prov,
                "selected": selection_manifest,
                "scientific_detail": scientific_detail,
                "candidate_ledger_count": len(candidate_ledger),
            }
        )

    axes = development_readiness_axes(
        execution_ready=execution_ready,
        evaluable_cases=evaluable,
        locked_cohort_size=3,
        scientific_counts=scientific_counts,
    )
    coverage_axes = derive_coverage_axes(case_results)
    # Rename coverage field to computed inside package
    computed_coverage = coverage_axes["material_effect_evaluation_coverage"]["status"]
    axes_package = {
        **axes,
        **coverage_axes,
        "computed_evaluation_coverage": computed_coverage,
        "coverage_contract_pass": True,
        "development_execution_readiness": "NOT_READY",
        "cohort_completion_claim_allowed": False,
        "final_attestation_status": "PENDING",
        # Do NOT include attested_evaluation_coverage inside package
        "execution_scope": "TWO_EVENT_ONLY",
        "three_event_execution_status": "OUT_OF_SCOPE",
        "interpretation_label": "Development3 selection-blind contract verification evidence",
        "scientifically_independent_holdout": False,
        "training_seen": True,
    }
    # Strip ambiguous attested-ready claim until external verify
    axes_package["development_evaluation_coverage"] = computed_coverage
    axes_package["development_execution_readiness"] = "NOT_READY"

    trace_summary = global_trace.summarize()
    post_stable = recompute_stable_lock_manifest(
        authorization_artifact.get("stable_lock_manifest") or {},
        root=repo_root,
    )
    try:
        pre_post_identical = assert_stable_locks_identical(pre_stable, post_stable)
    except CoreContractError:
        pre_post_identical = False
        execution_ready = False
        axes_package["development_execution_readiness"] = "NOT_READY"

    summary = {
        **axes_package,
        "case_results": case_results,
        "forward_counts": global_trace.to_forward_counts(),
        "trace_summary": {
            k: v for k, v in trace_summary.items() if k != "events"
        },
        "runtime_lock": runtime_lock.to_dict(),
        "pre_post_lock_identical": bool(pre_post_identical),
        "pre_stable_lock_sha256": pre_stable.get("stable_lock_sha256"),
        "post_stable_lock_sha256": post_stable.get("stable_lock_sha256"),
    }
    _atomic_write_json(quarantine_dir / "runner_summary.json", summary)
    _atomic_write_json(quarantine_dir / "post_stable_lock.json", post_stable)
    _atomic_write_json(quarantine_dir / "trace_events.json", {"events": trace_summary["events"]})

    report_lines = [
        "Execution completed." if execution_ready else "Execution completed with readiness gaps.",
        f"Computed evaluation coverage: {computed_coverage}.",
        f"Scientific result counts: {dict(scientific_counts)}.",
        "Attested coverage is verifier-derived only and is not stored in this package.",
        "execution_scope=TWO_EVENT_ONLY; three_event_execution_status=OUT_OF_SCOPE.",
    ]
    (quarantine_dir / "REPORT.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    final_rerun_observation: Dict[str, Any] = {}
    final_rerun_path = config.get("final_rerun_observation_manifest_path")
    if final_rerun_path:
        observation_source = Path(str(final_rerun_path))
        if not observation_source.is_absolute():
            observation_source = repo_root / observation_source
        final_rerun_observation = json.loads(
            observation_source.read_text(encoding="utf-8")
        )
        test_log_source = Path(str(config.get("final_rerun_test_log_path") or ""))
        node_manifest_source = Path(
            str(config.get("final_rerun_node_id_manifest_path") or "")
        )
        if not test_log_source.is_absolute():
            test_log_source = repo_root / test_log_source
        if not node_manifest_source.is_absolute():
            node_manifest_source = repo_root / node_manifest_source
        test_log_snapshot = (
            quarantine_dir / "final_rerun" / "final_rerun_test.log"
        )
        node_manifest_snapshot = (
            quarantine_dir / "final_rerun" / "final_rerun_node_id_manifest.json"
        )
        _atomic_copy_file(test_log_source, test_log_snapshot)
        _atomic_copy_file(node_manifest_source, node_manifest_snapshot)
        final_rerun_observation.update(
            {
                "final_rerun_test_log_relpath": str(
                    test_log_snapshot.relative_to(quarantine_dir)
                ),
                "final_rerun_node_id_manifest_relpath": str(
                    node_manifest_snapshot.relative_to(quarantine_dir)
                ),
                "final_rerun_test_log_sha256": sha256_file(test_log_snapshot),
                "final_rerun_node_id_manifest_sha256": sha256_file(
                    node_manifest_snapshot
                ),
            }
        )
        final_rerun_observation["final_rerun_observation_manifest_sha256"] = (
            canonical_json_sha256(
                {
                    key: value
                    for key, value in final_rerun_observation.items()
                    if key != "final_rerun_observation_manifest_sha256"
                }
            )
        )
        _atomic_write_json(
            sidecar_dir / "final_rerun_observation_manifest.json",
            final_rerun_observation,
        )

    artifact_paths = {
        "runner_summary.json": quarantine_dir / "runner_summary.json",
        "REPORT.txt": quarantine_dir / "REPORT.txt",
        "pre_stable_lock.json": quarantine_dir / "pre_stable_lock.json",
        "post_stable_lock.json": quarantine_dir / "post_stable_lock.json",
        "runtime_observation.json": quarantine_dir / "runtime_observation.json",
        "trace_events.json": quarantine_dir / "trace_events.json",
    }
    if final_rerun_observation:
        artifact_paths.update(
            {
                "final_rerun/final_rerun_test.log": (
                    quarantine_dir / "final_rerun" / "final_rerun_test.log"
                ),
                "final_rerun/final_rerun_node_id_manifest.json": (
                    quarantine_dir
                    / "final_rerun"
                    / "final_rerun_node_id_manifest.json"
                ),
            }
        )
    for cr in case_results:
        p = quarantine_dir / cr["case_id"] / "case_edit_proposal.json"
        if p.is_file():
            artifact_paths[f"{cr['case_id']}/case_edit_proposal.json"] = p

    evidence_root = compute_evidence_package_root(
        artifact_paths,
        root=quarantine_dir,
    )
    artifact_relpaths = {
        key: str(Path(value).relative_to(quarantine_dir))
        for key, value in artifact_paths.items()
    }
    _atomic_write_json(
        sidecar_dir / "EVIDENCE_ROOT.json",
        {
            "evidence_root_sha": evidence_root,
            "computed_evaluation_coverage": computed_coverage,
            "final_attestation_status": "PENDING",
            "package_path": str(quarantine_dir),
            "artifact_relpaths": artifact_relpaths,
            "sidecar_only": True,
        },
    )
    package_state = QUARANTINE_ONLY
    attested_coverage = "UNVERIFIED"
    package_dir: Optional[Path] = quarantine_dir
    receipt_valid = False
    readback_ok = False
    pre_promotion_verdict: Optional[Dict[str, Any]] = None
    final_verdict: Optional[Dict[str, Any]] = None
    receipt: Optional[Dict[str, Any]] = None
    verifier_code_sha = compute_verifier_code_sha256()

    if signing_private_key_hex and pre_post_identical and execution_ready:
        att = issue_post_execution_attestation(
            trust_root=trust_root,
            trust_root_sha256=tr_sha,
            issuer_id=str(authorization_artifact["issuer_id"]),
            key_id=str(authorization_artifact["key_id"]),
            private_key_hex=signing_private_key_hex,
            evidence_root_sha=evidence_root,
            pre_execution_payload_sha256=str(authorization_artifact["payload_sha256"]),
            run_id=run_id,
            pre_stable_lock_sha256=str(pre_stable["stable_lock_sha256"]),
            post_stable_lock_sha256=str(post_stable["stable_lock_sha256"]),
            intended_final_destination=str(final_dir),
            final_rerun_observation_manifest_sha256=final_rerun_observation.get(
                "final_rerun_observation_manifest_sha256"
            ),
            acceptance_summary={
                "computed_evaluation_coverage": computed_coverage,
                "development_execution_readiness": axes_package[
                    "development_execution_readiness"
                ],
                "development_scientific_result": dict(scientific_counts),
                "execution_scope": "TWO_EVENT_ONLY",
            },
            pre_post_lock_identical=True,
        )
        _atomic_write_json(sidecar_dir / "POST_EXECUTION_EVIDENCE_ATTESTATION.json", att)
        try:
            verify_post_attestation_public_key_only(
                att,
                trust_root=trust_root,
                trust_root_sha256=tr_sha,
                expected_evidence_root=evidence_root,
                expected_pre_payload_sha=str(authorization_artifact["payload_sha256"]),
                expected_run_id=run_id,
                expected_stable_lock_sha256=str(pre_stable["stable_lock_sha256"]),
                expected_final_destination=str(final_dir),
                expected_final_rerun_observation_manifest_sha256=(
                    final_rerun_observation.get(
                        "final_rerun_observation_manifest_sha256"
                    )
                ),
            )
            pre_promotion_verdict = verify_cf1s_package(
                phase=PRE_PROMOTION,
                package_dir=quarantine_dir,
                expected_final_dir=final_dir,
                sidecar_dir=sidecar_dir,
                run_id=run_id,
                authorization_artifact=authorization_artifact,
                post_attestation=att,
                trust_root=trust_root,
                trust_root_sha256=tr_sha,
                verifier_code_sha256=verifier_code_sha,
            )
            _atomic_write_json(
                sidecar_dir / "PRE_PROMOTION_VERDICT.json",
                pre_promotion_verdict,
            )
        except CoreContractError as exc:
            report_lines.append(f"PRE_PROMOTION verifier failed: {exc}")

        if (
            pre_promotion_verdict
            and pre_promotion_verdict.get("promotion_eligible")
            and quarantine
        ):
            try:
                promo = atomic_promote_directory(
                    quarantine_dir=quarantine_dir, final_dir=final_dir
                )
                package_dir = final_dir
                package_state = PROMOTED
            except CoreContractError as exc:
                report_lines.append(f"Promotion failed: {exc}")
                promo = None

            if package_state == PROMOTED:
                readback = readback_final_package(
                    final_dir=final_dir,
                    expected_evidence_root=evidence_root,
                    artifact_relpaths=artifact_relpaths,
                )
                readback_ok = bool(readback.get("ok"))
                post_att_sha = payload_sha256(
                    att, exclude_keys=("signature_hex", "payload_sha256")
                )
                try:
                    receipt_path, receipt = write_promotion_receipt(
                        receipts_dir=out_root / "promotion_receipts",
                        run_id=run_id,
                        final_path=final_dir,
                        quarantine_path=str(quarantine_dir),
                        evidence_root_sha=evidence_root,
                        post_attestation_sha=post_att_sha,
                        promotion_meta=promo or {},
                        final_readback=readback,
                    )
                    validate_promotion_receipt(
                        receipt,
                        expected_run_id=run_id,
                        expected_final_path=final_dir,
                        expected_evidence_root=evidence_root,
                        expected_post_attestation_sha=post_att_sha,
                        artifact_relpaths=artifact_relpaths,
                    )
                    receipt_valid = True
                    report_lines.append(f"Promotion receipt: {receipt_path}")
                except Exception as exc:
                    # Do NOT move package back or delete
                    package_state = PROMOTED_BUT_UNATTESTED_BY_RECEIPT
                    receipt_valid = False
                    report_lines.append(
                        f"PROMOTED_BUT_UNATTESTED_BY_RECEIPT: receipt failed: {exc}"
                    )
                try:
                    final_verdict = verify_cf1s_package(
                        phase=FINAL,
                        package_dir=final_dir,
                        expected_final_dir=final_dir,
                        sidecar_dir=sidecar_dir,
                        run_id=run_id,
                        authorization_artifact=authorization_artifact,
                        post_attestation=att,
                        trust_root=trust_root,
                        trust_root_sha256=tr_sha,
                        verifier_code_sha256=verifier_code_sha,
                        receipt=receipt,
                        pre_promotion_verdict=pre_promotion_verdict,
                    )
                    _atomic_write_json(
                        sidecar_dir / "FINAL_VERDICT.json",
                        final_verdict,
                    )
                except CoreContractError as exc:
                    report_lines.append(f"FINAL verifier failed: {exc}")
                all_final_pass = bool(
                    final_verdict
                    and all(
                        value == "PASS"
                        for value in (final_verdict.get("gate_results") or {}).values()
                    )
                )
                attested_coverage = (
                    computed_coverage if all_final_pass else "UNVERIFIED"
                )
                if not all_final_pass and package_state == PROMOTED:
                    package_state = PROMOTED_BUT_UNVERIFIED
    else:
        report_lines.append(
            "Quarantine-only: missing signing key, lock drift, or execution not ready."
        )
    gate_results = dict(
        (final_verdict or pre_promotion_verdict or {}).get("gate_results")
        or {f"G{i}": "NOT_RUN" for i in range(1, 13)}
    )

    if attested_coverage == "UNVERIFIED":
        axes_package["development_execution_readiness"] = "NOT_READY"
        # Keep computed coverage in summary but claim axes fail-closed for external use
    elif final_verdict and all(
        value == "PASS"
        for value in (final_verdict.get("gate_results") or {}).values()
    ):
        axes_package["development_execution_readiness"] = "READY"

    return OrchestratorResult(
        execution_readiness=axes_package["development_execution_readiness"],
        evaluation_coverage=computed_coverage,
        attested_evaluation_coverage=attested_coverage,
        scientific_counts=scientific_counts,
        case_results=case_results,
        package_dir=package_dir if quarantine else quarantine_dir,
        package_state=package_state,
        report_lines=report_lines,
        gate_results=gate_results,
    )
