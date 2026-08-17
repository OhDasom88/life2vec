"""Read-only two-phase CF1S package verifier and external completion projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .core_authorization import (
    compute_evidence_package_root,
    consume_pre_execution_authorization,
    payload_sha256,
)
from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError, sha256_file
from .core_development import evaluate_bundle_effects
from .core_locks import (
    REPO_ROOT,
    validate_stable_lock_manifest,
    verify_post_attestation_public_key_only,
)
from .core_preflight import validate_measured_preflight_contract
from .core_promotion import validate_promotion_receipt
from .core_scientific import (
    check_risk_logit_contract,
    control_result_from_delta,
    derive_coverage_axes,
)
from .core_selection import select_after_search
from .core_trace import validate_persisted_trace
from .disposition_profiles import (
    NOT_CONSTRUCTIBLE,
    case_trace_events,
    classify_case_disposition,
    evaluate_case_trace_profile,
)


PRE_PROMOTION = "PRE_PROMOTION"
FINAL = "FINAL"
_PASS = "PASS"
_FAIL = "FAIL"
_VERIFIER_MODULES = (
    "core_verifier.py",
    "core_authorization.py",
    "core_locks.py",
    "core_trace.py",
    "core_scientific.py",
    "core_canonical.py",
    "core_contract.py",
    "core_promotion.py",
    "disposition_profiles.py",
)


def compute_verifier_code_sha256(module_dir: Optional[Path] = None) -> str:
    root = Path(module_dir or Path(__file__).parent)
    files = {}
    for name in _VERIFIER_MODULES:
        path = root / name
        if not path.is_file():
            raise CoreContractError(f"verifier module missing: {path}")
        files[name] = sha256_file(path)
    return canonical_json_sha256(
        {"version": "CF1S_VERIFIER_CODE_TREE_V1", "files": files}
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise CoreContractError(f"required verifier artifact missing: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_proposals(
    package_dir: Path, expected_case_count: int = 3
) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(Path(package_dir).glob("*/case_edit_proposal.json")):
        rows.append(_load_json(path))
    if len(rows) != int(expected_case_count):
        raise CoreContractError(
            f"cohort verifier requires exactly {expected_case_count} case proposals"
        )
    return rows


def _gate_transaction(proposals: Sequence[Mapping[str, Any]]) -> None:
    for proposal in proposals:
        for row in proposal.get("candidate_results") or []:
            if not row.get("transaction_sha"):
                raise CoreContractError("candidate transaction_sha missing")
            tx_payload = row.get("validated_transaction_payload")
            atomic_payload = row.get("canonical_atomic_set_payload")
            if not tx_payload or not atomic_payload:
                raise CoreContractError("typed canonical transaction payload missing")
            if canonical_json_sha256(tx_payload) != row.get("transaction_sha"):
                raise CoreContractError("canonical transaction SHA mismatch")
            if canonical_json_sha256(atomic_payload) != row.get(
                "canonical_atomic_set_sha"
            ):
                raise CoreContractError("canonical atomic-set SHA mismatch")
            if row.get("gate0_status") not in (None, "PASS"):
                raise CoreContractError("candidate Gate0 failed")
            if row.get("gate4_status") not in (None, "PASS"):
                raise CoreContractError("candidate Gate4 failed")


def _gate_runtime(
    summary: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    pre_stable_lock: Mapping[str, Any],
) -> None:
    if summary.get("execution_scope") != "TWO_EVENT_ONLY":
        raise CoreContractError("execution_scope is not TWO_EVENT_ONLY")
    if summary.get("three_event_execution_status") != "OUT_OF_SCOPE":
        raise CoreContractError("three-event execution must be OUT_OF_SCOPE")
    runtime = summary.get("runtime_lock") or {}
    if not runtime:
        raise CoreContractError("deterministic runtime lock missing")
    source = (pre_stable_lock.get("artifact_sources") or {}).get(
        "qualification_preflight_sha256"
    )
    if not source:
        raise CoreContractError("qualification preflight source missing from stable lock")
    preflight_path = Path(str(source))
    if not preflight_path.is_absolute():
        preflight_path = REPO_ROOT / preflight_path
    preflight = _load_json(preflight_path)
    validate_measured_preflight_contract(preflight.get("checks") or [])
    for proposal in proposals:
        evidence = proposal.get("execution_evidence") or {}
        for field in (
            "production_input_verified",
            "pipeline_integrity_verified",
            "full_sequence_cold_rebuild",
            "baseline_forward_complete",
            "identity_forward_complete",
        ):
            if evidence.get(field) is not True:
                raise CoreContractError(f"runtime evidence flag false: {field}")
        for row in proposal.get("candidate_results") or []:
            if row.get("disposition") in (
                "SELECTED_MATERIAL",
                "SELECTED_CONTROL_NO_MATERIAL",
                "PARENT_OF_SELECTED",
            ):
                if not row.get("stage_a_output_sha_search"):
                    raise CoreContractError("selected/parent Stage-A search SHA missing")
                if not row.get("stage_a_output_sha_reeval"):
                    raise CoreContractError("selected/parent Stage-A Fold2 SHA missing")


def _gate_trace(package_dir: Path, summary: Mapping[str, Any]) -> None:
    persisted = _load_json(Path(package_dir) / "trace_events.json")
    events = list(persisted.get("events") or [])
    recalculated = validate_persisted_trace(events)
    stored = summary.get("trace_summary") or {}
    for key, value in recalculated.items():
        if key == "events":
            continue
        if stored.get(key) != value:
            raise CoreContractError(f"trace summary mismatch: {key}")
    for key in (
        "fold2_attribution_load_count",
        "fold2_attribution_forward_count",
        "fold2_critic_load_before_closure_count",
        "fold2_candidate_forward_before_closure_count",
        "nonrequested_fold_forward_count",
    ):
        if int(recalculated.get(key) or 0) != 0:
            raise CoreContractError(f"trace isolation violation: {key}")
    for event in events:
        if event.get("state") != "COMPLETED":
            continue
        fold_id = event.get("fold_id")
        scope = str(event.get("scope"))
        kind = str(event.get("kind"))
        if kind in ("ATTRIBUTION_MODEL_LOAD", "ATTRIBUTION_FORWARD"):
            if scope != "search" or fold_id not in (0, 1):
                raise CoreContractError("attribution trace escaped search folds")
        if kind in ("CRITIC_LOAD", "CHECKPOINT_FORWARD"):
            if scope == "search" and fold_id not in (0, 1):
                raise CoreContractError("search critic trace used nonrequested fold")
            if scope in ("selection_blind_reevaluation", "holdout", "fold2"):
                if fold_id != 2 or not event.get("closure_frozen"):
                    raise CoreContractError(
                        "Fold2 critic/forward occurred outside frozen reevaluation"
                    )


def _gate_trace_per_case(
    package_dir: Path,
    summary: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
) -> None:
    """G3: package/isolation invariants plus a mandatory per-case disposition check.

    A package-wide nonzero count is never sufficient on its own — each case's
    disposition is recomputed and matched against that case's OWN filtered
    trace, so a missing case's required trace can never be satisfied by
    another case's operations.
    """
    _gate_trace(package_dir, summary)
    trace_events = _load_json(Path(package_dir) / "trace_events.json").get("events") or []
    for proposal in proposals:
        case_id = str(proposal.get("case_id"))
        disposition = classify_case_disposition(proposal)
        case_events = case_trace_events(trace_events, case_id)
        closure_sha = ((proposal.get("extra") or {}).get("closure") or {}).get(
            "evaluation_closure_hash"
        )
        evaluate_case_trace_profile(
            case_id=case_id,
            disposition=disposition,
            case_events=case_events,
            closure_sha=closure_sha,
        )


def _gate_evidence(
    proposals: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    for proposal in proposals:
        rows = proposal.get("candidate_results") or []
        if not rows:
            if classify_case_disposition(proposal) == NOT_CONSTRUCTIBLE:
                continue
            raise CoreContractError("candidate ledger empty")
        for row in rows:
            if row.get("failure_reason"):
                raise CoreContractError("candidate ledger contains failure")
            selected_or_parent = row.get("disposition") in (
                "SELECTED_MATERIAL",
                "SELECTED_CONTROL_NO_MATERIAL",
                "PARENT_OF_SELECTED",
            )
            evidence_scopes = (
                ("search_candidate_evidence_by_fold", {"0", "1"}),
                ("reeval_candidate_evidence_by_fold", {"2"}),
            )
            for scope, required_folds in evidence_scopes:
                evidence = row.get(scope) or {}
                if selected_or_parent and set(evidence) != required_folds:
                    raise CoreContractError(f"selected/parent ledger missing {scope}")
                for fold_row in evidence.values():
                    for field in (
                        "risk",
                        "abnormal_logit",
                        "delta_risk",
                        "delta_logit",
                        "checkpoint_sha256",
                        "semantic_critic_input_sha",
                        "trace_ref",
                        "trace_invocation_id",
                        "baseline_observation_id",
                    ):
                        if fold_row.get(field) is None:
                            raise CoreContractError(f"fold ledger field missing: {field}")
                    check_risk_logit_contract(
                        float(fold_row["risk"]), float(fold_row["abnormal_logit"])
                    )
            if not selected_or_parent:
                continue
            search_baseline = row.get("search_baseline_observation_by_fold") or {}
            reeval_baseline = row.get("reeval_baseline_evidence_by_fold") or {}
            if set(search_baseline) != {"0", "1"} or set(reeval_baseline) != {"2"}:
                raise CoreContractError("fold-complete baseline evidence missing")
            for fold_id, baseline in search_baseline.items():
                for field in (
                    "risk",
                    "logit",
                    "trace_ref",
                    "critic_input_sha",
                    "checkpoint_sha256",
                ):
                    if baseline.get(field) is None:
                        raise CoreContractError(
                            f"search baseline observation missing {field}"
                        )
                check_risk_logit_contract(
                    float(baseline["risk"]), float(baseline["logit"])
                )
                candidate = row["search_candidate_evidence_by_fold"][fold_id]
                expected_delta = float(candidate["risk"]) - float(baseline["risk"])
                if abs(expected_delta - float(row["search_delta_by_fold"][fold_id])) > 1e-12:
                    raise CoreContractError("search ledger delta mismatch")
                expected_logit_delta = float(candidate["abnormal_logit"]) - float(
                    baseline["logit"]
                )
                if (
                    abs(expected_logit_delta - float(candidate["delta_logit"]))
                    > 1e-12
                ):
                    raise CoreContractError("search ledger logit delta mismatch")
            for fold_id, baseline in reeval_baseline.items():
                for field in (
                    "risk",
                    "abnormal_logit",
                    "checkpoint_sha256",
                    "semantic_critic_input_sha",
                    "trace_ref",
                ):
                    if baseline.get(field) is None:
                        raise CoreContractError(
                            f"Fold2 baseline evidence missing {field}"
                        )
                check_risk_logit_contract(
                    float(baseline["risk"]), float(baseline["abnormal_logit"])
                )
                candidate = row["reeval_candidate_evidence_by_fold"][fold_id]
                expected_delta = float(candidate["risk"]) - float(baseline["risk"])
                if abs(expected_delta - float(row["reeval_delta_by_fold"][fold_id])) > 1e-12:
                    raise CoreContractError("Fold2 ledger delta mismatch")
                expected_logit_delta = float(candidate["abnormal_logit"]) - float(
                    baseline["abnormal_logit"]
                )
                if (
                    abs(expected_logit_delta - float(candidate["delta_logit"]))
                    > 1e-12
                ):
                    raise CoreContractError("Fold2 ledger logit delta mismatch")
    recomputed_coverage = derive_coverage_axes(summary.get("case_results") or [])
    for axis, value in recomputed_coverage.items():
        if summary.get(axis) != value:
            raise CoreContractError(f"coverage axis mismatch: {axis}")


def _gate_selection(proposals: Sequence[Mapping[str, Any]]) -> None:
    for proposal in proposals:
        selection = (proposal.get("extra") or {}).get("selection_manifest") or {}
        selected = selection.get("selected") or []
        for row in selected:
            if row.get("disposition") not in (
                "SELECTED_MATERIAL",
                "SELECTED_CONTROL_NO_MATERIAL",
            ):
                raise CoreContractError("invalid selected disposition")
        ledger_pairs = [
            row
            for row in proposal.get("candidate_results") or []
            if row.get("kind") == "PAIR"
        ]
        threshold = (
            (proposal.get("extra") or {}).get("threshold") or {}
        ).get("locked_min_effect_abs_delta")
        if threshold is None:
            raise CoreContractError("selection locked threshold missing")
        recomputed = select_after_search(
            pair_rows=[
                {
                    "candidate_id": row["candidate_id"],
                    "event_ids": row.get("event_ids") or [],
                    "kind": "PAIR",
                    "fold_deltas": row.get("search_delta_by_fold") or {},
                    "transaction_sha": row.get("transaction_sha"),
                }
                for row in ledger_pairs
            ],
            locked_threshold=float(threshold),
            max_pair_budget=10,
            max_control=1,
        )
        projected = [
            (
                str(row.get("candidate_id")),
                str(row.get("disposition")),
                row.get("expected_effect_direction"),
            )
            for row in selected
        ]
        expected = [
            (
                str(row.get("candidate_id")),
                str(row.get("disposition")),
                row.get("expected_effect_direction"),
            )
            for row in recomputed.get("selected") or []
        ]
        if projected != expected:
            raise CoreContractError("selection manifest differs from ledger recomputation")


def _gate_closure(proposals: Sequence[Mapping[str, Any]]) -> None:
    for proposal in proposals:
        rows = proposal.get("candidate_results") or []
        by_sha = {
            str(row.get("transaction_sha")): row
            for row in rows
            if row.get("transaction_sha")
        }
        closure = (proposal.get("extra") or {}).get("closure") or {}
        closure_parents = set(closure.get("required_parent_transaction_shas") or [])
        selected_parent_union = set()
        for row in rows:
            if row.get("kind") != "PAIR" or row.get("disposition") not in (
                "SELECTED_MATERIAL",
                "SELECTED_CONTROL_NO_MATERIAL",
            ):
                continue
            parents = list(row.get("parent_transaction_shas") or [])
            if len(parents) != 2 or len(set(parents)) != 2:
                raise CoreContractError("selected pair exact canonical parents invalid")
            if any(by_sha.get(parent, {}).get("disposition") != "PARENT_OF_SELECTED" for parent in parents):
                raise CoreContractError("selected pair parent role mismatch")
            canonical_parent_rows = sorted(
                (by_sha[parent] for parent in parents),
                key=lambda parent: (
                    tuple(parent.get("event_ids") or []),
                    str(parent.get("candidate_id") or ""),
                ),
            )
            if parents != [
                str(parent["transaction_sha"]) for parent in canonical_parent_rows
            ]:
                raise CoreContractError("selected pair parent canonical order mismatch")
            bundle_atoms = {
                canonical_json_sha256(atomic)
                for atomic in (
                    row.get("canonical_atomic_set_payload") or {}
                ).get("atomics", [])
            }
            parent_atoms = {
                canonical_json_sha256(atomic)
                for parent in canonical_parent_rows
                for atomic in (
                    parent.get("canonical_atomic_set_payload") or {}
                ).get("atomics", [])
            }
            if bundle_atoms != parent_atoms or len(bundle_atoms) != 2:
                raise CoreContractError(
                    "selected pair atomics are not exact parent union"
                )
            selected_parent_union.update(parents)
        if selected_parent_union != closure_parents:
            raise CoreContractError("ledger/closure parent SHA set mismatch")


def _gate_reevaluation(
    proposals: Sequence[Mapping[str, Any]],
    trace_events: Sequence[Mapping[str, Any]],
) -> None:
    for proposal in proposals:
        case_id = str(proposal.get("case_id"))
        closure = (proposal.get("extra") or {}).get("closure") or {}
        closure_sha = closure.get("evaluation_closure_hash")
        if not closure_sha:
            raise CoreContractError("reevaluation closure hash missing")
        closure_candidates = set(closure.get("candidate_identity_hashes") or [])
        for row in proposal.get("candidate_results") or []:
            if row.get("disposition") in (
                "SELECTED_MATERIAL",
                "SELECTED_CONTROL_NO_MATERIAL",
                "PARENT_OF_SELECTED",
            ):
                fold2 = row.get("reeval_candidate_evidence_by_fold") or {}
                if set(fold2) != {"2"}:
                    raise CoreContractError("selected/parent Fold2 evidence incomplete")
                if row.get("candidate_id") not in closure_candidates:
                    raise CoreContractError("reevaluated candidate absent from frozen closure")
        completed_fold2 = [
            event
            for event in trace_events
            if event.get("case_id") == case_id
            and event.get("state") == "COMPLETED"
            and event.get("scope")
            in ("selection_blind_reevaluation", "holdout", "fold2")
            and event.get("kind") in ("CHECKPOINT_FORWARD", "CRITIC_LOAD")
        ]
        if not completed_fold2:
            raise CoreContractError("Fold2 replay trace missing")
        if any(
            event.get("closure_sha") != closure_sha
            or event.get("closure_frozen") is not True
            for event in completed_fold2
        ):
            raise CoreContractError("Fold2 replay not bound to frozen closure")


def _gate_scientific(proposals: Sequence[Mapping[str, Any]]) -> None:
    for proposal in proposals:
        status = str(proposal.get("scientific_status") or "")
        detail = (proposal.get("extra") or {}).get("scientific_detail") or {}
        selected = [
            row
            for row in proposal.get("candidate_results") or []
            if row.get("disposition")
            in ("SELECTED_MATERIAL", "SELECTED_CONTROL_NO_MATERIAL")
        ]
        if selected and status in (
            "NOT_EVALUABLE",
            "ScientificStatus.NOT_EVALUABLE",
        ):
            raise CoreContractError(
                "selected bundle scientific result is NOT_EVALUABLE"
            )
        control_only = bool(selected) and all(
            row.get("disposition") == "SELECTED_CONTROL_NO_MATERIAL"
            for row in selected
        )
        if control_only and status not in ("NOT_EVALUATED", "ScientificStatus.NOT_EVALUATED"):
            raise CoreContractError("control-only case cannot be NOT_SUPPORTED")
        if control_only and detail.get("control_result") not in (
            "CONTROL_WITHIN_LOCKED_THRESHOLD",
            "CONTROL_EXCEEDED_LOCKED_THRESHOLD",
            "CONTROL_NOT_EVALUABLE",
        ):
            raise CoreContractError("control-only result missing")
        threshold = float(
            ((proposal.get("extra") or {}).get("threshold") or {})[
                "locked_min_effect_abs_delta"
            ]
        )
        target_id = detail.get("bundle_candidate_id")
        target = next(
            (row for row in selected if row.get("candidate_id") == target_id),
            selected[0] if selected else None,
        )
        if control_only and target is not None:
            fold2_delta = (target.get("reeval_delta_by_fold") or {}).get("2")
            expected_control = control_result_from_delta(
                delta_risk_value=fold2_delta,
                locked_threshold=threshold,
                evaluable=fold2_delta is not None,
            )
            if detail.get("control_result") != expected_control:
                raise CoreContractError("control result differs from ledger")
        if target is not None and target.get("disposition") == "SELECTED_MATERIAL":
            by_sha = {
                str(row.get("transaction_sha")): row
                for row in proposal.get("candidate_results") or []
            }
            parents = [
                by_sha[parent_sha]
                for parent_sha in target.get("parent_transaction_shas") or []
                if parent_sha in by_sha
            ]
            if len(parents) != 2:
                raise CoreContractError("material scientific parents missing")
            direction = target.get("expected_effect_direction")
            if not direction:
                raise CoreContractError("material expected direction missing")
            recomputed = evaluate_bundle_effects(
                search_fold_deltas=target.get("search_delta_by_fold") or {},
                reeval_fold_deltas=target.get("reeval_delta_by_fold") or {},
                search_parent_deltas_by_fold={
                    fold_id: [
                        float((parent.get("search_delta_by_fold") or {})[fold_id])
                        for parent in parents
                    ]
                    for fold_id in ("0", "1")
                },
                reeval_parent_deltas_by_fold={
                    "2": [
                        float((parent.get("reeval_delta_by_fold") or {})["2"])
                        for parent in parents
                    ]
                },
                expected_effect_direction=str(direction),
                locked_threshold=threshold,
            )
            if recomputed["development_scientific_result"] != status:
                raise CoreContractError("scientific status differs from ledger recomputation")


def _gate_lifecycle(
    package_dir: Path,
    sidecar_dir: Path,
    pre_stable_lock: Mapping[str, Any],
) -> None:
    observation = _load_json(Path(sidecar_dir) / "final_rerun_observation_manifest.json")
    for field in (
        "qualification_node_id_manifest_sha256",
        "final_rerun_test_log_sha256",
        "final_rerun_node_id_manifest_sha256",
        "final_rerun_observation_manifest_sha256",
        "final_rerun_test_log_relpath",
        "final_rerun_node_id_manifest_relpath",
        "executed_at",
        "node_identity",
        "host_identity",
    ):
        if not observation.get(field):
            raise CoreContractError(f"G12 observation missing {field}")
    supplied = observation["final_rerun_observation_manifest_sha256"]
    observed = canonical_json_sha256(
        {
            key: value
            for key, value in observation.items()
            if key != "final_rerun_observation_manifest_sha256"
        }
    )
    if supplied != observed:
        raise CoreContractError("final-rerun observation manifest SHA mismatch")

    package_root = Path(package_dir).resolve()

    def snapshot_path(field: str) -> Path:
        relpath = Path(str(observation[field]))
        if relpath.is_absolute() or ".." in relpath.parts:
            raise CoreContractError(f"G12 snapshot path escapes package: {field}")
        resolved = (package_root / relpath).resolve()
        if package_root not in resolved.parents:
            raise CoreContractError(f"G12 snapshot path escapes package: {field}")
        if not resolved.is_file():
            raise CoreContractError(f"G12 snapshot missing: {field}")
        return resolved

    test_log_path = snapshot_path("final_rerun_test_log_relpath")
    node_manifest_path = snapshot_path("final_rerun_node_id_manifest_relpath")
    if sha256_file(test_log_path) != observation["final_rerun_test_log_sha256"]:
        raise CoreContractError("final-rerun test-log snapshot SHA mismatch")
    if (
        sha256_file(node_manifest_path)
        != observation["final_rerun_node_id_manifest_sha256"]
    ):
        raise CoreContractError("final-rerun node manifest snapshot SHA mismatch")
    node_manifest = _load_json(node_manifest_path)
    if node_manifest.get("node_ids") != observation.get("final_rerun_node_ids"):
        raise CoreContractError("final-rerun node IDs differ from snapshot")
    if node_manifest.get("test_selection") != observation.get(
        "final_rerun_test_selection"
    ):
        raise CoreContractError("final-rerun test selection differs from snapshot")
    for count_field in ("exit_code", "passed_count", "failed_count"):
        value = observation.get(count_field)
        if not isinstance(value, int) or value < 0:
            raise CoreContractError(f"G12 observation invalid {count_field}")
    if observation["exit_code"] != 0 or observation["failed_count"] != 0:
        raise CoreContractError("final-rerun measured result did not pass")
    if (
        observation["qualification_node_id_manifest_sha256"]
        != pre_stable_lock["qualification_node_id_manifest_sha256"]
    ):
        raise CoreContractError(
            "qualification node-ID artifact does not match PRE stable lock"
        )
    if observation.get("qualification_node_ids") != observation.get(
        "final_rerun_node_ids"
    ):
        raise CoreContractError("qualification/final-rerun node IDs differ")
    if observation.get("qualification_test_selection") != observation.get(
        "final_rerun_test_selection"
    ):
        raise CoreContractError("qualification/final-rerun test selection differs")


def _canonical_verdict(body: Mapping[str, Any]) -> Dict[str, Any]:
    verdict = dict(body)
    verdict["gate_map_sha256"] = canonical_json_sha256(verdict["gate_results"])
    verdict["verdict_sha256"] = canonical_json_sha256(
        {key: value for key, value in verdict.items() if key != "verdict_sha256"}
    )
    return verdict


def verify_cf1s_package(
    *,
    phase: str,
    package_dir: Path,
    expected_final_dir: Path,
    sidecar_dir: Path,
    run_id: str,
    authorization_artifact: Mapping[str, Any],
    post_attestation: Mapping[str, Any],
    trust_root: Mapping[str, Any],
    trust_root_sha256: str,
    verifier_code_sha256: str,
    receipt: Optional[Mapping[str, Any]] = None,
    pre_promotion_verdict: Optional[Mapping[str, Any]] = None,
    expected_case_count: int = 3,
    cohort_id: str = "DEVELOPMENT3",
) -> Dict[str, Any]:
    if phase not in (PRE_PROMOTION, FINAL):
        raise CoreContractError("unknown verifier phase")
    package_dir = Path(package_dir)
    expected_final_dir = Path(expected_final_dir).resolve()
    sidecar_dir = Path(sidecar_dir)
    actual_verifier_code_sha256 = compute_verifier_code_sha256()
    if verifier_code_sha256 != actual_verifier_code_sha256:
        raise CoreContractError("verifier code SHA differs from executing verifier")
    summary = _load_json(package_dir / "runner_summary.json")
    trace_events = (
        _load_json(package_dir / "trace_events.json").get("events") or []
    )
    root_artifact = _load_json(sidecar_dir / "EVIDENCE_ROOT.json")
    evidence_root = str(root_artifact["evidence_root_sha"])
    relpaths = dict(root_artifact.get("artifact_relpaths") or {})
    if not relpaths:
        raise CoreContractError("evidence root artifact_relpaths missing")
    recalculated_root = compute_evidence_package_root(
        {
            name: package_dir / relpath
            for name, relpath in relpaths.items()
        },
        root=package_dir,
    )
    if recalculated_root != evidence_root:
        raise CoreContractError("immutable evidence root mismatch")
    proposals = _case_proposals(package_dir, expected_case_count)
    pre_lock = validate_stable_lock_manifest(
        _load_json(package_dir / "pre_stable_lock.json"),
        expected_cohort=cohort_id.lower(),
    )
    post_lock = validate_stable_lock_manifest(
        _load_json(package_dir / "post_stable_lock.json"),
        expected_cohort=cohort_id.lower(),
    )
    final_rerun_observation = _load_json(
        sidecar_dir / "final_rerun_observation_manifest.json"
    )

    gate_results = {f"G{index}": "NOT_RUN" for index in range(1, 13)}
    gate_errors: Dict[str, str] = {}

    def evaluate(gate: str, check: Callable[[], None]) -> None:
        try:
            check()
            gate_results[gate] = _PASS
        except Exception as exc:
            gate_results[gate] = _FAIL
            gate_errors[gate] = str(exc)

    evaluate("G1", lambda: _gate_transaction(proposals))
    evaluate("G2", lambda: _gate_runtime(summary, proposals, pre_lock))
    evaluate("G3", lambda: _gate_trace_per_case(package_dir, summary, proposals))
    evaluate("G4", lambda: _gate_evidence(proposals, summary))
    def check_authorization_chain() -> None:
        consume_pre_execution_authorization(
            authorization_artifact,
            trust_root=trust_root,
            trust_root_sha256=trust_root_sha256,
            observed_stable_lock=pre_lock,
            expected_run_id=run_id,
            expected_cohort_id=cohort_id,
        )
        stable_sha = pre_lock["stable_lock_sha256"]
        if stable_sha != post_lock["stable_lock_sha256"]:
            raise CoreContractError("PRE/POST stable lock mismatch")
        if post_attestation.get("run_id") != run_id:
            raise CoreContractError("POST run_id mismatch")
        if (
            post_attestation.get("pre_execution_payload_sha256")
            != authorization_artifact.get("payload_sha256")
        ):
            raise CoreContractError("POST PRE-authorization link mismatch")
        if post_attestation.get("pre_stable_lock_sha256") != stable_sha:
            raise CoreContractError("POST PRE stable-lock link mismatch")
        if post_attestation.get("post_stable_lock_sha256") != stable_sha:
            raise CoreContractError("POST observed stable-lock link mismatch")
        if post_attestation.get("evidence_root_sha") != evidence_root:
            raise CoreContractError("POST evidence-root link mismatch")

    evaluate("G5", check_authorization_chain)
    evaluate("G6", lambda: _gate_selection(proposals))
    evaluate("G7", lambda: _gate_closure(proposals))
    evaluate("G8", lambda: _gate_reevaluation(proposals, trace_events))
    evaluate("G9", lambda: _gate_scientific(proposals))
    def check_post_attestation() -> None:
        verify_post_attestation_public_key_only(
            post_attestation,
            trust_root=trust_root,
            trust_root_sha256=trust_root_sha256,
            expected_evidence_root=evidence_root,
            expected_pre_payload_sha=str(authorization_artifact["payload_sha256"]),
            expected_run_id=run_id,
            expected_stable_lock_sha256=pre_lock["stable_lock_sha256"],
            expected_final_destination=str(expected_final_dir),
            expected_final_rerun_observation_manifest_sha256=str(
                final_rerun_observation[
                    "final_rerun_observation_manifest_sha256"
                ]
            ),
            expected_cohort_id=cohort_id,
        )

    evaluate("G10", check_post_attestation)
    evaluate("G12", lambda: _gate_lifecycle(package_dir, sidecar_dir, pre_lock))

    pre_gates = [f"G{index}" for index in range(1, 11)] + ["G12"]
    first_failed = next(
        (gate for gate in pre_gates if gate_results[gate] != _PASS),
        None,
    )
    promotion_eligible = first_failed is None

    def build_current_pre_verdict() -> Dict[str, Any]:
        pre_gate_results = dict(gate_results)
        pre_gate_results["G11"] = (
            "NOT_RUN" if promotion_eligible else f"BLOCKED_BY_{first_failed}"
        )
        return _canonical_verdict(
            {
                "verdict_kind": "CF1S_PRE_PROMOTION_VERDICT_V1",
                "verifier_phase": PRE_PROMOTION,
                "schema_version": "CF1S_VERIFIER_VERDICT_V1",
                "verifier_code_sha256": verifier_code_sha256,
                "run_id": run_id,
                "evidence_root_sha256": evidence_root,
                "receipt_sha": None,
                "promotion_eligible": promotion_eligible,
                "gate_results": pre_gate_results,
                "gate_errors": gate_errors,
            }
        )

    recomputed_pre_verdict = build_current_pre_verdict()
    if phase == PRE_PROMOTION:
        return recomputed_pre_verdict

    if not pre_promotion_verdict:
        raise CoreContractError("FINAL requires PRE_PROMOTION verdict")
    expected_pre_sha = canonical_json_sha256(
        {
            key: value
            for key, value in pre_promotion_verdict.items()
            if key != "verdict_sha256"
        }
    )
    if pre_promotion_verdict.get("verdict_sha256") != expected_pre_sha:
        raise CoreContractError("PRE_PROMOTION verdict SHA invalid")
    if not pre_promotion_verdict.get("promotion_eligible"):
        raise CoreContractError("PRE_PROMOTION verdict not eligible")
    if dict(pre_promotion_verdict) != recomputed_pre_verdict:
        raise CoreContractError(
            "supplied PRE_PROMOTION verdict differs from current-input recomputation"
        )

    def check_promotion() -> None:
        if not receipt:
            raise CoreContractError("promotion receipt missing")
        if package_dir.resolve() != expected_final_dir:
            raise CoreContractError(
                "actual package path differs from expected final destination"
            )
        validate_promotion_receipt(
            receipt,
            expected_run_id=run_id,
            expected_final_path=expected_final_dir,
            expected_evidence_root=evidence_root,
            expected_post_attestation_sha=payload_sha256(
                post_attestation,
                exclude_keys=("signature_hex", "payload_sha256"),
            ),
            artifact_relpaths=relpaths,
        )

    evaluate("G11", check_promotion)
    all_pass = all(value == _PASS for value in gate_results.values())
    return _canonical_verdict(
        {
            "verdict_kind": "CF1S_FINAL_VERIFIER_VERDICT_V1",
            "verifier_phase": FINAL,
            "schema_version": "CF1S_VERIFIER_VERDICT_V1",
            "verifier_code_sha256": verifier_code_sha256,
            "run_id": run_id,
            "evidence_root_sha256": evidence_root,
            "receipt_sha": (receipt or {}).get("receipt_sha256"),
            "source_pre_promotion_verdict_sha256": pre_promotion_verdict.get(
                "verdict_sha256"
            ),
            "final_report_allowed": all_pass,
            "gate_results": gate_results,
            "gate_errors": gate_errors,
        }
    )


def project_completion(final_verdict: Mapping[str, Any]) -> Dict[str, Any]:
    supplied_sha = final_verdict.get("verdict_sha256")
    observed_sha = canonical_json_sha256(
        {key: value for key, value in final_verdict.items() if key != "verdict_sha256"}
    )
    if supplied_sha != observed_sha:
        raise CoreContractError("FINAL verdict SHA invalid")
    gates = dict(final_verdict.get("gate_results") or {})
    if set(gates) != {f"G{index}" for index in range(1, 13)}:
        raise CoreContractError("FINAL gate map incomplete")
    if any(value != _PASS for value in gates.values()):
        raise CoreContractError("completion forbidden unless G1-G12 all PASS")
    if not final_verdict.get("final_report_allowed"):
        raise CoreContractError("FINAL final_report_allowed=false")
    completion = {
        "artifact_kind": "CF1S_EXTERNAL_COMPLETION_PROJECTION_V1",
        "run_id": final_verdict["run_id"],
        "source_verdict_sha256": supplied_sha,
        "gate_map_sha256": final_verdict["gate_map_sha256"],
        "evidence_root_sha256": final_verdict["evidence_root_sha256"],
        "receipt_sha": final_verdict["receipt_sha"],
        "gate_results": gates,
        "final_report_allowed": True,
    }
    completion["completion_sha256"] = canonical_json_sha256(completion)
    return completion


def validate_completion_projection(
    completion: Mapping[str, Any], final_verdict: Mapping[str, Any]
) -> bool:
    expected = project_completion(final_verdict)
    if dict(completion) != expected:
        raise CoreContractError("completion projection does not exactly match FINAL verdict")
    return True
