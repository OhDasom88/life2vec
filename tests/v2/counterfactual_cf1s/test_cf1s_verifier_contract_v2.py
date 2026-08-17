from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
    compute_evidence_package_root,
    consume_pre_execution_authorization,
    issue_post_execution_attestation,
    issue_pre_execution_authorization,
    load_trust_root,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
    canonical_json_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
    CoreContractError,
    sha256_file,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
    REQUIRED_STABLE_LOCK_SHA_FIELDS,
    validate_stable_lock_manifest,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
    atomic_promote_directory,
    readback_final_package,
    write_promotion_receipt,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_expansion import (
    authorize_validation20,
    build_train35_reference_report,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
    build_allowed_change_set,
    build_validated_raw_transaction,
    verify_exact_parent_transactions,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_scientific import (
    derive_coverage_axes,
    scientific_status_from_effects,
    select_observed_median,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_trace import (
    GlobalForwardTrace,
    TraceKind,
    validate_persisted_trace,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_verifier import (
    FINAL,
    PRE_PROMOTION,
    compute_verifier_code_sha256,
    project_completion,
    validate_completion_projection,
    verify_cf1s_package,
)
from tests.v2.counterfactual_cf1s.cf1s_test_tx import (
    make_test_candidate_transaction,
)


ROOT = Path("/home/dasom/life2vec")
TRUST = ROOT / "conf/m1/cf1s_policies/CF1S_DEVELOPMENT_TRUST_ROOT_V1.json"
DEV_PRIV = "d3d340008b8a043b537d3a49093983854381f0863d5acba31ef712910162facc"


def _stable() -> dict:
    body = {
        "version": "CF1S_STABLE_LOCK_MANIFEST_V2",
        "cohort": "development3",
        "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
        "execution_scope": "TWO_EVENT_ONLY",
        "runtime_versions": {},
        "risk_head_contract": {},
        "artifact_sources": {},
    }
    body.update({field: "a" * 64 for field in REQUIRED_STABLE_LOCK_SHA_FIELDS})
    body["stable_lock_sha256"] = canonical_json_sha256(body)
    return body


def test_stable_lock_null_and_placeholder_fail_closed():
    stable = _stable()
    for invalid in (None, "", "TODO", "a" * 63):
        bad = dict(stable)
        bad["qualification_test_log_sha256"] = invalid
        bad["stable_lock_sha256"] = canonical_json_sha256(
            {key: value for key, value in bad.items() if key != "stable_lock_sha256"}
        )
        with pytest.raises(CoreContractError):
            validate_stable_lock_manifest(bad)


def test_pre_authorization_is_bound_to_run_and_full_stable_lock():
    trust = load_trust_root(TRUST)
    stable = _stable()
    artifact = issue_pre_execution_authorization(
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        stable_lock_manifest=stable,
        run_id="run-a",
    )
    assert consume_pre_execution_authorization(
        artifact,
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        observed_stable_lock=stable,
        expected_run_id="run-a",
    )["ok"]
    with pytest.raises(CoreContractError, match="run_id"):
        consume_pre_execution_authorization(
            artifact,
            trust_root=trust,
            trust_root_sha256=sha256_file(TRUST),
            observed_stable_lock=stable,
            expected_run_id="run-b",
        )


def test_persisted_trace_recalculation_and_mutation_failure():
    trace = GlobalForwardTrace()
    trace.begin_operation(
        kind=TraceKind.ATTRIBUTION_FORWARD,
        invocation_id="ixg-1",
        phase="ATTRIBUTION",
        scope="search",
    )
    trace.start(
        kind=TraceKind.ATTRIBUTION_FORWARD,
        invocation_id="ixg-1",
        phase="ATTRIBUTION",
        scope="search",
    )
    trace.complete(
        kind=TraceKind.ATTRIBUTION_FORWARD,
        invocation_id="ixg-1",
        phase="ATTRIBUTION",
        scope="search",
    )
    summary = validate_persisted_trace(trace.events)
    assert summary["attribution_forward_count"] == 1

    mutated = [dict(event) for event in trace.events]
    mutated[-1]["scope"] = "fold2"
    with pytest.raises(CoreContractError, match="event_sha"):
        validate_persisted_trace(mutated)


def test_failed_trace_requires_explicit_successful_retry_lineage():
    trace = GlobalForwardTrace()
    kwargs = dict(
        kind=TraceKind.CRITIC_LOAD,
        invocation_id="load-1",
        phase="SEARCH",
        scope="search",
        extra={"logical_operation_id": "critic-fold-0"},
    )
    trace.begin_operation(**kwargs)
    trace.start(**kwargs)
    trace.fail(**kwargs, failure_code="IOError")
    with pytest.raises(CoreContractError, match="without successful retry"):
        validate_persisted_trace(trace.events)


def test_exact_parent_chain_and_canonical_order():
    a = make_test_candidate_transaction(case_id="case", event_id="e1", feature_id="f")
    b = make_test_candidate_transaction(case_id="case", event_id="e2", feature_id="f")
    allowed = build_allowed_change_set(
        targeted_raw_fields=["raw.f"],
        tokenizer_derived_tokens=["tokens.e1", "tokens.e2"],
    )
    bundle = build_validated_raw_transaction(
        case_id="case",
        atomics=[b.atomics[0], a.atomics[0]],
        allowed_change_set=allowed,
        original_caseevents_sha=a.original_caseevents_sha,
        edited_caseevents_sha="e" * 64,
        skip_edited_sha_required=True,
    )
    parents = sorted([a, b], key=lambda row: row.atomics[0].sort_key())
    shas = [row.canonical_validated_transaction_sha for row in parents]
    assert verify_exact_parent_transactions(
        bundle=bundle,
        parents=[b, a],
        declared_parent_shas=shas,
    ) == shas
    with pytest.raises(CoreContractError):
        verify_exact_parent_transactions(
            bundle=bundle,
            parents=[a, b],
            declared_parent_shas=[],
        )


def test_control_only_is_not_evaluated_and_coverage_is_separate():
    status = scientific_status_from_effects(
        stable_effect_replicated=False,
        multievent_increment_supported=False,
        no_material_control=True,
    )
    assert status["development_scientific_result"] == "NOT_EVALUATED"
    axes = derive_coverage_axes(
        [
            {
                "execution_complete": True,
                "selection_disposition": "SELECTED_CONTROL_NO_MATERIAL",
                "scientifically_evaluable": False,
                "control_result": "CONTROL_WITHIN_LOCKED_THRESHOLD",
            }
        ]
    )
    assert axes["execution_coverage"]["status"] == "COMPLETE"
    assert axes["control_evaluation_coverage"]["status"] == "COMPLETE"
    assert axes["material_effect_evaluation_coverage"]["status"] == "NONE"


def test_observation_median_never_mixes_risk_and_logit():
    observations = [
        {
            "observation_id": "b",
            "risk": 0.5,
            "logit": 0.0,
            "trace_ref": "t2",
            "critic_input_sha": "b" * 64,
        },
        {
            "observation_id": "a",
            "risk": 0.2689414213699951,
            "logit": -1.0,
            "trace_ref": "t1",
            "critic_input_sha": "a" * 64,
        },
    ]
    selected = select_observed_median(observations)
    assert selected["observation_id"] == "a"
    assert selected["logit"] == -1.0


def test_completion_only_projects_all_pass_final_verdict():
    gates = {f"G{index}": "PASS" for index in range(1, 13)}
    verdict = {
        "verdict_kind": "CF1S_FINAL_VERIFIER_VERDICT_V1",
        "verifier_phase": "FINAL",
        "schema_version": "CF1S_VERIFIER_VERDICT_V1",
        "verifier_code_sha256": "a" * 64,
        "run_id": "run",
        "evidence_root_sha256": "b" * 64,
        "receipt_sha": "c" * 64,
        "final_report_allowed": True,
        "gate_results": gates,
        "gate_errors": {},
        "gate_map_sha256": canonical_json_sha256(gates),
    }
    verdict["verdict_sha256"] = canonical_json_sha256(verdict)
    completion = project_completion(verdict)
    assert validate_completion_projection(completion, verdict)
    verdict["gate_results"]["G3"] = "NOT_RUN"
    verdict["gate_map_sha256"] = canonical_json_sha256(verdict["gate_results"])
    verdict["verdict_sha256"] = canonical_json_sha256(
        {key: value for key, value in verdict.items() if key != "verdict_sha256"}
    )
    with pytest.raises(CoreContractError):
        project_completion(verdict)


def test_validation20_blocks_null_inputs_and_train35_is_reference_only():
    gates = {f"G{index}": "PASS" for index in range(1, 13)}
    verdict = {
        "verdict_kind": "CF1S_FINAL_VERIFIER_VERDICT_V1",
        "verifier_phase": "FINAL",
        "schema_version": "CF1S_VERIFIER_VERDICT_V1",
        "verifier_code_sha256": "a" * 64,
        "run_id": "dev-run",
        "evidence_root_sha256": "b" * 64,
        "receipt_sha": "c" * 64,
        "final_report_allowed": True,
        "gate_results": gates,
        "gate_errors": {},
        "gate_map_sha256": canonical_json_sha256(gates),
    }
    verdict["verdict_sha256"] = canonical_json_sha256(verdict)
    completion = project_completion(verdict)
    validation = {
        "ordered_case_ids": [f"v{index}" for index in range(20)],
        "source_problem20_manifest_sha256": "d" * 64,
        "selection_rule": "LOCKED_BEFORE_EXECUTION",
        "cases": [
            {"case_id": f"v{index}", "input_artifact_sha256": None}
            for index in range(20)
        ],
    }
    with pytest.raises(CoreContractError, match="input_artifact"):
        authorize_validation20(
            development_final_verdict=verdict,
            development_completion=completion,
            validation_manifest=validation,
            problem20_manifest_sha256="d" * 64,
        )
    report = build_train35_reference_report(
        development3_evidence_root_sha256="a" * 64,
        primary32_evidence_root_sha256="b" * 64,
        development3_case_ids=["d1", "d2", "d3"],
        primary32_case_ids=[f"p{index}" for index in range(32)],
    )
    assert report["case_count"] == 35
    assert report["copies_source_evidence"] is False


def test_two_phase_verifier_end_to_end(tmp_path):
    def write(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    qualification_node = tmp_path / "qualification_nodes.json"
    qualification_preflight = tmp_path / "qualification_preflight.json"
    write(qualification_node, {"node_ids": ["n1", "n2"]})
    checks = [
        {
            "check_name": name,
            "target_sha256": "1" * 64,
            "observations": {"measured": 0.0},
            "tolerance": 0.0,
            "status": "PASS",
            "producer_code_sha256": "2" * 64,
        }
        for name in (
            "cold_rebuild_equivalence",
            "official_batch_field_parity",
            "stage_a_critic_pairing",
            "fold_isolation",
            "deterministic_runtime",
            "risk_logit_contract",
        )
    ]
    write(qualification_preflight, {"checks": checks})
    stable = _stable()
    stable["qualification_preflight_sha256"] = sha256_file(
        qualification_preflight
    )
    stable["qualification_node_id_manifest_sha256"] = sha256_file(
        qualification_node
    )
    stable["artifact_sources"] = {
        "qualification_preflight_sha256": str(qualification_preflight)
    }
    stable["stable_lock_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in stable.items()
            if key != "stable_lock_sha256"
        }
    )
    trust = load_trust_root(TRUST)
    run_id = "verifier-e2e"
    authorization = issue_pre_execution_authorization(
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        stable_lock_manifest=stable,
        run_id=run_id,
    )

    package = tmp_path / "quarantine" / run_id
    sidecar = tmp_path / "sidecars" / run_id
    package.mkdir(parents=True)
    sidecar.mkdir(parents=True)

    trace = GlobalForwardTrace()
    case_ids = ["case-1", "case-2", "case-3"]
    closures = {}

    def traced(kind, invocation_id, case_id, fold_id, scope, phase, extra=None):
        kwargs = {
            "kind": kind,
            "invocation_id": invocation_id,
            "case_id": case_id,
            "fold_id": fold_id,
            "scope": scope,
            "phase": phase,
        }
        if extra is not None:
            kwargs["extra"] = extra
        trace.begin_operation(**kwargs)
        trace.start(**kwargs)
        trace.complete(**kwargs)

    proposals = []
    case_results = []
    candidate_risk = 0.5001
    candidate_logit = math.log(candidate_risk / (1.0 - candidate_risk))
    for case_index, case_id in enumerate(case_ids):
        for fold_id in (0, 1):
            traced(
                TraceKind.ATTRIBUTION_MODEL_LOAD,
                f"attr-load-{case_index}-{fold_id}",
                case_id,
                fold_id,
                "search",
                "ATTRIBUTION",
            )
            traced(
                TraceKind.ATTRIBUTION_FORWARD,
                f"attr-forward-{case_index}-{fold_id}",
                case_id,
                fold_id,
                "search",
                "ATTRIBUTION",
            )
        traced(
            TraceKind.STAGE_A_LOAD,
            f"stage-a-{case_index}",
            case_id,
            None,
            "search",
            "SEARCH_BASELINE_IDENTITY",
        )
        traced(
            TraceKind.CRITIC_LOAD,
            f"critic-search-{case_index}",
            case_id,
            0,
            "search",
            "SEARCH_EFFECTS",
        )
        traced(
            TraceKind.CHECKPOINT_FORWARD,
            f"checkpoint-search-{case_index}",
            case_id,
            0,
            "search",
            "SEARCH_EFFECTS",
        )
        traced(
            TraceKind.PIPELINE_INVOCATION,
            f"pipeline-{case_index}",
            case_id,
            None,
            "search",
            "SEARCH_EFFECTS",
            extra={"forward_kind": "CANDIDATE_EFFECT_FORWARD"},
        )

        parent_a = make_test_candidate_transaction(
            case_id=case_id, event_id="e1", feature_id="f"
        )
        parent_b = make_test_candidate_transaction(
            case_id=case_id, event_id="e2", feature_id="f"
        )
        parents = sorted(
            [parent_a, parent_b], key=lambda value: value.atomics[0].sort_key()
        )
        bundle = build_validated_raw_transaction(
            case_id=case_id,
            atomics=[parent.atomics[0] for parent in parents],
            allowed_change_set=build_allowed_change_set(
                targeted_raw_fields=["raw.f"],
                tokenizer_derived_tokens=["tokens.e1", "tokens.e2"],
            ),
            original_caseevents_sha=parent_a.original_caseevents_sha,
            edited_caseevents_sha="e" * 64,
            skip_edited_sha_required=True,
        )

        def tx_payload(tx):
            atomics = [
                atomic.to_dict()
                for atomic in sorted(tx.atomics, key=lambda value: value.sort_key())
            ]
            return (
                {
                    "case_id": tx.case_id,
                    "atomics": atomics,
                    "allowed_change_set_sha": tx.allowed_change_set.sha256,
                    "original_caseevents_sha": tx.original_caseevents_sha,
                    "transaction_mode": tx.transaction_mode,
                    "identity": bool(tx.identity),
                },
                {"case_id": tx.case_id, "atomics": atomics},
            )

        baseline_search = {
            fold_id: {
                "observation_id": f"{case_id}-baseline-{fold_id}",
                "risk": 0.5,
                "logit": 0.0,
                "trace_ref": f"base-{case_id}-{fold_id}",
                "critic_input_sha": "3" * 64,
                "checkpoint_sha256": "4" * 64,
            }
            for fold_id in ("0", "1")
        }
        baseline_reeval = {
            "2": {
                "risk": 0.5,
                "abnormal_logit": 0.0,
                "checkpoint_sha256": "5" * 64,
                "semantic_critic_input_sha": "6" * 64,
                "trace_ref": f"base-{case_id}-2",
                "trace_invocation_id": f"base-{case_id}-2",
            }
        }

        def ledger_row(tx, candidate_id, event_ids, kind, disposition, parent_shas):
            validated, atomic_set = tx_payload(tx)
            search_evidence = {
                fold_id: {
                    "risk": candidate_risk,
                    "abnormal_logit": candidate_logit,
                    "delta_risk": candidate_risk - 0.5,
                    "delta_logit": candidate_logit,
                    "checkpoint_sha256": "4" * 64,
                    "semantic_critic_input_sha": "3" * 64,
                    "trace_ref": f"candidate-{case_id}-{fold_id}",
                    "trace_invocation_id": f"candidate-{case_id}-{fold_id}",
                    "baseline_observation_id": baseline_search[fold_id][
                        "observation_id"
                    ],
                }
                for fold_id in ("0", "1")
            }
            reeval_evidence = {
                "2": {
                    "risk": candidate_risk,
                    "abnormal_logit": candidate_logit,
                    "delta_risk": candidate_risk - 0.5,
                    "delta_logit": candidate_logit,
                    "checkpoint_sha256": "5" * 64,
                    "semantic_critic_input_sha": "6" * 64,
                    "trace_ref": f"candidate-{case_id}-2",
                    "trace_invocation_id": f"candidate-{case_id}-2",
                    "baseline_observation_id": f"reeval-{case_id}-2",
                }
            }
            return {
                "candidate_id": candidate_id,
                "kind": kind,
                "event_ids": event_ids,
                "transaction_sha": tx.canonical_validated_transaction_sha,
                "validated_transaction_payload": validated,
                "canonical_atomic_set_payload": atomic_set,
                "canonical_atomic_set_sha": tx.canonical_atomic_set_sha,
                "parent_transaction_shas": parent_shas,
                "gate0_status": "PASS",
                "gate4_status": "PASS",
                "disposition": disposition,
                "failure_reason": None,
                "expected_effect_direction": None,
                "stage_a_output_sha_search": "7" * 64,
                "stage_a_output_sha_reeval": "8" * 64,
                "search_baseline_observation_by_fold": baseline_search,
                "search_candidate_evidence_by_fold": search_evidence,
                "search_delta_by_fold": {
                    "0": candidate_risk - 0.5,
                    "1": candidate_risk - 0.5,
                },
                "reeval_baseline_evidence_by_fold": baseline_reeval,
                "reeval_candidate_evidence_by_fold": reeval_evidence,
                "reeval_delta_by_fold": {"2": candidate_risk - 0.5},
            }

        parent_rows = [
            ledger_row(
                parent,
                f"parent-{index}",
                [parent.atomics[0].event_id],
                "ATOMIC",
                "PARENT_OF_SELECTED",
                [],
            )
            for index, parent in enumerate(parents)
        ]
        parent_shas = [
            parent.canonical_validated_transaction_sha for parent in parents
        ]
        pair_row = ledger_row(
            bundle,
            "pair",
            ["e1", "e2"],
            "PAIR",
            "SELECTED_CONTROL_NO_MATERIAL",
            parent_shas,
        )
        closure_sha = canonical_json_sha256({"case_id": case_id, "closure": True})
        closures[case_id] = closure_sha
        trace.mark_closure_frozen(closure_sha=closure_sha)
        traced(
            TraceKind.CRITIC_LOAD,
            f"critic-fold2-{case_index}",
            case_id,
            2,
            "selection_blind_reevaluation",
            "REEVAL_EFFECTS",
        )
        traced(
            TraceKind.CHECKPOINT_FORWARD,
            f"checkpoint-fold2-{case_index}",
            case_id,
            2,
            "selection_blind_reevaluation",
            "REEVAL_EFFECTS",
        )
        proposal = {
            "case_id": case_id,
            "scientific_status": "NOT_EVALUATED",
            "execution_evidence": {
                "production_input_verified": True,
                "pipeline_integrity_verified": True,
                "full_sequence_cold_rebuild": True,
                "baseline_forward_complete": True,
                "identity_forward_complete": True,
            },
            "candidate_results": parent_rows + [pair_row],
            "extra": {
                "threshold": {"locked_min_effect_abs_delta": 0.001},
                "selection_manifest": {
                    "selected": [
                        {
                            "candidate_id": "pair",
                            "event_ids": ["e1", "e2"],
                            "kind": "PAIR",
                            "disposition": "SELECTED_CONTROL_NO_MATERIAL",
                            "expected_effect_direction": None,
                        }
                    ]
                },
                "closure": {
                    "evaluation_closure_hash": closure_sha,
                    "candidate_identity_hashes": [
                        "parent-0",
                        "parent-1",
                        "pair",
                    ],
                    "required_parent_transaction_shas": sorted(parent_shas),
                },
                "scientific_detail": {
                    "bundle_candidate_id": "pair",
                    "development_scientific_result": "NOT_EVALUATED",
                    "control_result": "CONTROL_WITHIN_LOCKED_THRESHOLD",
                },
            },
        }
        proposals.append(proposal)
        case_results.append(
            {
                "case_id": case_id,
                "execution_complete": True,
                "selection_disposition": "SELECTED_CONTROL_NO_MATERIAL",
                "scientifically_evaluable": False,
                "control_result": "CONTROL_WITHIN_LOCKED_THRESHOLD",
            }
        )

    coverage = derive_coverage_axes(case_results)
    trace_summary = trace.summarize()
    summary = {
        "execution_scope": "TWO_EVENT_ONLY",
        "three_event_execution_status": "OUT_OF_SCOPE",
        "runtime_lock": {"policy": "EXACT_BYTE_DETERMINISTIC"},
        "trace_summary": {
            key: value for key, value in trace_summary.items() if key != "events"
        },
        "case_results": case_results,
        **coverage,
    }
    artifact_paths = {}
    for name, value in (
        ("runner_summary.json", summary),
        ("pre_stable_lock.json", stable),
        ("post_stable_lock.json", stable),
        ("runtime_observation.json", {"run_id": run_id}),
        ("trace_events.json", {"events": trace.events}),
    ):
        path = package / name
        write(path, value)
        artifact_paths[name] = path
    report_path = package / "REPORT.txt"
    report_path.write_text("contract evidence\n", encoding="utf-8")
    artifact_paths["REPORT.txt"] = report_path
    final_rerun_dir = package / "final_rerun"
    final_rerun_dir.mkdir()
    final_rerun_log = final_rerun_dir / "final_rerun_test.log"
    final_rerun_log.write_text(
        "1 passed in 0.01s\nexit_code=0\n",
        encoding="utf-8",
    )
    final_rerun_node = final_rerun_dir / "final_rerun_node_id_manifest.json"
    write(
        final_rerun_node,
        {"node_ids": ["n1", "n2"], "test_selection": ["test-a"]},
    )
    artifact_paths["final_rerun/final_rerun_test.log"] = final_rerun_log
    artifact_paths[
        "final_rerun/final_rerun_node_id_manifest.json"
    ] = final_rerun_node
    for case_id, proposal in zip(case_ids, proposals):
        path = package / case_id / "case_edit_proposal.json"
        write(path, proposal)
        artifact_paths[f"{case_id}/case_edit_proposal.json"] = path

    evidence_root = compute_evidence_package_root(
        artifact_paths,
        root=package,
    )
    relpaths = {
        name: str(path.relative_to(package))
        for name, path in artifact_paths.items()
    }
    write(
        sidecar / "EVIDENCE_ROOT.json",
        {
            "evidence_root_sha": evidence_root,
            "artifact_relpaths": relpaths,
        },
    )
    final_observation = {
        "artifact_kind": "CF1S_FINAL_RERUN_OBSERVATION_MANIFEST_V1",
        "qualification_node_id_manifest_sha256": sha256_file(qualification_node),
        "final_rerun_test_log_sha256": sha256_file(final_rerun_log),
        "final_rerun_node_id_manifest_sha256": sha256_file(final_rerun_node),
        "final_rerun_test_log_relpath": (
            "final_rerun/final_rerun_test.log"
        ),
        "final_rerun_node_id_manifest_relpath": (
            "final_rerun/final_rerun_node_id_manifest.json"
        ),
        "qualification_node_ids": ["n1", "n2"],
        "final_rerun_node_ids": ["n1", "n2"],
        "qualification_test_selection": ["test-a"],
        "final_rerun_test_selection": ["test-a"],
        "exit_code": 0,
        "passed_count": 1,
        "failed_count": 0,
        "executed_at": "2026-07-19T00:00:00Z",
        "node_identity": "qualification-node",
        "host_identity": "test-host",
    }
    final_observation["final_rerun_observation_manifest_sha256"] = (
        canonical_json_sha256(final_observation)
    )
    write(sidecar / "final_rerun_observation_manifest.json", final_observation)
    final_dir = tmp_path / "final" / run_id
    post = issue_post_execution_attestation(
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        evidence_root_sha=evidence_root,
        pre_execution_payload_sha256=authorization["payload_sha256"],
        acceptance_summary={"execution_scope": "TWO_EVENT_ONLY"},
        pre_post_lock_identical=True,
        run_id=run_id,
        pre_stable_lock_sha256=stable["stable_lock_sha256"],
        post_stable_lock_sha256=stable["stable_lock_sha256"],
        intended_final_destination=str(final_dir),
        final_rerun_observation_manifest_sha256=final_observation[
            "final_rerun_observation_manifest_sha256"
        ],
    )
    pre_verdict = verify_cf1s_package(
        phase=PRE_PROMOTION,
        package_dir=package,
        expected_final_dir=final_dir,
        sidecar_dir=sidecar,
        run_id=run_id,
        authorization_artifact=authorization,
        post_attestation=post,
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        verifier_code_sha256=compute_verifier_code_sha256(),
    )
    assert pre_verdict["promotion_eligible"] is True

    wrong_destination_post = issue_post_execution_attestation(
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        evidence_root_sha=evidence_root,
        pre_execution_payload_sha256=authorization["payload_sha256"],
        acceptance_summary={"execution_scope": "TWO_EVENT_ONLY"},
        pre_post_lock_identical=True,
        run_id=run_id,
        pre_stable_lock_sha256=stable["stable_lock_sha256"],
        post_stable_lock_sha256=stable["stable_lock_sha256"],
        intended_final_destination=str(tmp_path / "wrong-final"),
        final_rerun_observation_manifest_sha256=final_observation[
            "final_rerun_observation_manifest_sha256"
        ],
    )
    wrong_destination_verdict = verify_cf1s_package(
        phase=PRE_PROMOTION,
        package_dir=package,
        expected_final_dir=final_dir,
        sidecar_dir=sidecar,
        run_id=run_id,
        authorization_artifact=authorization,
        post_attestation=wrong_destination_post,
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        verifier_code_sha256=compute_verifier_code_sha256(),
    )
    assert wrong_destination_verdict["gate_results"]["G10"] == "FAIL"

    missing_snapshot_observation = {
        **final_observation,
        "final_rerun_test_log_relpath": "final_rerun/missing.log",
        "final_rerun_test_log_sha256": "9" * 64,
    }
    missing_snapshot_observation[
        "final_rerun_observation_manifest_sha256"
    ] = canonical_json_sha256(
        {
            key: value
            for key, value in missing_snapshot_observation.items()
            if key != "final_rerun_observation_manifest_sha256"
        }
    )
    write(
        sidecar / "final_rerun_observation_manifest.json",
        missing_snapshot_observation,
    )
    missing_snapshot_post = issue_post_execution_attestation(
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        issuer_id="cf1s-development-issuer-v1",
        key_id="dev-key-1",
        private_key_hex=DEV_PRIV,
        evidence_root_sha=evidence_root,
        pre_execution_payload_sha256=authorization["payload_sha256"],
        acceptance_summary={"execution_scope": "TWO_EVENT_ONLY"},
        pre_post_lock_identical=True,
        run_id=run_id,
        pre_stable_lock_sha256=stable["stable_lock_sha256"],
        post_stable_lock_sha256=stable["stable_lock_sha256"],
        intended_final_destination=str(final_dir),
        final_rerun_observation_manifest_sha256=missing_snapshot_observation[
            "final_rerun_observation_manifest_sha256"
        ],
    )
    missing_snapshot_verdict = verify_cf1s_package(
        phase=PRE_PROMOTION,
        package_dir=package,
        expected_final_dir=final_dir,
        sidecar_dir=sidecar,
        run_id=run_id,
        authorization_artifact=authorization,
        post_attestation=missing_snapshot_post,
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        verifier_code_sha256=compute_verifier_code_sha256(),
    )
    assert missing_snapshot_verdict["gate_results"]["G12"] == "FAIL"
    write(sidecar / "final_rerun_observation_manifest.json", final_observation)

    promotion = atomic_promote_directory(
        quarantine_dir=package,
        final_dir=final_dir,
    )
    readback = readback_final_package(
        final_dir=final_dir,
        expected_evidence_root=evidence_root,
        artifact_relpaths=relpaths,
    )
    _, receipt = write_promotion_receipt(
        receipts_dir=tmp_path / "receipts",
        run_id=run_id,
        final_path=final_dir,
        quarantine_path=str(package),
        evidence_root_sha=evidence_root,
        post_attestation_sha=post["payload_sha256"],
        promotion_meta=promotion,
        final_readback=readback,
    )
    with pytest.raises(CoreContractError, match="semantic fields differ"):
        write_promotion_receipt(
            receipts_dir=tmp_path / "receipts",
            run_id=run_id,
            final_path=final_dir,
            quarantine_path=str(package),
            evidence_root_sha=evidence_root,
            post_attestation_sha="e" * 64,
            promotion_meta=promotion,
            final_readback=readback,
        )

    foreign_pre_verdict = dict(pre_verdict)
    foreign_pre_verdict["run_id"] = "foreign-run"
    foreign_pre_verdict["verdict_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in foreign_pre_verdict.items()
            if key != "verdict_sha256"
        }
    )
    with pytest.raises(CoreContractError, match="recomputation"):
        verify_cf1s_package(
            phase=FINAL,
            package_dir=final_dir,
            expected_final_dir=final_dir,
            sidecar_dir=sidecar,
            run_id=run_id,
            authorization_artifact=authorization,
            post_attestation=post,
            trust_root=trust,
            trust_root_sha256=sha256_file(TRUST),
            verifier_code_sha256=compute_verifier_code_sha256(),
            receipt=receipt,
            pre_promotion_verdict=foreign_pre_verdict,
        )

    forged_receipt = {
        **receipt,
        "post_attestation_sha": "f" * 64,
        "final_readback": {"ok": True},
    }
    forged_receipt["receipt_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in forged_receipt.items()
            if key != "receipt_sha256"
        }
    )
    forged_receipt_verdict = verify_cf1s_package(
        phase=FINAL,
        package_dir=final_dir,
        expected_final_dir=final_dir,
        sidecar_dir=sidecar,
        run_id=run_id,
        authorization_artifact=authorization,
        post_attestation=post,
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        verifier_code_sha256=compute_verifier_code_sha256(),
        receipt=forged_receipt,
        pre_promotion_verdict=pre_verdict,
    )
    assert forged_receipt_verdict["gate_results"]["G11"] == "FAIL"

    final_verdict = verify_cf1s_package(
        phase=FINAL,
        package_dir=final_dir,
        expected_final_dir=final_dir,
        sidecar_dir=sidecar,
        run_id=run_id,
        authorization_artifact=authorization,
        post_attestation=post,
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        verifier_code_sha256=compute_verifier_code_sha256(),
        receipt=receipt,
        pre_promotion_verdict=pre_verdict,
    )
    assert all(value == "PASS" for value in final_verdict["gate_results"].values())
