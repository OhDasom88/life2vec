"""P0-6 failure-injection: G3 must be a per-case disposition/trace check, not a
package-wide nonzero-count shortcut."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
    compute_evidence_package_root,
    issue_post_execution_attestation,
    issue_pre_execution_authorization,
    load_trust_root,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
    canonical_json_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import sha256_file
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
    REQUIRED_STABLE_LOCK_SHA_FIELDS,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_trace import (
    GlobalForwardTrace,
    TraceKind,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_verifier import (
    PRE_PROMOTION,
    compute_verifier_code_sha256,
    verify_cf1s_package,
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


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _selected_case(trace: GlobalForwardTrace, case_id: str, closure_sha: str) -> dict:
    """A CONSTRUCTIBLE_SELECTED case with a fully valid trace + proposal."""
    for fold_id in (0, 1):
        trace.begin_operation(
            kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
            invocation_id=f"attr-load-{case_id}-{fold_id}",
            phase="ATTRIBUTION",
            scope="search",
            case_id=case_id,
            fold_id=fold_id,
        )
        trace.start(
            kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
            invocation_id=f"attr-load-{case_id}-{fold_id}",
            phase="ATTRIBUTION",
            scope="search",
            case_id=case_id,
            fold_id=fold_id,
        )
        trace.complete(
            kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
            invocation_id=f"attr-load-{case_id}-{fold_id}",
            phase="ATTRIBUTION",
            scope="search",
            case_id=case_id,
            fold_id=fold_id,
        )
        trace.begin_operation(
            kind=TraceKind.ATTRIBUTION_FORWARD,
            invocation_id=f"attr-fwd-{case_id}-{fold_id}",
            phase="ATTRIBUTION",
            scope="search",
            case_id=case_id,
            fold_id=fold_id,
        )
        trace.start(
            kind=TraceKind.ATTRIBUTION_FORWARD,
            invocation_id=f"attr-fwd-{case_id}-{fold_id}",
            phase="ATTRIBUTION",
            scope="search",
            case_id=case_id,
            fold_id=fold_id,
        )
        trace.complete(
            kind=TraceKind.ATTRIBUTION_FORWARD,
            invocation_id=f"attr-fwd-{case_id}-{fold_id}",
            phase="ATTRIBUTION",
            scope="search",
            case_id=case_id,
            fold_id=fold_id,
        )
    trace.begin_operation(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id=f"ckpt-search-{case_id}",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id=case_id,
        fold_id=0,
    )
    trace.start(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id=f"ckpt-search-{case_id}",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id=case_id,
        fold_id=0,
    )
    trace.complete(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id=f"ckpt-search-{case_id}",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id=case_id,
        fold_id=0,
    )
    pipeline_kwargs = dict(
        kind=TraceKind.PIPELINE_INVOCATION,
        invocation_id=f"pipeline-{case_id}",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id=case_id,
        extra={"forward_kind": "CANDIDATE_EFFECT_FORWARD"},
    )
    trace.begin_operation(**pipeline_kwargs)
    trace.start(**pipeline_kwargs)
    trace.complete(**pipeline_kwargs)

    trace.mark_closure_frozen(closure_sha=closure_sha)
    for kind, phase in (
        (TraceKind.CRITIC_LOAD, "REEVAL_EFFECTS"),
        (TraceKind.CHECKPOINT_FORWARD, "REEVAL_EFFECTS"),
    ):
        kwargs = dict(
            kind=kind,
            invocation_id=f"fold2-{kind.value}-{case_id}",
            phase=phase,
            scope="selection_blind_reevaluation",
            case_id=case_id,
            fold_id=2,
        )
        trace.begin_operation(**kwargs)
        trace.start(**kwargs)
        trace.complete(**kwargs)

    return {
        "case_id": case_id,
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [{"disposition": "SELECTED_MATERIAL"}],
        "extra": {"closure": {"evaluation_closure_hash": closure_sha}},
    }


def _build_and_verify(tmp_path: Path, run_id: str, proposals: list, trace: GlobalForwardTrace):
    package = tmp_path / "quarantine" / run_id
    sidecar = tmp_path / "sidecars" / run_id
    package.mkdir(parents=True)
    sidecar.mkdir(parents=True)

    trace_summary = trace.summarize()
    summary = {
        "execution_scope": "TWO_EVENT_ONLY",
        "three_event_execution_status": "OUT_OF_SCOPE",
        "runtime_lock": {"policy": "EXACT_BYTE_DETERMINISTIC"},
        "trace_summary": {
            key: value for key, value in trace_summary.items() if key != "events"
        },
        "case_results": [],
    }
    stable = _stable()

    artifact_paths = {}
    for name, value in (
        ("runner_summary.json", summary),
        ("pre_stable_lock.json", stable),
        ("post_stable_lock.json", stable),
        ("trace_events.json", {"events": trace.events}),
    ):
        path = package / name
        _write(path, value)
        artifact_paths[name] = path
    for proposal in proposals:
        path = package / str(proposal["case_id"]) / "case_edit_proposal.json"
        _write(path, proposal)
        artifact_paths[f"{proposal['case_id']}/case_edit_proposal.json"] = path

    evidence_root = compute_evidence_package_root(artifact_paths, root=package)
    relpaths = {
        name: str(path.relative_to(package)) for name, path in artifact_paths.items()
    }
    _write(
        sidecar / "EVIDENCE_ROOT.json",
        {"evidence_root_sha": evidence_root, "artifact_relpaths": relpaths},
    )
    final_observation = {
        "artifact_kind": "CF1S_FINAL_RERUN_OBSERVATION_MANIFEST_V1",
        "qualification_node_id_manifest_sha256": "1" * 64,
        "final_rerun_test_log_sha256": "2" * 64,
        "final_rerun_node_id_manifest_sha256": "3" * 64,
        "final_rerun_test_log_relpath": "missing.log",
        "final_rerun_node_id_manifest_relpath": "missing.json",
        "executed_at": "2026-07-19T00:00:00Z",
        "node_identity": "node",
        "host_identity": "host",
    }
    final_observation["final_rerun_observation_manifest_sha256"] = canonical_json_sha256(
        final_observation
    )
    _write(sidecar / "final_rerun_observation_manifest.json", final_observation)

    trust = load_trust_root(TRUST)
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
        intended_final_destination=str(tmp_path / "final" / run_id),
        final_rerun_observation_manifest_sha256=final_observation[
            "final_rerun_observation_manifest_sha256"
        ],
    )
    return verify_cf1s_package(
        phase=PRE_PROMOTION,
        package_dir=package,
        expected_final_dir=tmp_path / "final" / run_id,
        sidecar_dir=sidecar,
        run_id=run_id,
        authorization_artifact=authorization,
        post_attestation=post,
        trust_root=trust,
        trust_root_sha256=sha256_file(TRUST),
        verifier_code_sha256=compute_verifier_code_sha256(),
    )


def _baseline_others(trace: GlobalForwardTrace, closure_sha: str) -> list:
    return [_selected_case(trace, "case-2", closure_sha), _selected_case(trace, "case-3", closure_sha)]


def test_a_delete_fold2_trace_from_selected_case_fails(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    subject = _selected_case(trace, "case-1", closure_sha)
    others = _baseline_others(trace, closure_sha)
    trace.events = [
        e for e in trace.events
        if not (e["case_id"] == "case-1" and e["kind"] in ("CRITIC_LOAD", "CHECKPOINT_FORWARD") and e.get("fold_id") == 2)
    ]
    verdict = _build_and_verify(tmp_path, "run-a", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "FAIL"


def test_b_delete_entire_case_trace_fails(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    subject = _selected_case(trace, "case-1", closure_sha)
    others = _baseline_others(trace, closure_sha)
    trace.events = [e for e in trace.events if e["case_id"] != "case-1"]
    verdict = _build_and_verify(tmp_path, "run-b", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "FAIL"


def test_c_not_constructible_zero_forward_passes(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    kwargs = dict(
        kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
        invocation_id="ground-case-1",
        phase="GROUNDING",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**kwargs)
    trace.start(**kwargs)
    trace.fail(
        kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
        invocation_id="ground-case-1",
        phase="GROUNDING",
        scope="search",
        case_id="case-1",
        fold_id=0,
        failure_code="NOT_CONSTRUCTIBLE",
        extra={"terminal_failure": True},
    )
    subject = {
        "case_id": "case-1",
        "construction_status": "NOT_CONSTRUCTIBLE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-c", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "PASS"


def test_d_not_constructible_without_failure_termination_fails(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    kwargs = dict(
        kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
        invocation_id="ground-case-1",
        phase="GROUNDING",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**kwargs, allow=False, denial_code="NOT_CONSTRUCTIBLE")
    subject = {
        "case_id": "case-1",
        "construction_status": "NOT_CONSTRUCTIBLE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-d", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "FAIL"


def test_e_not_evaluated_denied_then_side_effect_fails(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    trace.begin_operation(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id="gate-block-case-1",
        phase="SCOPE_GATE",
        scope="search",
        case_id="case-1",
        allow=False,
        denial_code="SCOPE_GATE_BLOCKED",
    )
    side_kwargs = dict(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id="side-effect-case-1",
        phase="SCOPE_GATE",
        scope="search",
        case_id="case-1",
    )
    trace.begin_operation(**side_kwargs)
    trace.start(**side_kwargs)
    trace.complete(**side_kwargs)
    subject = {
        "case_id": "case-1",
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": False,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-e", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "FAIL"


def test_f_borrowing_another_case_trace_cannot_satisfy_missing_case(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    others = _baseline_others(trace, closure_sha)
    subject = {
        "case_id": "case-1",
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [{"disposition": "SELECTED_MATERIAL"}],
        "extra": {"closure": {"evaluation_closure_hash": closure_sha}},
    }
    verdict = _build_and_verify(tmp_path, "run-f", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "FAIL"


def test_g_all_gate_failed_side_effect_before_block_fails(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    for fold_id in (0, 1):
        for kind in (TraceKind.ATTRIBUTION_MODEL_LOAD, TraceKind.ATTRIBUTION_FORWARD):
            kwargs = dict(
                kind=kind,
                invocation_id=f"{kind.value}-case-1-{fold_id}",
                phase="ATTRIBUTION",
                scope="search",
                case_id="case-1",
                fold_id=fold_id,
            )
            trace.begin_operation(**kwargs)
            trace.start(**kwargs)
            trace.complete(**kwargs)
    side_kwargs = dict(
        kind=TraceKind.PIPELINE_INVOCATION,
        invocation_id="pipeline-case-1",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id="case-1",
        extra={"forward_kind": "CANDIDATE_EFFECT_FORWARD"},
    )
    trace.begin_operation(**side_kwargs)
    trace.start(**side_kwargs)
    trace.complete(**side_kwargs)
    trace.begin_operation(
        kind=TraceKind.PIPELINE_INVOCATION,
        invocation_id="gate-block-case-1",
        phase="GATE",
        scope="search",
        case_id="case-1",
        allow=False,
        denial_code="GATE_FAILURE",
    )
    subject = {
        "case_id": "case-1",
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_SUPPORTED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-g", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "FAIL"


def test_g_all_gate_failed_blocked_before_side_effect_passes(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    for fold_id in (0, 1):
        for kind in (TraceKind.ATTRIBUTION_MODEL_LOAD, TraceKind.ATTRIBUTION_FORWARD):
            kwargs = dict(
                kind=kind,
                invocation_id=f"{kind.value}-case-1-{fold_id}",
                phase="ATTRIBUTION",
                scope="search",
                case_id="case-1",
                fold_id=fold_id,
            )
            trace.begin_operation(**kwargs)
            trace.start(**kwargs)
            trace.complete(**kwargs)
    trace.begin_operation(
        kind=TraceKind.PIPELINE_INVOCATION,
        invocation_id="gate-block-case-1",
        phase="GATE",
        scope="search",
        case_id="case-1",
        allow=False,
        denial_code="GATE_FAILURE",
    )
    subject = {
        "case_id": "case-1",
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_SUPPORTED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-g2", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "PASS"


def test_h_not_evaluable_partial_scope_exact_match_passes(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    kwargs = dict(
        kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
        invocation_id="attr-case-1",
        phase="ATTRIBUTION",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**kwargs)
    trace.start(**kwargs)
    trace.complete(**kwargs)
    fail_kwargs = dict(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id="ckpt-fail-case-1",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**fail_kwargs)
    trace.start(**fail_kwargs)
    trace.fail(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id="ckpt-fail-case-1",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id="case-1",
        fold_id=0,
        failure_code="IDENTITY_NOISE_FAILURE",
        extra={"terminal_failure": True},
    )
    subject = {
        "case_id": "case-1",
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_EVALUABLE",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-h1", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "PASS"


def test_i_gate_evidence_empty_ledger_exemption_scoped_to_not_constructible():
    """G4 (_gate_evidence)'s empty-ledger check must exempt NOT_CONSTRUCTIBLE
    cases only — an empty candidate_results ledger is the correct state for
    that disposition (real Validation20 cases hit this first), but must still
    FAIL for any other disposition with an empty ledger (e.g. NOT_EVALUABLE)."""
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_scientific import (
        derive_coverage_axes,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_verifier import (
        _gate_evidence,
    )

    not_constructible = {
        "case_id": "case-1",
        "construction_status": "NOT_CONSTRUCTIBLE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    case_results = [
        {
            "case_id": "case-1",
            "execution_complete": False,
            "selection_disposition": None,
            "scientifically_evaluable": False,
            "control_result": None,
        }
    ]
    summary = {"case_results": case_results, **derive_coverage_axes(case_results)}
    _gate_evidence([not_constructible], summary)  # must not raise

    not_evaluable = dict(not_constructible)
    not_evaluable["construction_status"] = "COMPLETE"
    not_evaluable["scientific_status"] = "NOT_EVALUABLE"
    with pytest.raises(Exception, match="candidate ledger empty"):
        _gate_evidence([not_evaluable], summary)


def test_h_not_evaluable_without_failure_termination_fails(tmp_path):
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    kwargs = dict(
        kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
        invocation_id="attr-case-1",
        phase="ATTRIBUTION",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**kwargs)
    trace.start(**kwargs)
    trace.complete(**kwargs)
    subject = {
        "case_id": "case-1",
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_EVALUABLE",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-h2", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "FAIL"


def test_j_all_excluded_full_search_no_denial_passes(tmp_path):
    """CONSTRUCTIBLE_ALL_EXCLUDED: search ran to completion (no gate denial
    anywhere), candidate scored, but excluded at selection — must PASS with
    nonzero candidate forward and no DENIED/FAILED terminal required."""
    trace = GlobalForwardTrace()
    closure_sha = "z" * 64
    kwargs = dict(
        kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
        invocation_id="attr-case-1",
        phase="ATTRIBUTION",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**kwargs)
    trace.start(**kwargs)
    trace.complete(**kwargs)
    fwd_kwargs = dict(
        kind=TraceKind.ATTRIBUTION_FORWARD,
        invocation_id="fwd-case-1",
        phase="ATTRIBUTION",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**fwd_kwargs)
    trace.start(**fwd_kwargs)
    trace.complete(**fwd_kwargs)
    ckpt_kwargs = dict(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id="ckpt-case-1",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id="case-1",
        fold_id=0,
    )
    trace.begin_operation(**ckpt_kwargs)
    trace.start(**ckpt_kwargs)
    trace.complete(**ckpt_kwargs)
    pipeline_kwargs = dict(
        kind=TraceKind.PIPELINE_INVOCATION,
        invocation_id="pipeline-case-1",
        phase="SEARCH_EFFECTS",
        scope="search",
        case_id="case-1",
        extra={"forward_kind": "CANDIDATE_EFFECT_FORWARD"},
    )
    trace.begin_operation(**pipeline_kwargs)
    trace.start(**pipeline_kwargs)
    trace.complete(**pipeline_kwargs)
    subject = {
        "case_id": "case-1",
        "construction_status": "COMPLETE",
        "scientific_status": "NOT_EVALUATED",
        "execution_evidence": {
            "production_input_verified": True,
            "pipeline_integrity_verified": True,
        },
        "candidate_results": [{"disposition": None, "kind": "PAIR"}],
    }
    others = _baseline_others(trace, closure_sha)
    verdict = _build_and_verify(tmp_path, "run-j", [subject] + others, trace)
    assert verdict["gate_results"]["G3"] == "PASS"
