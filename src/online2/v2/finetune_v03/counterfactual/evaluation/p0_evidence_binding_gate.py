"""P0 evidence-binding gate v2 — synthetic behavior validators + test-run verify-only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .reconstruction_q4_q5 import aggregate_q4_q5_from_per_mg
from .test_run_manifest import (
    COUNTERFACTUAL_M1_SUITE,
    sha256_path,
    verify_test_run_manifest,
)

GATE_VERSION = "p0_evidence_binding_gate_v2"
DEFAULT_TEST_RUN_MANIFEST_REL = "reports/p0_evidence_binding/test_run_manifest.json"
DEFAULT_GATE_REL = "reports/p0_evidence_binding_gate.json"
DEFAULT_STATUS_REL = "reports/p0_evidence_binding/STATUS.json"
FIXTURE_REL = "reports/_gate_fixture"


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(doc), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _row(*, tokens, case_id, event_id, timestamp, feature="t"):
    from .bank_integrity import SOURCE_PARTITION_TRAINING

    return {
        "feature": feature,
        "tokens": list(tokens),
        "roles": ["R"],
        "signature": "R",
        "case_id": case_id,
        "event_id": event_id,
        "measurement_group_id": "mg1",
        "timestamp": timestamp,
        "source_partition": SOURCE_PARTITION_TRAINING,
    }


def _validate_executed_query_contract() -> tuple[bool, Dict[str, Any], List[str]]:
    from ..candidates.bundle_bank_index import query_unique_bundles_from_bank
    from .bank_integrity import BANK_MODE_DEPLOYMENT, bank_query_manifest_sha256

    errs: List[str] = []
    a = bank_query_manifest_sha256(
        target_event_id="e",
        case_id="c",
        feature="f",
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=False,
        target_timestamp_utc="2020-01-01T00:00:00.000000Z",
    )
    b = bank_query_manifest_sha256(
        target_event_id="e",
        case_id="c",
        feature="f",
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=False,
        target_timestamp_utc="2025-12-31T00:00:00.000000Z",
    )
    timestamp_in_hash = a != b
    if not timestamp_in_hash:
        errs.append("timestamp_not_in_query_hash")
    noncanonical_rejected = False
    try:
        query_unique_bundles_from_bank(
            [],
            [],
            feature="f",
            mode=BANK_MODE_DEPLOYMENT,
            exclude_target_event=False,
        )
        errs.append("noncanonical_override_not_rejected")
    except ValueError:
        noncanonical_rejected = True
    result = {
        "timestamp_in_query_hash": timestamp_in_hash,
        "noncanonical_override_rejected": noncanonical_rejected,
        "errors": list(errs),
    }
    return len(errs) == 0, result, errs


def _run_synthetic_a8_1(fixture_root: Path) -> tuple[bool, Dict[str, Any], List[str]]:
    from ..candidates.bundle_bank_index import (
        build_bank_query_set_manifest,
        observations_and_uniques_from_rows,
        query_unique_bundles_from_bank,
        wrap_executed_query_record,
        write_two_tier_bank,
    )
    from .bank_integrity import (
        A8_1_REQUIRED_CONSUMER_RUN_IDS,
        BANK_MODE_DEPLOYMENT,
        BANK_MODE_RECONSTRUCTION_EVAL,
        evaluate_a8_1_bank_integrity,
    )

    errs: List[str] = []
    bank_dir = fixture_root / "bank"
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    rows = [
        _row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01"),
        _row(tokens=["B"], case_id="c2", event_id="e2", timestamp="2020-01-01"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(bank_dir, observations=obs, uniques=uniq)
    man["source_partition_assignment_complete"] = True
    man["training_problem_case_overlap_count"] = 0
    man_path = bank_dir / "mlm_bundle_bank_manifest.json"
    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    mode_by_run = {
        "functional_smoke": BANK_MODE_DEPLOYMENT,
        "curated_fixture": BANK_MODE_DEPLOYMENT,
        "natural_path_a": BANK_MODE_DEPLOYMENT,
        "reconstruction_selection": BANK_MODE_RECONSTRUCTION_EVAL,
        "reconstruction_eval": BANK_MODE_RECONSTRUCTION_EVAL,
    }
    query_records = []
    for rid, mode in mode_by_run.items():
        _b, executed = query_unique_bundles_from_bank(
            obs,
            uniq,
            feature="t",
            case_id="c1",
            target_event_id="e1",
            target_timestamp="2020-01-01",
            mode=mode,
            expected_roles=["R"],
        )
        if mode == BANK_MODE_RECONSTRUCTION_EVAL:
            qset = build_bank_query_set_manifest(
                run_id=rid,
                bank_manifest=man,
                mode=mode,
                executed_entries=[
                    {"mg_key": "c1|e1|mg1|t", "executed_query_manifest": executed}
                ],
            )
            qset["expected_mg_keys"] = ["c1|e1|mg1|t"]
            qset["query_set_manifest"] = {
                "run_id": qset["run_id"],
                "query_count": qset["query_count"],
                "query_records": qset["query_records"],
                "query_set_sha256": qset["query_set_sha256"],
            }
            query_records.append(qset)
        else:
            query_records.append(
                wrap_executed_query_record(
                    run_id=rid,
                    bank_manifest=man,
                    executed_query_manifest=executed,
                )
            )

    result = evaluate_a8_1_bank_integrity(
        bank_dir=bank_dir,
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        query_records=query_records,
        required_consumer_run_ids=A8_1_REQUIRED_CONSUMER_RUN_IDS,
    )
    if not result.get("a8_1_pass"):
        errs.extend(f"synthetic_a8_1:{e}" for e in (result.get("errors") or []))
    return bool(result.get("a8_1_pass")), result, errs


def _run_synthetic_partition(fixture_root: Path) -> tuple[bool, Dict[str, Any], List[str]]:
    from ..candidates.bundle_bank_index import (
        observations_and_uniques_from_rows,
        write_two_tier_bank,
    )
    from .bank_integrity import evaluate_a8_1_bank_integrity
    from .reconstruction_selection import MIN_RECONSTRUCTION_CANDIDATES
    from .bank_integrity import EXPECTED_POLICY_BY_MODE, BANK_MODE_DEPLOYMENT, BANK_MODE_RECONSTRUCTION_EVAL

    errs: List[str] = []
    dep = EXPECTED_POLICY_BY_MODE[BANK_MODE_DEPLOYMENT]
    rec = EXPECTED_POLICY_BY_MODE[BANK_MODE_RECONSTRUCTION_EVAL]
    if dep.get("allowed_problem_case_scope") != "TARGET_CASE_ONLY":
        errs.append("deployment_problem_scope")
    if rec.get("allowed_problem_case_scope") != "NONE":
        errs.append("reconstruction_problem_scope")
    if MIN_RECONSTRUCTION_CANDIDATES != 5:
        errs.append("min_reconstruction_candidates")

    missing_partition_rejected = False
    bad_rows = [
        {
            "feature": "t",
            "tokens": ["A"],
            "roles": ["R"],
            "signature": "R",
            "case_id": "c1",
            "event_id": "e1",
            "measurement_group_id": "mg1",
            "timestamp": "2020-01-01",
        }
    ]
    try:
        observations_and_uniques_from_rows(bad_rows)
        errs.append("missing_partition_not_rejected")
    except ValueError:
        missing_partition_rejected = True

    bank_dir = fixture_root / "bank_overlap"
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    rows = [_row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01")]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(bank_dir, observations=obs, uniques=uniq)
    man["source_partition_assignment_complete"] = True
    man["training_problem_case_overlap_count"] = 1
    (bank_dir / "mlm_bundle_bank_manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    overlap_result = evaluate_a8_1_bank_integrity(
        bank_dir=bank_dir,
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        query_records=[],
        required_consumer_run_ids=(),
    )
    if overlap_result.get("a8_1_pass"):
        errs.append("overlap_not_rejected_by_a8_1")
    elif "training_problem_case_overlap" not in (overlap_result.get("errors") or []):
        errs.append("overlap_error_missing")

    result = {
        "missing_partition_rejected": missing_partition_rejected,
        "overlap_a8_1_pass": bool(overlap_result.get("a8_1_pass")),
        "overlap_errors": list(overlap_result.get("errors") or []),
        "min_reconstruction_candidates": MIN_RECONSTRUCTION_CANDIDATES,
        "errors": list(errs),
    }
    return len(errs) == 0, result, errs


def _run_synthetic_a7(fixture_root: Path, root: Path) -> tuple[bool, Dict[str, Any], List[str]]:
    from .execution_provenance import (
        REQUIRED_RUN_IDS,
        build_run_record,
        evaluate_a7_provenance,
        file_sha256,
    )

    errs: List[str] = []
    imports_dir = fixture_root / "runtime_imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    bank_sha = "f" * 64
    runs = []
    by_run_imports: Dict[str, List[str]] = {}
    for i, rid in enumerate(REQUIRED_RUN_IDS):
        paths_list = [f"src/online2/v2/tokenizer.py"] if rid else []
        # pytest also needs nonempty imports for A7 with root
        if not paths_list:
            paths_list = ["src/online2/v2/tokenizer.py"]
        doc = {
            "run_id": rid,
            "runtime_imported_project_paths": paths_list,
        }
        art_name = f"runtime_imports_{rid}.json"
        art_path = imports_dir / art_name
        _write_json(art_path, doc)
        rel = str(art_path.relative_to(root.resolve()))
        arts = [{"relative_path": rel, "sha256": file_sha256(art_path)}]
        by_run_imports[rid] = paths_list
        runs.append(
            build_run_record(
                run_id=rid,
                entrypoint=f"scripts/{rid}.py",
                git_commit="abc",
                dependency_tree_sha256="dep",
                exit_code=0,
                pre_dependency_tree_sha256="dep",
                post_dependency_tree_sha256="dep",
                execution_attempt_id="gate_synth",
                orchestrator_invocation_id="gate_inv",
                final_lock_id="gate_lock",
                run_sequence=i + 1,
                log_sha256="b" * 64,
                artifacts=arts,
                config_sha256="cfg",
                stage_a_checkpoint_sha256="ckpt",
                vocab_sha256="vocab",
                bundle_bank_content_sha256=bank_sha
                if rid
                in {
                    "bundle_bank_build",
                    "functional_smoke",
                    "curated_fixture",
                    "natural_path_a",
                    "reconstruction_selection",
                    "reconstruction_eval",
                }
                else None,
                runtime_imported_project_paths=paths_list,
            )
        )
    provenance = {
        "active_execution_attempt_id": "gate_synth",
        "runs": runs,
        "runtime_imports_by_run": by_run_imports,
        "runtime_imported_project_paths_union": sorted(
            {p for ps in by_run_imports.values() for p in ps}
        ),
    }
    result = evaluate_a7_provenance(provenance, root=root.resolve())
    if not result.get("a7_pass"):
        errs.extend(f"synthetic_a7:{e}" for e in (result.get("errors") or []))
    return bool(result.get("a7_pass")), result, errs


def _run_synthetic_q4_q5() -> tuple[bool, Dict[str, Any], List[str]]:
    errs: List[str] = []
    rows = [
        {
            "metric_eligible": True,
            "candidate_count": 12,
            "scoring_completed": True,
            "original_rank": 1,
            "recall_at_3": 1,
            "random_recall_at_3": 0.2,
        },
        {
            "metric_eligible": False,
            "metric_ineligible_reason": "insufficient_candidates",
            "candidate_count": 2,
            "scoring_completed": False,
        },
        {
            "metric_eligible": True,
            "candidate_count": 8,
            "scoring_completed": True,
            "original_rank": 2,
            "recall_at_3": 0,
            "random_recall_at_3": 0.25,
        },
    ]
    result = aggregate_q4_q5_from_per_mg(rows)
    if result.get("reconstruction_metric_audit_status") != "PASS":
        errs.append("synthetic_q4_q5_audit_fail")
    if not result.get("q4_q5_evaluable"):
        errs.append("synthetic_q4_q5_not_evaluable")
    if result.get("missing_metric_eligibility_count", 1) != 0:
        errs.append("synthetic_q4_q5_missing_eligibility")

    missing = aggregate_q4_q5_from_per_mg([{"original_rank": 1, "recall_at_3": 1}])
    if missing.get("reconstruction_metric_audit_status") != "FAIL":
        errs.append("missing_metric_eligible_not_audit_fail")

    weak = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "original_rank": 1,
                "recall_at_3": 1,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    if weak.get("reconstruction_metric_audit_status") != "FAIL":
        errs.append("weak_metric_eligible_without_candidate_scoring_not_rejected")

    nonint = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 5,
                "scoring_completed": True,
                "original_rank": 1.5,
                "recall_at_3": 1,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    if nonint.get("reconstruction_metric_audit_status") != "FAIL":
        errs.append("noninteger_original_rank_not_rejected")

    return len(errs) == 0, result, errs


def _persist_synthetic_artifact(
    *,
    root: Path,
    relative_path: str,
    doc: Mapping[str, Any],
) -> Dict[str, str]:
    path = Path(root) / relative_path
    _write_json(path, doc)
    return {
        "path": relative_path,
        "sha256": sha256_path(path),
    }


def compute_p0_evidence_binding_gate(
    *,
    root: Path,
    test_run_manifest_path: Optional[Path] = None,
    pytest_log: Optional[Path] = None,
    expected_pytest_pass_count: int = 0,
) -> Dict[str, Any]:
    """Derive ready_for_final_locked_rerun from synthetic validators + test-run manifest."""
    del pytest_log, expected_pytest_pass_count
    root = Path(root).resolve()
    failure_reasons: List[str] = []

    fixture_root = root / FIXTURE_REL
    fixture_root.mkdir(parents=True, exist_ok=True)
    synth_dir_rel = f"{FIXTURE_REL}/synthetic"

    executed_ok, executed_result, executed_errs = _validate_executed_query_contract()
    failure_reasons.extend(executed_errs)

    a8_ok, a8_result, a8_errs = _run_synthetic_a8_1(fixture_root)
    failure_reasons.extend(a8_errs)

    part_ok, part_result, part_errs = _run_synthetic_partition(fixture_root)
    failure_reasons.extend(part_errs)

    a7_ok, a7_result, a7_errs = _run_synthetic_a7(fixture_root, root)
    failure_reasons.extend(a7_errs)

    q45_ok, q45_result, q45_errs = _run_synthetic_q4_q5()
    failure_reasons.extend(q45_errs)

    a8_art = _persist_synthetic_artifact(
        root=root,
        relative_path=f"{synth_dir_rel}/synthetic_a8_1_result.json",
        doc=a8_result,
    )
    a7_art = _persist_synthetic_artifact(
        root=root,
        relative_path=f"{synth_dir_rel}/synthetic_a7_result.json",
        doc=a7_result,
    )
    part_art = _persist_synthetic_artifact(
        root=root,
        relative_path=f"{synth_dir_rel}/synthetic_partition_result.json",
        doc=part_result,
    )
    q45_art = _persist_synthetic_artifact(
        root=root,
        relative_path=f"{synth_dir_rel}/synthetic_q4_q5_result.json",
        doc=q45_result,
    )
    exec_art = _persist_synthetic_artifact(
        root=root,
        relative_path=f"{synth_dir_rel}/executed_query_contract_result.json",
        doc=executed_result,
    )

    man_path = (
        Path(test_run_manifest_path)
        if test_run_manifest_path is not None
        else root / DEFAULT_TEST_RUN_MANIFEST_REL
    )
    full_test_run_verified = False
    dep_matches = False
    test_run_manifest_sha = None
    test_run_details: Dict[str, Any] = {}
    if not man_path.is_file():
        failure_reasons.append("test_run_manifest_missing")
    else:
        try:
            manifest = json.loads(man_path.read_text(encoding="utf-8"))
            test_run_manifest_sha = sha256_path(man_path)
            ok, man_errs, details = verify_test_run_manifest(root=root, manifest=manifest)
            test_run_details = details
            full_test_run_verified = ok
            dep_matches = bool(details.get("dependency_tree_matches_tested_source"))
            failure_reasons.extend(man_errs)
        except Exception as exc:
            failure_reasons.append(f"test_run_manifest_unreadable:{exc}")

    ready = (
        a8_ok
        and part_ok
        and a7_ok
        and q45_ok
        and executed_ok
        and full_test_run_verified
        and dep_matches
        and len(failure_reasons) == 0
    )

    return {
        "gate_version": GATE_VERSION,
        "synthetic_a8_1_behavior_validation_pass": a8_ok,
        "synthetic_partition_behavior_validation_pass": part_ok,
        "synthetic_runtime_import_behavior_validation_pass": a7_ok,
        "synthetic_q4_q5_behavior_validation_pass": q45_ok,
        "executed_query_contract_validation_pass": executed_ok,
        "full_test_run_verified": full_test_run_verified,
        "dependency_tree_matches_tested_source": dep_matches,
        "test_run_manifest_path": str(man_path),
        "test_run_manifest_sha256": test_run_manifest_sha,
        "synthetic_a8_1_result_path": a8_art["path"],
        "synthetic_a8_1_result_sha256": a8_art["sha256"],
        "synthetic_a7_result_path": a7_art["path"],
        "synthetic_a7_result_sha256": a7_art["sha256"],
        "synthetic_partition_result_path": part_art["path"],
        "synthetic_partition_result_sha256": part_art["sha256"],
        "synthetic_q4_q5_result_path": q45_art["path"],
        "synthetic_q4_q5_result_sha256": q45_art["sha256"],
        "executed_query_contract_result_path": exec_art["path"],
        "executed_query_contract_result_sha256": exec_art["sha256"],
        "test_run_details": test_run_details,
        "failure_reasons": failure_reasons,
        "actual_a8_1_acceptance_status": "NOT_RUN",
        "actual_a7_acceptance_status": "NOT_RUN",
        "acceptance_recovery_execution": "NOT_RUN",
        "mlm_implementation": "FAIL",
        "p1_readiness": "FAIL",
        "ready_for_final_locked_rerun": bool(ready),
    }


def status_from_gate(gate: Mapping[str, Any], *, authoritative_gate_sha256: Optional[str] = None) -> Dict[str, Any]:
    ready = bool(gate.get("ready_for_final_locked_rerun"))
    impl = "PASS" if ready else "PARTIAL_PASS"
    return {
        "remediation_contract_design": "PASS",
        "remediation_code_implementation": impl,
        "gate_evidence_implementation": impl,
        "remediation_unit_tests": "PASS"
        if gate.get("full_test_run_verified")
        else "NOT_VERIFIED",
        "acceptance_recovery_execution": "NOT_RUN",
        "ready_for_final_locked_rerun": ready,
        "mlm_implementation": "FAIL",
        "p1_readiness": "FAIL",
        "overall": "CONDITIONAL_APPROVE_FOR_LOCKED_RERUN"
        if ready
        else "CONDITIONAL_REJECT",
        "gate_version": gate.get("gate_version"),
        "authoritative_gate_path": DEFAULT_GATE_REL,
        "authoritative_gate_sha256": authoritative_gate_sha256
        or gate.get("authoritative_gate_sha256"),
        "test_run_manifest_sha256": gate.get("test_run_manifest_sha256"),
        "source": DEFAULT_GATE_REL,
    }


def write_p0_evidence_binding_gate(
    out_path: Path,
    *,
    root: Path,
    test_run_manifest_path: Optional[Path] = None,
    pytest_log: Optional[Path] = None,
    expected_pytest_pass_count: int = 0,
    status_path: Optional[Path] = None,
) -> Dict[str, Any]:
    gate = compute_p0_evidence_binding_gate(
        root=root,
        test_run_manifest_path=test_run_manifest_path,
        pytest_log=pytest_log,
        expected_pytest_pass_count=expected_pytest_pass_count,
    )
    out_path = Path(out_path)
    _write_json(out_path, gate)
    gate_sha = sha256_path(out_path)
    if status_path is not None:
        _write_json(
            Path(status_path),
            status_from_gate(gate, authoritative_gate_sha256=gate_sha),
        )
    gate = dict(gate)
    gate["authoritative_gate_sha256"] = gate_sha
    return gate
