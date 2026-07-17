"""P0 evidence-binding behavioral regression tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    build_bank_query_record,
    build_bank_query_set_manifest,
    observations_and_uniques_from_rows,
    query_unique_bundles_from_bank,
    wrap_executed_query_record,
    write_two_tier_bank,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    BANK_MODE_RECONSTRUCTION_EVAL,
    EXPECTED_POLICY_BY_MODE,
    SOURCE_PARTITION_TRAINING,
    bank_query_manifest_sha256,
    evaluate_a8_1_bank_integrity,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.p0_evidence_binding_gate import (
    compute_p0_evidence_binding_gate,
    status_from_gate,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_q4_q5 import (
    aggregate_q4_q5_from_per_mg,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.test_run_manifest import (
    build_test_run_manifest,
    current_dependency_tree_sha256,
    fixed_pytest_command,
    verify_test_run_manifest,
    write_test_run_manifest,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import (
    P1_ENTRYPOINT_PATHS,
    collect_declared_dependency_paths,
    compute_dependency_tree_sha256,
)


def _row(*, tokens, case_id, event_id, timestamp, partition=SOURCE_PARTITION_TRAINING, feature="t"):
    return {
        "feature": feature,
        "tokens": list(tokens),
        "roles": ["R"],
        "signature": "R",
        "case_id": case_id,
        "event_id": event_id,
        "measurement_group_id": "mg1",
        "timestamp": timestamp,
        "source_partition": partition,
    }


def _write_minimal_junit(
    path: Path, *, tests: int = 2, failures: int = 0, errors: int = 0, skipped: int = 0
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            f'<testsuite name="pytest" tests="{tests}" failures="{failures}" '
            f'errors="{errors}" skipped="{skipped}"></testsuite>\n'
        ),
        encoding="utf-8",
    )


def _write_passing_test_run_manifest(root: Path, *, dep_sha: str | None = None) -> Path:
    root = Path(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log_path = reports / "pytest_counterfactual_m1.log"
    junit_path = reports / "pytest_counterfactual_m1.xml"
    log_path.write_text("ok\n", encoding="utf-8")
    _write_minimal_junit(junit_path, tests=2, failures=0, errors=0, skipped=0)
    sha = dep_sha if dep_sha is not None else current_dependency_tree_sha256(root)
    cmd = fixed_pytest_command(
        python_executable=sys.executable,
        junit_xml_absolute=junit_path.resolve(),
    )
    man = build_test_run_manifest(
        root=root,
        test_command=cmd,
        pre_dependency_tree_sha256=sha,
        post_dependency_tree_sha256=sha,
        exit_code=0,
        pytest_log_path=log_path,
        junit_xml_path=junit_path,
    )
    out = reports / "p0_evidence_binding" / "test_run_manifest.json"
    write_test_run_manifest(out, man)
    return out


def test_query_manifest_hash_changes_when_target_timestamp_changes():
    a = bank_query_manifest_sha256(
        target_event_id="e",
        case_id="c",
        feature="t",
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        target_timestamp_utc="2020-01-01T00:00:00.000000Z",
    )
    b = bank_query_manifest_sha256(
        target_event_id="e",
        case_id="c",
        feature="t",
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        target_timestamp_utc="2025-12-31T00:00:00.000000Z",
    )
    assert a != b


def test_query_set_hash_changes_when_any_target_timestamp_changes(tmp_path: Path):
    rows = [
        _row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01"),
        _row(tokens=["B"], case_id="c2", event_id="e2", timestamp="2020-01-01"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path / "bank", observations=obs, uniques=uniq)
    entries_a = []
    entries_b = []
    for row, ts_a, ts_b in [
        (rows[0], "2020-01-01", "2020-01-01"),
        (rows[1], "2020-06-01", "2025-12-31"),
    ]:
        _b1, ex_a = query_unique_bundles_from_bank(
            obs,
            uniq,
            feature="t",
            case_id=row["case_id"],
            target_event_id=row["event_id"],
            target_timestamp=ts_a,
            mode=BANK_MODE_RECONSTRUCTION_EVAL,
            expected_roles=["R"],
        )
        _b2, ex_b = query_unique_bundles_from_bank(
            obs,
            uniq,
            feature="t",
            case_id=row["case_id"],
            target_event_id=row["event_id"],
            target_timestamp=ts_b,
            mode=BANK_MODE_RECONSTRUCTION_EVAL,
            expected_roles=["R"],
        )
        k = f"{row['case_id']}|{row['event_id']}|mg1|t"
        entries_a.append({"mg_key": k, "executed_query_manifest": ex_a})
        entries_b.append({"mg_key": k, "executed_query_manifest": ex_b})
    qa = build_bank_query_set_manifest(
        run_id="reconstruction_eval",
        bank_manifest=man,
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        executed_entries=entries_a,
    )
    qb = build_bank_query_set_manifest(
        run_id="reconstruction_eval",
        bank_manifest=man,
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        executed_entries=entries_b,
    )
    assert qa["query_set_sha256"] != qb["query_set_sha256"]


def test_a8_1_rejects_wrong_target_timestamp(tmp_path: Path):
    rows = [_row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01")]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path / "bank", observations=obs, uniques=uniq)
    man["source_partition_assignment_complete"] = True
    man["training_problem_case_overlap_count"] = 0
    (tmp_path / "bank" / "mlm_bundle_bank_manifest.json").write_text(
        json.dumps(man, indent=2) + "\n"
    )
    _b, executed = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="c1",
        target_event_id="e1",
        target_timestamp="2020-01-01",
        mode=BANK_MODE_DEPLOYMENT,
        expected_roles=["R"],
    )
    rec = wrap_executed_query_record(
        run_id="functional_smoke",
        bank_manifest=man,
        executed_query_manifest=executed,
    )
    rec["query_manifest"]["target_timestamp_utc"] = "2099-01-01T00:00:00.000000Z"
    result = evaluate_a8_1_bank_integrity(
        bank_dir=tmp_path / "bank",
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        query_records=[rec],
        required_consumer_run_ids=("functional_smoke",),
    )
    assert result["a8_1_pass"] is False
    assert any("query_manifest_sha_mismatch" in e for e in result["errors"])


def test_query_rejects_noncanonical_policy_override():
    with pytest.raises(ValueError, match="noncanonical_policy_override"):
        query_unique_bundles_from_bank(
            [],
            [],
            feature="t",
            mode=BANK_MODE_DEPLOYMENT,
            exclude_target_event=False,
        )


def test_recorded_query_manifest_equals_executed_query_manifest(tmp_path: Path):
    rows = [_row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-02")]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path / "bank", observations=obs, uniques=uniq)
    bundles, executed = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="c1",
        target_event_id="e1",
        target_timestamp=pd.Timestamp("2020-01-02"),
        mode=BANK_MODE_DEPLOYMENT,
        expected_roles=["R"],
    )
    del bundles
    rec = wrap_executed_query_record(
        run_id="functional_smoke",
        bank_manifest=man,
        executed_query_manifest=executed,
    )
    clean = {k: v for k, v in executed.items() if k != "query_manifest_sha256"}
    assert rec["query_manifest"] == clean
    assert rec["bundle_bank_query_manifest_sha256"] == executed["query_manifest_sha256"]
    assert "target_timestamp_utc" in rec["query_manifest"]


def test_natural_path_config_cannot_disable_target_leaveout():
    src = Path("src/online2/v2/finetune_v03/counterfactual/candidates/mlm_path_a.py").read_text()
    assert "exclude_target_event=exclude_target" not in src
    assert 'mlm_cfg.get("exclude_target_event"' not in src
    assert "wrap_executed_query_record" in src


def test_missing_source_partition_fails_observations_build():
    rows = [
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
    with pytest.raises(ValueError, match="invalid_or_missing_source_partition"):
        observations_and_uniques_from_rows(rows)


def test_allowed_problem_case_scope_in_expected_policy():
    assert (
        EXPECTED_POLICY_BY_MODE[BANK_MODE_DEPLOYMENT]["allowed_problem_case_scope"]
        == "TARGET_CASE_ONLY"
    )
    assert (
        EXPECTED_POLICY_BY_MODE[BANK_MODE_RECONSTRUCTION_EVAL]["allowed_problem_case_scope"]
        == "NONE"
    )


def test_problem_scope_bound_into_policy_hash():
    from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
        bank_mode_policy_sha256,
    )

    a = bank_mode_policy_sha256(
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=False,
        allowed_problem_case_scope="TARGET_CASE_ONLY",
    )
    b = bank_mode_policy_sha256(
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=False,
        allowed_problem_case_scope="NONE",
    )
    assert a != b


def test_gate_ready_false_without_pytest_log(tmp_path: Path):
    gate = compute_p0_evidence_binding_gate(root=tmp_path, test_run_manifest_path=None)
    assert gate["executed_query_contract_validation_pass"] is True
    assert gate["full_test_run_verified"] is False
    assert gate["ready_for_final_locked_rerun"] is False
    assert "test_run_manifest_missing" in gate["failure_reasons"]


def test_gate_fails_without_runtime_import_artifacts(tmp_path: Path):
    gate = compute_p0_evidence_binding_gate(root=tmp_path)
    assert gate["ready_for_final_locked_rerun"] is False
    assert gate["acceptance_recovery_execution"] == "NOT_RUN"


def test_gate_fails_without_a8_query_set_pass(tmp_path: Path):
    gate = compute_p0_evidence_binding_gate(root=tmp_path)
    assert gate["actual_a8_1_acceptance_status"] == "NOT_RUN"
    assert gate["ready_for_final_locked_rerun"] is False
    assert "a8_1_query_set_pass" not in gate


def test_gate_source_string_alone_does_not_pass(tmp_path: Path):
    scripts = tmp_path / "scripts" / "online2_v2" / "v03"
    scripts.mkdir(parents=True)
    (scripts / "run_p1_acceptance_recovery_v03.py").write_text(
        "runtime_imports_\n", encoding="utf-8"
    )
    (scripts / "run_mlm_reconstruction_eval_v03.py").write_text(
        "metric_eligible\n", encoding="utf-8"
    )
    gate = compute_p0_evidence_binding_gate(root=tmp_path)
    assert gate["ready_for_final_locked_rerun"] is False
    assert gate["full_test_run_verified"] is False


def test_gate_uses_root_not_cwd(tmp_path: Path, monkeypatch):
    other = tmp_path / "other_cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    gate = compute_p0_evidence_binding_gate(root=tmp_path)
    assert gate["executed_query_contract_validation_pass"] is True
    assert gate["synthetic_partition_behavior_validation_pass"] is True
    assert isinstance(gate["ready_for_final_locked_rerun"], bool)


def test_build_bank_query_record_requires_executed_manifest(tmp_path: Path):
    del tmp_path
    with pytest.raises(ValueError, match="executed_query_manifest_required"):
        build_bank_query_record(
            run_id="functional_smoke",
            bank_manifest={"bundle_bank_content_sha256": "a" * 64},
            mode=BANK_MODE_DEPLOYMENT,
            target_event_id="e1",
            case_id="c1",
            feature="t",
            target_timestamp="2020-01-01",
        )


def test_gate_rejects_old_test_log_from_different_dependency_tree(tmp_path: Path):
    man_path = _write_passing_test_run_manifest(tmp_path, dep_sha="deadbeef" * 8)
    gate = compute_p0_evidence_binding_gate(
        root=tmp_path, test_run_manifest_path=man_path
    )
    assert gate["full_test_run_verified"] is False
    assert gate["dependency_tree_matches_tested_source"] is False
    assert "dependency_tree_does_not_match_tested_source" in gate["failure_reasons"]
    assert gate["ready_for_final_locked_rerun"] is False


def test_gate_rejects_pre_post_test_source_hash_mismatch(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    log_path = reports / "pytest_counterfactual_m1.log"
    junit_path = reports / "pytest_counterfactual_m1.xml"
    log_path.write_text("ok\n", encoding="utf-8")
    _write_minimal_junit(junit_path)
    current = current_dependency_tree_sha256(tmp_path)
    man = build_test_run_manifest(
        root=tmp_path,
        test_command=["python", "-m", "pytest", "tests/v2/counterfactual_m1/", "-q"],
        pre_dependency_tree_sha256=current,
        post_dependency_tree_sha256="ffff" * 16,
        exit_code=0,
        pytest_log_path=log_path,
        junit_xml_path=junit_path,
    )
    man_path = reports / "p0_evidence_binding" / "test_run_manifest.json"
    write_test_run_manifest(man_path, man)
    gate = compute_p0_evidence_binding_gate(
        root=tmp_path, test_run_manifest_path=man_path
    )
    assert "test_run_pre_post_dependency_mismatch" in gate["failure_reasons"]
    assert gate["ready_for_final_locked_rerun"] is False


def test_gate_rejects_tampered_junit_xml(tmp_path: Path):
    man_path = _write_passing_test_run_manifest(tmp_path)
    junit = tmp_path / "reports" / "pytest_counterfactual_m1.xml"
    junit.write_text(
        '<?xml version="1.0"?><testsuite tests="99" failures="0" errors="0" skipped="0"/>\n',
        encoding="utf-8",
    )
    gate = compute_p0_evidence_binding_gate(
        root=tmp_path, test_run_manifest_path=man_path
    )
    assert "junit_xml_sha_mismatch" in gate["failure_reasons"]
    assert gate["ready_for_final_locked_rerun"] is False


def test_gate_rejects_missing_metric_eligible_field():
    out = aggregate_q4_q5_from_per_mg(
        [{"original_rank": 1, "recall_at_3": 1.0, "random_recall_at_3": 0.1}]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert out["missing_metric_eligibility_count"] == 1
    assert out["q4_q5_evaluable"] is False
    assert out["mean_recall_at_3"] is None


def test_gate_rejects_metric_eligible_row_missing_rank():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 12,
                "scoring_completed": True,
                "recall_at_3": 1,
                "random_recall_at_3": 0.1,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert any("invalid_original_rank" in e for e in out["audit_errors"])


def test_gate_rejects_zero_metric_eligible_rows():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": False,
                "metric_ineligible_reason": "insufficient_candidates",
            }
        ]
    )
    assert out["q4_q5_evaluable"] is False
    assert out["q4_q5_status"] == "NOT_EVALUATED"
    assert out["metric_eligible_mg_count"] == 0


def test_synthetic_gate_does_not_mark_a7_or_a8_as_executed(tmp_path: Path):
    man_path = _write_passing_test_run_manifest(tmp_path)
    gate = compute_p0_evidence_binding_gate(
        root=tmp_path, test_run_manifest_path=man_path
    )
    assert "synthetic_a8_1_behavior_validation_pass" in gate
    assert "synthetic_runtime_import_behavior_validation_pass" in gate
    assert gate["acceptance_recovery_execution"] == "NOT_RUN"
    assert gate["actual_a8_1_acceptance_status"] == "NOT_RUN"
    assert gate["actual_a7_acceptance_status"] == "NOT_RUN"
    assert gate["mlm_implementation"] == "FAIL"
    assert gate["p1_readiness"] == "FAIL"
    assert gate.get("a7_pass") is None or gate.get("a7_pass") is False
    assert (tmp_path / gate["synthetic_a8_1_result_path"]).is_file()
    status = status_from_gate(gate)
    assert status["acceptance_recovery_execution"] == "NOT_RUN"
    assert status["mlm_implementation"] == "FAIL"
    assert status["p1_readiness"] == "FAIL"


def test_gate_test_wrapper_is_in_dependency_scope():
    ep = "scripts/online2_v2/v03/run_p0_evidence_binding_tests_v03.py"
    assert ep in P1_ENTRYPOINT_PATHS
    root = Path(".").resolve()
    declared = collect_declared_dependency_paths(root)
    assert ep in declared


def test_wrapper_change_invalidates_tested_dependency_hash(tmp_path: Path):
    root = tmp_path
    ep_rel = "scripts/online2_v2/v03/run_p0_evidence_binding_tests_v03.py"
    ep = root / ep_rel
    ep.parent.mkdir(parents=True)
    ep.write_text("print('A')\n", encoding="utf-8")
    # Minimal other entrypoints so collect doesn't fail oddly
    for other in P1_ENTRYPOINT_PATHS:
        if other == ep_rel or other.endswith(".yaml"):
            continue
        p = root / other
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("# stub\n", encoding="utf-8")
    conf = root / "conf/m1/cf_m2_prereq_smoke.yaml"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text("x: 1\n", encoding="utf-8")
    sha_a, _ = compute_dependency_tree_sha256(
        root, relative_paths=collect_declared_dependency_paths(root)
    )
    ep.write_text("print('B')\n", encoding="utf-8")
    sha_b, _ = compute_dependency_tree_sha256(
        root, relative_paths=collect_declared_dependency_paths(root)
    )
    assert sha_a != sha_b


def test_gate_rejects_filtered_pytest_command_with_k(tmp_path: Path):
    man_path = _write_passing_test_run_manifest(tmp_path)
    man = json.loads(man_path.read_text())
    man["test_command"] = list(man["test_command"]) + ["-k", "test_one"]
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    ok, errs, _ = verify_test_run_manifest(root=tmp_path, manifest=man)
    assert ok is False
    assert any("forbidden_flags" in e for e in errs)


def test_gate_rejects_collect_only_command(tmp_path: Path):
    man_path = _write_passing_test_run_manifest(tmp_path)
    man = json.loads(man_path.read_text())
    man["test_command"] = list(man["test_command"]) + ["--collect-only"]
    ok, errs, _ = verify_test_run_manifest(root=tmp_path, manifest=man)
    assert ok is False
    assert any("forbidden_flags" in e for e in errs)


def test_manifest_records_exact_subprocess_argv(tmp_path: Path):
    junit = tmp_path / "reports" / "pytest_counterfactual_m1.xml"
    junit.parent.mkdir(parents=True)
    _write_minimal_junit(junit)
    cmd = fixed_pytest_command(
        python_executable=sys.executable, junit_xml_absolute=junit.resolve()
    )
    log = tmp_path / "reports" / "pytest_counterfactual_m1.log"
    log.write_text("ok\n")
    man = build_test_run_manifest(
        root=tmp_path,
        test_command=cmd,
        pre_dependency_tree_sha256="a" * 64,
        post_dependency_tree_sha256="a" * 64,
        exit_code=0,
        pytest_log_path=log,
        junit_xml_path=junit,
    )
    assert man["test_command"] == cmd
    assert man["test_command"][0] == sys.executable
    assert man["test_command"][-1].startswith("--junitxml=")
    assert Path(man["test_command"][-1].split("=", 1)[1]).is_absolute()


def test_metric_eligible_requires_minimum_candidate_count():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 3,
                "scoring_completed": True,
                "original_rank": 1,
                "recall_at_3": 1,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert any("insufficient_candidate_count" in e for e in out["audit_errors"])


def test_metric_eligible_requires_scoring_completed():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 12,
                "scoring_completed": False,
                "original_rank": 1,
                "recall_at_3": 1,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert any("scoring_not_completed" in e for e in out["audit_errors"])


def test_metric_eligible_rejects_invalid_rank():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 12,
                "scoring_completed": True,
                "original_rank": 0,
                "recall_at_3": 1,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert any("invalid_original_rank" in e for e in out["audit_errors"])


def test_metric_eligible_rejects_zero_random_baseline():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 12,
                "scoring_completed": True,
                "original_rank": 1,
                "recall_at_3": 1,
                "random_recall_at_3": 0.0,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert any("invalid_random_recall_at_3" in e for e in out["audit_errors"])


def test_metric_eligible_rejects_noninteger_candidate_count():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 5.9,
                "scoring_completed": True,
                "original_rank": 1,
                "recall_at_3": 1,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert out["mean_recall_at_3"] is None
    assert any("invalid_candidate_count_type" in e for e in out["audit_errors"])


def test_metric_eligible_rejects_noninteger_original_rank():
    out = aggregate_q4_q5_from_per_mg(
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
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert out["q4_q5_evaluable"] is False
    assert any("invalid_original_rank_type" in e for e in out["audit_errors"])


def test_metric_eligible_rejects_noninteger_recall_at_3():
    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 5,
                "scoring_completed": True,
                "original_rank": 1,
                "recall_at_3": 1.5,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"
    assert any("invalid_recall_at_3" in e for e in out["audit_errors"])


def test_metric_eligible_rejects_boolean_rank_or_recall():
    out_rank = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 5,
                "scoring_completed": True,
                "original_rank": True,
                "recall_at_3": 1,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    assert out_rank["reconstruction_metric_audit_status"] == "FAIL"
    assert any("invalid_original_rank_type" in e for e in out_rank["audit_errors"])

    out_recall = aggregate_q4_q5_from_per_mg(
        [
            {
                "metric_eligible": True,
                "candidate_count": 5,
                "scoring_completed": True,
                "original_rank": 1,
                "recall_at_3": False,
                "random_recall_at_3": 0.2,
            }
        ]
    )
    assert out_recall["reconstruction_metric_audit_status"] == "FAIL"
    assert any("invalid_recall_at_3" in e for e in out_recall["audit_errors"])
