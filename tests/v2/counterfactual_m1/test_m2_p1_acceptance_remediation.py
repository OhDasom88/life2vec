"""P1 acceptance remediation unit tests (source lock, funnel, bank, provenance, A8-2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.evaluation.acceptance import (
    evaluate_p1_acceptance_conditions,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    BANK_MODE_RECONSTRUCTION_EVAL,
    bank_mode_policy_sha256,
    bundle_bank_content_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    evaluate_a7_provenance,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.funnel_accounting import (
    SELECTION_PRUNED,
    SELECTION_TOPK,
    TERMINAL_FAILED,
    TERMINAL_VALID,
    aggregate_topk_funnel,
    evaluate_a5_contracts,
    validate_funnel_conservation,
    validate_funnel_monotonicity,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    exact_match_selection_metrics,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import (
    P1_ENTRYPOINT_PATHS,
    SCOPE_VERSION_V2,
    collect_declared_dependency_paths,
    compute_dependency_tree_sha256,
    build_source_lock_v2,
)


def test_source_lock_includes_all_execution_entrypoints(tmp_path: Path):
    # Create stub entrypoints + deps under tmp root
    for ep in P1_ENTRYPOINT_PATHS:
        p = tmp_path / ep
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n", encoding="utf-8")
    dep = tmp_path / "src/online2/v2/finetune_v03/counterfactual/evaluation/source_lock.py"
    dep.parent.mkdir(parents=True, exist_ok=True)
    dep.write_text("# dep\n", encoding="utf-8")
    declared = collect_declared_dependency_paths(tmp_path)
    for ep in P1_ENTRYPOINT_PATHS:
        assert ep in declared


def test_source_lock_rejects_dirty_runtime_dependency(tmp_path: Path, monkeypatch):
    for ep in P1_ENTRYPOINT_PATHS:
        p = tmp_path / ep
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n", encoding="utf-8")
    dep = tmp_path / "src/online2/v2/finetune_v03/counterfactual/foo.py"
    dep.parent.mkdir(parents=True, exist_ok=True)
    dep.write_text("x=1\n", encoding="utf-8")

    import src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock as sl

    def fake_git(cwd, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("status", "--porcelain"):
            return " M src/online2/v2/finetune_v03/counterfactual/foo.py"
        return ""

    monkeypatch.setattr(sl, "_git", fake_git)
    lock = sl.build_source_lock_v2(
        root=tmp_path,
        config_path=tmp_path / "conf/m1/cf_m2_prereq_smoke.yaml",
        config_sha256="deadbeef",
    )
    assert lock["cf_source_clean"] is False
    assert lock["a6_pass"] is False
    assert lock["final_code_source_lock"] == "FAIL"


def test_dependency_tree_hash_matches_all_imported_runtime_modules(tmp_path: Path):
    files = [
        "scripts/online2_v2/v03/run_p1_finalize_pack_v03.py",
        "src/online2/v2/tokenizer.py",
    ]
    for rel in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content:{rel}\n", encoding="utf-8")
    sha1, man1 = compute_dependency_tree_sha256(tmp_path, relative_paths=files)
    sha2, man2 = compute_dependency_tree_sha256(tmp_path, relative_paths=list(reversed(files)))
    assert sha1 == sha2
    assert man1["scope_version"] == SCOPE_VERSION_V2
    assert [f["relative_path"] for f in man1["files"]] == sorted(files)


def test_source_lock_ignores_dirty_outside_declared_scope(tmp_path: Path, monkeypatch):
    for ep in P1_ENTRYPOINT_PATHS:
        p = tmp_path / ep
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# stub\n", encoding="utf-8")
    (tmp_path / "src/online2/v2/finetune_v03/counterfactual").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/online2/v2/finetune_v03/counterfactual/x.py").write_text("1\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/notes.md").write_text("dirty\n")

    import src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock as sl

    def fake_git(cwd, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("status", "--porcelain"):
            return " M docs/notes.md"
        return ""

    monkeypatch.setattr(sl, "_git", fake_git)
    lock = sl.build_source_lock_v2(
        root=tmp_path,
        config_path=tmp_path / "conf/m1/cf_m2_prereq_smoke.yaml",
        config_sha256="deadbeef",
    )
    assert lock["repository_dirty_outside_scope"] is True
    assert lock["cf_source_clean"] is True
    assert lock["a6_pass"] is True


def test_topk_candidates_partition_into_valid_or_single_terminal_failure():
    records = [
        {
            "candidate_id": "a",
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_VALID,
            "terminal_failure_reason": None,
        },
        {
            "candidate_id": "b",
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_FAILED,
            "terminal_failure_reason": "RETOKENIZATION_MISMATCH",
        },
    ]
    funnel = aggregate_topk_funnel(
        n_bank_unique=10, n_scored_unique=5, topk_records=records, n_not_selected_topk=3
    )
    assert funnel["n_topk"] == 2
    assert funnel["n_material_search_valid"] == 1
    assert funnel["n_gate4_terminal_fail"] == 1


def test_non_topk_candidates_are_pruned_not_failed():
    records = [
        {
            "candidate_id": "p",
            "selection_status": SELECTION_PRUNED,
            "terminal_status": None,
            "terminal_failure_reason": None,
        }
    ]
    a5 = evaluate_a5_contracts(
        lift_meta={"modified_channels": ["token_id"], "unchanged_auxiliary_channels": True,
                   "unchanged_attention_mask": True, "unchanged_non_target_positions": True},
        funnel={"n_topk": 0, "n_inversion_ok": 0, "n_inversion_terminal_fail": 0,
                "n_gate4_pass": 0, "n_gate4_terminal_fail": 0,
                "n_stage_a_ok": 0, "n_stage_a_terminal_fail": 0,
                "n_critic_scored": 0, "n_critic_execution_terminal_fail": 0,
                "n_material_search_valid": 0, "n_non_material_terminal_fail": 0,
                "n_other_postcritic_terminal_fail": 0,
                "n_bank_unique": 1, "n_scored_unique": 1},
        all_candidate_records=records,
        require_full_conservation=True,
    )
    assert a5["checks"]["non_topk_pruning_only"] is True


def test_funnel_stage_counts_satisfy_conservation_equations():
    funnel = {
        "n_topk": 3,
        "n_inversion_ok": 2,
        "n_gate4_pass": 1,
        "n_stage_a_ok": 1,
        "n_critic_scored": 1,
        "n_material_search_valid": 0,
        "n_inversion_terminal_fail": 1,
        "n_gate4_terminal_fail": 1,
        "n_stage_a_terminal_fail": 0,
        "n_critic_execution_terminal_fail": 0,
        "n_non_material_terminal_fail": 1,
        "n_other_postcritic_terminal_fail": 0,
        "n_bank_unique": 10,
        "n_scored_unique": 5,
    }
    ok, errs = validate_funnel_conservation(funnel)
    assert ok, errs
    mono_ok, mono_errs = validate_funnel_monotonicity(funnel)
    assert mono_ok, mono_errs


def test_gate4_pass_partitions_into_stage_a_ok_or_stage_a_failure():
    records = [
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_FAILED,
            "terminal_failure_reason": "STAGE_A_REENCODE_FAILURE",
        },
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_VALID,
            "terminal_failure_reason": None,
        },
    ]
    funnel = aggregate_topk_funnel(n_bank_unique=2, n_scored_unique=2, topk_records=records)
    assert funnel["n_gate4_pass"] == 2
    assert funnel["n_stage_a_ok"] + funnel["n_stage_a_terminal_fail"] == funnel["n_gate4_pass"]


def test_stage_a_ok_partitions_into_critic_scored_or_critic_execution_failure():
    records = [
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_FAILED,
            "terminal_failure_reason": "CRITIC_EXECUTION_FAILURE",
        },
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_FAILED,
            "terminal_failure_reason": "INVALID_CRITIC_SCORE",
        },
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_VALID,
            "terminal_failure_reason": None,
        },
    ]
    funnel = aggregate_topk_funnel(n_bank_unique=3, n_scored_unique=3, topk_records=records)
    assert funnel["n_stage_a_ok"] == 3
    assert (
        funnel["n_critic_scored"] + funnel["n_critic_execution_terminal_fail"]
        == funnel["n_stage_a_ok"]
    )


def test_terminal_failure_count_sum_matches_topk_minus_valid():
    records = [
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_VALID,
            "terminal_failure_reason": None,
        },
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_FAILED,
            "terminal_failure_reason": "NON_MATERIAL_SEARCH_DELTA",
        },
        {
            "selection_status": SELECTION_TOPK,
            "terminal_status": TERMINAL_FAILED,
            "terminal_failure_reason": "RETOKENIZATION_MISMATCH",
        },
    ]
    funnel = aggregate_topk_funnel(n_bank_unique=3, n_scored_unique=3, topk_records=records)
    fails = (
        funnel["n_inversion_terminal_fail"]
        + funnel["n_gate4_terminal_fail"]
        + funnel["n_stage_a_terminal_fail"]
        + funnel["n_critic_execution_terminal_fail"]
        + funnel["n_non_material_terminal_fail"]
        + funnel["n_other_postcritic_terminal_fail"]
    )
    assert funnel["n_topk"] - funnel["n_material_search_valid"] == fails


def test_funnel_counts_are_monotonic():
    funnel = {
        "n_bank_unique": 10,
        "n_scored_unique": 8,
        "n_topk": 3,
        "n_inversion_ok": 2,
        "n_gate4_pass": 2,
        "n_stage_a_ok": 1,
        "n_critic_scored": 1,
        "n_material_search_valid": 0,
    }
    ok, errs = validate_funnel_monotonicity(funnel)
    assert ok, errs


def test_bank_content_hash_stable_across_parquet_writer_options():
    bundles = [
        {"feature": "t", "tokens": ["A", "B"], "roles": ["VALUE_ABS"], "fingerprint": "f1"},
        {"feature": "t", "tokens": ["C"], "roles": ["VALUE_ABS"], "fingerprint": "f2"},
    ]
    meta = {"feature": "t", "exclude_target_event": True, "exclude_same_case_future": True}
    h1 = bundle_bank_content_sha256(bundles, meta=meta)
    h2 = bundle_bank_content_sha256(list(reversed(bundles)), meta=meta)
    assert h1 == h2


def test_bank_mode_policy_hash_matches_expected():
    d = bank_mode_policy_sha256(
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
    )
    r = bank_mode_policy_sha256(
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        exclude_target_event=True,
        exclude_same_case_future=True,
    )
    assert d != r


def test_all_run_provenance_matches_final_lock():
    from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
        REQUIRED_RUN_IDS,
        build_run_record,
        evaluate_a7_provenance,
    )

    lock = {
        "git_commit": "abc",
        "dependency_tree_sha256": "dep1",
        "cf_source_clean": True,
    }
    runs = []
    for i, rid in enumerate(REQUIRED_RUN_IDS):
        runs.append(
            build_run_record(
                run_id=rid,
                entrypoint=f"scripts/{rid}.py",
                git_commit="abc",
                dependency_tree_sha256="dep1",
                exit_code=0,
                pre_dependency_tree_sha256="dep1",
                post_dependency_tree_sha256="dep1",
                execution_attempt_id="att1",
                orchestrator_invocation_id="inv",
                final_lock_id="lock1",
                run_sequence=i + 1,
                log_sha256="b" * 64,
                artifacts=[{"relative_path": f"a/{rid}.json", "sha256": "a" * 64}],
                config_sha256="c" * 64,
                stage_a_checkpoint_sha256="d" * 64,
                vocab_sha256="e" * 64,
                bundle_bank_content_sha256="f" * 64,
                runtime_imported_project_paths=["src/online2/v2/tokenizer.py"],
            )
        )
    prov = {
        "active_execution_attempt_id": "att1",
        "git_commit": "abc",
        "dependency_tree_sha256": "dep1",
        "pre_dependency_tree_sha256": "dep1",
        "post_dependency_tree_sha256": "dep1",
        "runs": runs,
    }
    a7 = evaluate_a7_provenance(prov, final_lock=lock)
    assert a7["a7_pass"] is True, a7["errors"]


def test_missing_run_log_fails_a7():
    from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
        evaluate_a7_provenance,
    )

    a7 = evaluate_a7_provenance({"runs": [], "rerun_flag_only": True})
    assert a7["a7_pass"] is False


def test_selected_keys_exactly_match_metrics_keys():
    out = exact_match_selection_metrics(
        manifest_keys=["a", "b"],
        metrics_keys=["a", "b"],
        manifest_recoverable_flags=[
            {"mg_key": "a", "is_recoverable": True},
            {"mg_key": "b", "is_recoverable": False},
        ],
        metrics_recoverable_flags=[
            {"mg_key": "a", "is_recoverable": True},
            {"mg_key": "b", "is_recoverable": False},
        ],
    )
    assert out["a8_2_pass"] is True
    bad = exact_match_selection_metrics(
        manifest_keys=["a", "b"],
        metrics_keys=["a", "c"],
        manifest_recoverable_flags=[],
        metrics_recoverable_flags=[],
    )
    assert bad["a8_2_pass"] is False


def test_recoverable_flags_exactly_match_metrics():
    out = exact_match_selection_metrics(
        manifest_keys=["a"],
        metrics_keys=["a"],
        manifest_recoverable_flags=[{"mg_key": "a", "is_recoverable": True}],
        metrics_recoverable_flags=[{"mg_key": "a", "is_recoverable": False}],
    )
    assert out["recoverable_flags_exact_match"] is False
    assert out["a8_2_pass"] is False


def test_official_quality_enum_rejects_provisional_pass():
    rep = evaluate_p1_acceptance_conditions(
        a1_preflight_pass=True,
        a2_functional=True,
        a3_curated_non_original_critic=True,
        a4_outcome_equals_natural=True,
        a5_contracts=True,
        a6_final_code_lock=True,
        a7_rerun_after_lock=True,
        a8_artifact_lock=True,
        a8_2_selection_exact_match=True,
        q1=True,
        q2=True,
        q3=True,
        q4=True,
        q5=True,
        selection_manifest_chain_hash="abc",
        quality_override="PROVISIONAL_PASS",
    )
    assert rep["mlm_reconstruction_quality"] == "NOT_EVALUATED"
    assert rep["mlm_reconstruction_quality"] != "PROVISIONAL_PASS"


def test_outcome_equals_natural_auto():
    rep = evaluate_p1_acceptance_conditions(
        a1_preflight_pass=True,
        a2_functional=True,
        a3_curated_non_original_critic=True,
        a4_outcome_equals_natural=True,
        a5_contracts=True,
        a6_final_code_lock=True,
        a7_rerun_after_lock=True,
        a8_artifact_lock=True,
        mlm_cf_candidate_outcome="NO_VALID_MLM_CF",
        natural_integration_outcome="NO_VALID_MLM_CF",
        selection_manifest_chain_hash="abc",
    )
    assert rep["mlm_cf_candidate_outcome"] == rep["natural_integration_outcome"]
    assert rep["acceptance_conditions"]["A4"] == "PASS"


def test_a8_2_mismatch_forces_quality_not_evaluated_and_impl_fail():
    rep = evaluate_p1_acceptance_conditions(
        a1_preflight_pass=True,
        a2_functional=True,
        a3_curated_non_original_critic=True,
        a4_outcome_equals_natural=True,
        a5_contracts=True,
        a6_final_code_lock=True,
        a7_rerun_after_lock=True,
        a8_artifact_lock=True,
        a8_2_selection_exact_match=False,
        q1=True,
        q2=True,
        q3=True,
        q4=True,
        q5=True,
        selection_manifest_chain_hash="abc",
    )
    assert rep["mlm_implementation"] == "FAIL"
    assert rep["mlm_reconstruction_quality"] == "NOT_EVALUATED"
    assert rep["reconstruction_metric_audit_status"] == "FAIL"
