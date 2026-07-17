"""P0 remediation integrity tests (A2/A7 attempt/bank FK/source-lock entrypoints)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    observations_and_uniques_from_rows,
    write_two_tier_bank,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    validate_bank_relational_integrity,
    observation_index_content_sha256,
    unique_bundle_index_content_sha256,
    composite_bundle_bank_content_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    REQUIRED_RUN_IDS,
    build_run_record,
    compute_artifact_manifest_sha256,
    evaluate_a2_functional_smoke,
    evaluate_a7_provenance,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    exact_match_selection_metrics,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import (
    P1_ENTRYPOINT_PATHS,
    build_source_lock_v2,
)


def _mk_run(
    run_id: str,
    seq: int,
    attempt: str = "att1",
    *,
    exit_code: int = 0,
    bank: bool = True,
):
    arts = [{"relative_path": f"artifacts/{run_id}.json", "sha256": "a" * 64}]
    rec = build_run_record(
        run_id=run_id,
        entrypoint=f"scripts/{run_id}.py",
        git_commit="abc",
        dependency_tree_sha256="dep",
        exit_code=exit_code,
        pre_dependency_tree_sha256="dep",
        post_dependency_tree_sha256="dep",
        execution_attempt_id=attempt,
        orchestrator_invocation_id="inv1",
        final_lock_id="lock1",
        run_sequence=seq,
        log_sha256="b" * 64,
        artifacts=arts,
        config_sha256="c" * 64,
        stage_a_checkpoint_sha256="d" * 64,
        vocab_sha256="e" * 64,
        bundle_bank_content_sha256=("f" * 64) if bank else None,
        runtime_imported_project_paths=["src/online2/v2/tokenizer.py"],
    )
    return rec


def test_a2_requires_dedicated_functional_artifact():
    assert evaluate_a2_functional_smoke({}).get("a2_pass") is False
    ok = {
        "run_type": "FUNCTIONAL_DECODER_SMOKE",
        "cf_evaluation_performed": False,
        "decoder_forward_call_count": 1,
        "unique_scored_bundle_count": 3,
        "functional_smoke_passed": True,
    }
    assert evaluate_a2_functional_smoke(ok)["a2_pass"] is True


def test_a2_rejects_natural_funnel_as_evidence():
    bad = {
        "run_type": "NATURAL_PATH_A",
        "decoder_forward_call_count": 5,
        "unique_scored_bundle_count": 10,
        "functional_smoke_passed": True,
    }
    assert evaluate_a2_functional_smoke(bad)["a2_pass"] is False


def test_a2_rejects_no_eligible_locus_bypass():
    # preflight-like empty functional artifact must fail
    assert evaluate_a2_functional_smoke({"decoder_preflight_passed": True})["a2_pass"] is False


def test_a7_requires_single_complete_execution_attempt():
    runs = [_mk_run(rid, i + 1) for i, rid in enumerate(REQUIRED_RUN_IDS)]
    prov = {"active_execution_attempt_id": "att1", "runs": runs}
    a7 = evaluate_a7_provenance(prov)
    assert a7["a7_pass"] is True, a7["errors"]


def test_a7_rejects_required_runs_mixed_across_attempts():
    runs = []
    for i, rid in enumerate(REQUIRED_RUN_IDS[:4]):
        runs.append(_mk_run(rid, i + 1, attempt="att1"))
    for i, rid in enumerate(REQUIRED_RUN_IDS[4:]):
        runs.append(_mk_run(rid, i + 5, attempt="att2"))
    prov = {"active_execution_attempt_id": "att2", "runs": runs}
    a7 = evaluate_a7_provenance(prov)
    assert a7["a7_pass"] is False
    assert any("missing_required_run" in e for e in a7["errors"])


def test_a7_rejects_duplicate_required_run_in_same_attempt():
    runs = [_mk_run(rid, i + 1) for i, rid in enumerate(REQUIRED_RUN_IDS)]
    runs.append(_mk_run("pytest", 9))
    a7 = evaluate_a7_provenance({"active_execution_attempt_id": "att1", "runs": runs})
    assert a7["a7_pass"] is False
    assert any("duplicate_required_run:pytest" in e for e in a7["errors"])


def test_a7_requires_expected_run_sequence():
    runs = [_mk_run(rid, 99 - i) for i, rid in enumerate(REQUIRED_RUN_IDS)]
    a7 = evaluate_a7_provenance({"active_execution_attempt_id": "att1", "runs": runs})
    assert a7["a7_pass"] is False
    assert "run_sequence_mismatch" in a7["errors"]


def test_artifact_manifest_hash_canonical():
    arts = [
        {"relative_path": "b.json", "sha256": "2" * 64},
        {"relative_path": "a.json", "sha256": "1" * 64},
    ]
    h1 = compute_artifact_manifest_sha256(run_id="r", artifacts=arts)
    h2 = compute_artifact_manifest_sha256(run_id="r", artifacts=list(reversed(arts)))
    assert h1 == h2


def test_source_lock_requires_recovery_orchestrator():
    assert (
        "scripts/online2_v2/v03/run_p1_acceptance_recovery_v03.py" in P1_ENTRYPOINT_PATHS
    )


def test_source_lock_requires_reconstruction_selection_entrypoint():
    assert (
        "scripts/online2_v2/v03/run_mlm_reconstruction_selection_v03.py"
        in P1_ENTRYPOINT_PATHS
    )


def test_source_lock_requires_functional_smoke_entrypoint():
    assert "scripts/online2_v2/v03/run_mlm_functional_smoke_v03.py" in P1_ENTRYPOINT_PATHS


def test_source_lock_missing_entrypoint_fails_a6(tmp_path: Path, monkeypatch):
    # Create all but one required entrypoint
    for ep in P1_ENTRYPOINT_PATHS:
        if ep.endswith("run_p1_acceptance_recovery_v03.py"):
            continue
        p = tmp_path / ep
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#\n", encoding="utf-8")
    (tmp_path / "src/online2/v2/finetune_v03/counterfactual").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/online2/v2/finetune_v03/counterfactual/x.py").write_text("1\n")

    import src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock as sl

    def fake_git(cwd, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("status", "--porcelain"):
            return ""
        return ""

    monkeypatch.setattr(sl, "_git", fake_git)
    lock = sl.build_source_lock_v2(
        root=tmp_path,
        config_path=tmp_path / "conf/m1/cf_m2_prereq_smoke.yaml",
        config_sha256="dead",
        require_nonempty_runtime_imports=False,
    )
    assert lock["entrypoints_complete"] is False
    assert lock["a6_pass"] is False


def test_every_observation_bundle_id_exists_in_unique_index():
    rows = [
        {
            "feature": "t",
            "tokens": ["A", "B"],
            "roles": ["VALUE_ABS"],
            "case_id": "c1",
            "event_id": "e1",
            "timestamp": "2020-01-01",
            "source_partition": "TRAINING",
            "fingerprint": "fp1",
        },
        {
            "feature": "t",
            "tokens": ["A", "B"],
            "roles": ["VALUE_ABS"],
            "case_id": "c2",
            "event_id": "e2",
            "timestamp": "2020-01-02",
            "source_partition": "TRAINING",
            "fingerprint": "fp2",
        },
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    rel = validate_bank_relational_integrity(obs, uniq)
    assert rel["a8_1_relational_pass"] is True
    assert rel["orphan_observation_count"] == 0


def test_unique_observation_count_matches_observation_rows():
    rows = [
        {
            "feature": "t",
            "tokens": ["X"],
            "roles": ["R"],
            "case_id": "c",
            "event_id": f"e{i}",
            "measurement_group_id": "mg1",
            "timestamp": f"2020-01-0{i+1}",
            "source_partition": "TRAINING",
        }
        for i in range(3)
    ]
    # true duplicate observation (same event fingerprint components)
    rows.append(
        {
            "feature": "t",
            "tokens": ["X"],
            "roles": ["R"],
            "case_id": "c",
            "event_id": "e0",
            "measurement_group_id": "mg1",
            "timestamp": "2020-01-01",
            "source_partition": "TRAINING",
        }
    )
    obs, uniq = observations_and_uniques_from_rows(rows)
    assert uniq[0]["raw_observation_count"] == 4
    assert uniq[0]["deduplicated_observation_count"] == 3
    assert uniq[0]["observation_count_used_for_provenance"] == 3
    rel = validate_bank_relational_integrity(obs, uniq)
    assert rel["observation_counts_exact_match"] is True


def test_orphan_unique_bundle_fails_a8_1():
    obs = [
        {
            "bundle_id": "b1",
            "feature": "t",
            "role_signature": "R",
            "case_id": "c",
            "event_id": "e",
            "timestamp_utc": "t",
            "event_fingerprint": "f",
        }
    ]
    uniq = [
        {
            "bundle_id": "b1",
            "observation_count_used_for_provenance": 1,
            "deduplicated_observation_count": 1,
        },
        {
            "bundle_id": "orphan",
            "observation_count_used_for_provenance": 1,
            "deduplicated_observation_count": 1,
        },
    ]
    rel = validate_bank_relational_integrity(obs, uniq)
    assert rel["a8_1_relational_pass"] is False
    assert rel["orphan_unique_bundle_count"] == 1


def test_query_does_not_change_base_content_hash(tmp_path: Path):
    rows = [
        {
            "feature": "t",
            "tokens": ["A"],
            "roles": ["R"],
            "case_id": "c1",
            "event_id": "e1",
            "timestamp": "2020-01-01",
            "source_partition": "TRAINING",
            "fingerprint": "fp1",
        }
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path, observations=obs, uniques=uniq, mode=BANK_MODE_DEPLOYMENT)
    h1 = man["bundle_bank_content_sha256"]
    # recompute without query meta
    h2 = composite_bundle_bank_content_sha256(
        observation_index_content_sha256=observation_index_content_sha256(obs),
        unique_bundle_index_content_sha256=unique_bundle_index_content_sha256(uniq),
    )
    assert h1 == h2


def test_a8_2_fails_when_flags_missing():
    out = exact_match_selection_metrics(
        manifest_keys=["a"],
        metrics_keys=["a"],
        manifest_recoverable_flags=[],
        metrics_recoverable_flags=[],
    )
    # empty flag maps for key a → flags_match false because set(m_flags) != set(keys)
    assert out["a8_2_pass"] is False


def test_a8_2_detects_flag_mismatch():
    out = exact_match_selection_metrics(
        manifest_keys=["a"],
        metrics_keys=["a"],
        manifest_recoverable_flags=[{"mg_key": "a", "is_recoverable": True}],
        metrics_recoverable_flags=[{"mg_key": "a", "is_recoverable": False}],
    )
    assert out["a8_2_pass"] is False
