"""P0 rerun blocker behavioral regression tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    build_bank_query_record,
    build_bank_query_set_manifest,
    wrap_executed_query_record,
    make_bundle_id,
    normalize_utc_timestamp,
    observations_and_uniques_from_rows,
    query_unique_bundles_from_bank,
    write_two_tier_bank,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    BANK_MODE_RECONSTRUCTION_EVAL,
    EXPECTED_POLICY_BY_MODE,
    SOURCE_PARTITION_PROBLEM,
    SOURCE_PARTITION_TRAINING,
    bank_mode_policy_sha256,
    bank_query_manifest_sha256,
    evaluate_a8_1_bank_integrity,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    REQUIRED_RUN_IDS,
    build_run_record,
    evaluate_a7_provenance,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    MIN_RECONSTRUCTION_CANDIDATES,
    mg_key,
)


def _row(
    *,
    feature: str = "t",
    tokens,
    roles=None,
    case_id: str,
    event_id: str,
    mg: str = "mg1",
    timestamp: str,
    partition: str = SOURCE_PARTITION_TRAINING,
):
    roles = roles or ["R"]
    return {
        "feature": feature,
        "tokens": list(tokens),
        "roles": list(roles),
        "signature": "|".join(roles),
        "case_id": case_id,
        "event_id": event_id,
        "measurement_group_id": mg,
        "timestamp": timestamp,
        "source_partition": partition,
    }


def test_deployment_query_compares_naive_and_utc_timestamps_safely():
    rows = [
        _row(tokens=["PAST"], case_id="prob", event_id="e0", timestamp="2020-01-01T00:00:00"),
        _row(tokens=["FUT"], case_id="prob", event_id="e2", timestamp="2020-01-03T00:00:00"),
        _row(tokens=["TRAIN"], case_id="train1", event_id="e9", timestamp="2019-01-01T00:00:00"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    # naive target — previously raised TypeError vs Z-stored obs
    target = pd.Timestamp("2020-01-02T00:00:00")  # naive
    assert target.tzinfo is None
    out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="prob",
        target_event_id="e1",
        target_timestamp=target,
        mode=BANK_MODE_DEPLOYMENT,
    )
    toks = {tuple(u["tokens"]) for u in out}
    assert ("PAST",) in toks
    assert ("TRAIN",) in toks
    assert ("FUT",) not in toks


def test_same_timestamp_exclusion_uses_normalized_utc():
    rows = [
        _row(tokens=["SAME"], case_id="c1", event_id="e_same", timestamp="2020-01-02T12:00:00"),
        _row(tokens=["PAST"], case_id="c1", event_id="e_past", timestamp="2020-01-01T12:00:00"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    target = pd.Timestamp("2020-01-02T12:00:00Z")
    out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="c1",
        target_event_id="other",
        target_timestamp=target,
        mode=BANK_MODE_DEPLOYMENT,
    )
    toks = {tuple(u["tokens"]) for u in out}
    assert ("SAME",) not in toks
    assert ("PAST",) in toks
    assert normalize_utc_timestamp("2020-01-02T12:00:00") == normalize_utc_timestamp(
        "2020-01-02T12:00:00Z"
    )


def test_target_case_past_query_does_not_raise_timezone_error():
    rows = [
        _row(
            tokens=["P"],
            case_id="prob",
            event_id="e0",
            timestamp="2020-01-01T00:00:00.000000Z",
            partition=SOURCE_PARTITION_PROBLEM,
        ),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    _out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="prob",
        target_event_id="eX",
        target_timestamp=pd.Timestamp("2020-01-02"),
        mode=BANK_MODE_DEPLOYMENT,
    )


def test_a8_1_rejects_reconstruction_without_target_case_exclusion(tmp_path: Path):
    rows = [_row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01")]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path / "bank", observations=obs, uniques=uniq)
    _b, executed = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="c1",
        target_event_id="e1",
        target_timestamp="2020-01-01",
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        expected_roles=["R"],
    )
    bad = build_bank_query_record(
        run_id="reconstruction_eval",
        bank_manifest=man,
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        target_event_id="e1",
        case_id="c1",
        feature="t",
        executed_query_manifest=executed,
    )
    # Tamper: claim exclude_target_case=false but keep self-consistent wrong hashes
    bad["exclude_target_case"] = False
    bad["query_manifest"]["exclude_target_case"] = False
    bad["bundle_bank_policy_sha256"] = bank_mode_policy_sha256(
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=False,
        exclude_duplicate_event_fingerprints=True,
        same_timestamp_policy="EXCLUDE",
        allowed_source_partitions=["TRAINING"],
    )
    bad["bundle_bank_query_manifest_sha256"] = bank_query_manifest_sha256(
        **bad["query_manifest"]
    )
    bad["query_set_manifest"] = {
        "run_id": "reconstruction_eval",
        "query_count": 1,
        "query_records": [
            {
                "mg_key": "c1|e1|mg1|t",
                "query_manifest": bad["query_manifest"],
                "query_manifest_sha256": bad["bundle_bank_query_manifest_sha256"],
            }
        ],
    }
    from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
        query_set_manifest_sha256,
    )

    bad["query_set_sha256"] = query_set_manifest_sha256(
        run_id="reconstruction_eval",
        query_count=1,
        query_records=bad["query_set_manifest"]["query_records"],
    )
    bad["query_set_manifest"]["query_set_sha256"] = bad["query_set_sha256"]
    bad["expected_mg_keys"] = ["c1|e1|mg1|t"]
    bad["query_records"] = bad["query_set_manifest"]["query_records"]
    bad["query_count"] = 1

    result = evaluate_a8_1_bank_integrity(
        bank_dir=tmp_path / "bank",
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        query_records=[bad],
        required_consumer_run_ids=("reconstruction_eval",),
    )
    assert result["a8_1_pass"] is False
    assert any("canonical_policy_mismatch:exclude_target_case" in e for e in result["errors"])


def test_a8_1_rejects_wrong_same_timestamp_policy(tmp_path: Path):
    rows = [_row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01")]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path / "bank", observations=obs, uniques=uniq)
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
    rec = build_bank_query_record(
        run_id="functional_smoke",
        bank_manifest=man,
        mode=BANK_MODE_DEPLOYMENT,
        target_event_id="e1",
        case_id="c1",
        feature="t",
        executed_query_manifest=executed,
    )
    rec["same_timestamp_policy"] = "KEEP"
    rec["query_manifest"]["same_timestamp_policy"] = "KEEP"
    rec["bundle_bank_policy_sha256"] = bank_mode_policy_sha256(
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=False,
        exclude_duplicate_event_fingerprints=True,
        same_timestamp_policy="KEEP",
        allowed_source_partitions=list(
            EXPECTED_POLICY_BY_MODE[BANK_MODE_DEPLOYMENT]["allowed_source_partitions"]
        ),
    )
    rec["bundle_bank_query_manifest_sha256"] = bank_query_manifest_sha256(**rec["query_manifest"])
    result = evaluate_a8_1_bank_integrity(
        bank_dir=tmp_path / "bank",
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        query_records=[rec],
        required_consumer_run_ids=("functional_smoke",),
    )
    assert result["a8_1_pass"] is False
    assert any("same_timestamp_policy" in e for e in result["errors"])


def test_a8_1_rejects_wrong_duplicate_fingerprint_policy(tmp_path: Path):
    rows = [_row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01")]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path / "bank", observations=obs, uniques=uniq)
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
    rec = build_bank_query_record(
        run_id="functional_smoke",
        bank_manifest=man,
        mode=BANK_MODE_DEPLOYMENT,
        target_event_id="e1",
        case_id="c1",
        feature="t",
        executed_query_manifest=executed,
    )
    rec["exclude_duplicate_event_fingerprints"] = False
    rec["query_manifest"]["exclude_duplicate_event_fingerprints"] = False
    rec["bundle_bank_policy_sha256"] = bank_mode_policy_sha256(
        mode=BANK_MODE_DEPLOYMENT,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=False,
        exclude_duplicate_event_fingerprints=False,
        same_timestamp_policy="EXCLUDE",
        allowed_source_partitions=list(
            EXPECTED_POLICY_BY_MODE[BANK_MODE_DEPLOYMENT]["allowed_source_partitions"]
        ),
    )
    rec["bundle_bank_query_manifest_sha256"] = bank_query_manifest_sha256(**rec["query_manifest"])
    result = evaluate_a8_1_bank_integrity(
        bank_dir=tmp_path / "bank",
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        query_records=[rec],
        required_consumer_run_ids=("functional_smoke",),
    )
    assert result["a8_1_pass"] is False
    assert any("exclude_duplicate_event_fingerprints" in e for e in result["errors"])


def test_reconstruction_query_excludes_all_problem_partition_observations():
    rows = [
        _row(
            tokens=["TRAIN"],
            case_id="t1",
            event_id="e1",
            timestamp="2020-01-01",
            partition=SOURCE_PARTITION_TRAINING,
        ),
        _row(
            tokens=["PROB"],
            case_id="p1",
            event_id="e2",
            timestamp="2020-01-01",
            partition=SOURCE_PARTITION_PROBLEM,
        ),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="other",
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
    )
    toks = {tuple(u["tokens"]) for u in out}
    assert ("TRAIN",) in toks
    assert ("PROB",) not in toks


def test_deployment_query_allows_target_problem_case_past_only():
    rows = [
        _row(
            tokens=["TRAIN"],
            case_id="t1",
            event_id="e1",
            timestamp="2019-01-01",
            partition=SOURCE_PARTITION_TRAINING,
        ),
        _row(
            tokens=["PROB_PAST"],
            case_id="prob",
            event_id="e2",
            timestamp="2020-01-01",
            partition=SOURCE_PARTITION_PROBLEM,
        ),
        _row(
            tokens=["PROB_OTHER"],
            case_id="other_prob",
            event_id="e3",
            timestamp="2020-01-01",
            partition=SOURCE_PARTITION_PROBLEM,
        ),
        _row(
            tokens=["PROB_FUT"],
            case_id="prob",
            event_id="e4",
            timestamp="2020-01-03",
            partition=SOURCE_PARTITION_PROBLEM,
        ),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="prob",
        target_event_id="eX",
        target_timestamp=pd.Timestamp("2020-01-02"),
        mode=BANK_MODE_DEPLOYMENT,
    )
    toks = {tuple(u["tokens"]) for u in out}
    assert ("TRAIN",) in toks
    assert ("PROB_PAST",) in toks
    assert ("PROB_OTHER",) not in toks
    assert ("PROB_FUT",) not in toks


def test_source_partition_is_bound_into_query_policy_hash():
    a = bank_mode_policy_sha256(
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=True,
        exclude_duplicate_event_fingerprints=True,
        same_timestamp_policy="EXCLUDE",
        allowed_source_partitions=["TRAINING"],
    )
    b = bank_mode_policy_sha256(
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        exclude_target_event=True,
        exclude_same_case_future=True,
        exclude_target_case=True,
        exclude_duplicate_event_fingerprints=True,
        same_timestamp_policy="EXCLUDE",
        allowed_source_partitions=["TRAINING", "PROBLEM"],
    )
    assert a != b
    _b, executed = query_unique_bundles_from_bank(
        [],
        [],
        feature="t",
        case_id="c",
        target_event_id="e",
        target_timestamp="2020-01-01",
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        expected_roles=["R"],
    )
    rec = build_bank_query_record(
        run_id="x",
        bank_manifest={"bundle_bank_content_sha256": "a" * 64},
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        target_event_id="e",
        case_id="c",
        feature="t",
        executed_query_manifest=executed,
    )
    assert rec["allowed_source_partitions"] == ["TRAINING"]
    assert "allowed_source_partitions" in rec["query_manifest"]


def test_query_recomputes_observation_count_after_target_case_exclusion():
    # same bundle observed 3 times: 1 target case + 2 other
    bid_tokens = ["SHARED"]
    rows = [
        _row(tokens=bid_tokens, case_id="target", event_id="e1", timestamp="2020-01-01"),
        _row(tokens=bid_tokens, case_id="o1", event_id="e2", timestamp="2020-01-01"),
        _row(tokens=bid_tokens, case_id="o2", event_id="e3", timestamp="2020-01-01"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    assert uniq[0]["observation_count_used_for_provenance"] == 3
    out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="target",
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
    )
    assert len(out) == 1
    assert out[0]["base_observation_count"] == 3
    assert out[0]["eligible_observation_count"] == 2
    assert out[0]["observation_count_used_for_provenance"] == 2


def test_query_recomputes_count_after_future_exclusion():
    tokens = ["B"]
    rows = [
        _row(tokens=tokens, case_id="c1", event_id="e_past", timestamp="2020-01-01"),
        _row(tokens=tokens, case_id="c1", event_id="e_fut", timestamp="2020-01-03"),
        _row(tokens=tokens, case_id="c2", event_id="e_other", timestamp="2020-01-01"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="c1",
        target_event_id="eX",
        target_timestamp=pd.Timestamp("2020-01-02"),
        mode=BANK_MODE_DEPLOYMENT,
    )
    assert out[0]["eligible_observation_count"] == 2  # past + other case
    assert out[0]["base_observation_count"] == 3


def test_candidate_observation_count_uses_eligible_observations_only():
    tokens = ["C"]
    rows = [
        _row(tokens=tokens, case_id="t", event_id="e1", timestamp="2020-01-01"),
        _row(tokens=tokens, case_id="o", event_id="e2", timestamp="2020-01-01"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    out, _ex = query_unique_bundles_from_bank(
        obs, uniq, feature="t", case_id="t", mode=BANK_MODE_RECONSTRUCTION_EVAL
    )
    assert out[0]["observation_count"] == out[0]["eligible_observation_count"] == 1


def test_query_set_manifest_count_and_sha(tmp_path: Path):
    rows = [
        _row(tokens=["A"], case_id="c1", event_id="e1", timestamp="2020-01-01"),
        _row(tokens=["B"], case_id="c2", event_id="e2", timestamp="2020-01-01"),
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    man = write_two_tier_bank(tmp_path / "bank", observations=obs, uniques=uniq)
    sel_rows = [
        {
            "case_id": "c1",
            "event_id": "e1",
            "measurement_group_id": "mg1",
            "feature": "t",
            "original_roles": ["R"],
        },
        {
            "case_id": "c2",
            "event_id": "e2",
            "measurement_group_id": "mg1",
            "feature": "t",
            "original_roles": ["R"],
        },
    ]
    executed_entries = []
    for row in sel_rows:
        _b, executed = query_unique_bundles_from_bank(
            obs,
            uniq,
            feature="t",
            case_id=row["case_id"],
            target_event_id=row["event_id"],
            target_timestamp="2020-01-01",
            mode=BANK_MODE_RECONSTRUCTION_EVAL,
            expected_roles=["R"],
        )
        executed_entries.append({"mg_key": mg_key(row), "executed_query_manifest": executed})
    qset = build_bank_query_set_manifest(
        run_id="reconstruction_selection",
        bank_manifest=man,
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        executed_entries=executed_entries,
    )
    assert qset["query_count"] == 2
    assert len(qset["query_records"]) == 2
    assert qset["query_set_sha256"]
    qset["expected_mg_keys"] = [mg_key(r) for r in sel_rows]
    qset["query_set_manifest"] = {
        "run_id": qset["run_id"],
        "query_count": qset["query_count"],
        "query_records": qset["query_records"],
        "query_set_sha256": qset["query_set_sha256"],
    }
    result = evaluate_a8_1_bank_integrity(
        bank_dir=tmp_path / "bank",
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        query_records=[qset],
        required_consumer_run_ids=("reconstruction_selection",),
    )
    assert result["a8_1_pass"] is True, result["errors"]


def test_selection_and_eval_share_min_reconstruction_candidates():
    assert MIN_RECONSTRUCTION_CANDIDATES == 5
    sel = Path("scripts/online2_v2/v03/run_mlm_reconstruction_selection_v03.py").read_text()
    ev = Path("scripts/online2_v2/v03/run_mlm_reconstruction_eval_v03.py").read_text()
    assert "MIN_RECONSTRUCTION_CANDIDATES" in sel
    assert "MIN_RECONSTRUCTION_CANDIDATES" in ev
    assert "if not bundles:" not in sel.split("def evaluate_recoverable")[1].split("return n_ok")[0]
    assert "len(bundles) < 2" not in ev.split("def evaluate_recoverable")[1].split("return n_ok")[0]


def test_a7_fails_when_required_run_missing_runtime_imports():
    runs = [
        build_run_record(
            run_id=rid,
            entrypoint=f"scripts/{rid}.py",
            git_commit="abc",
            dependency_tree_sha256="dep",
            exit_code=0,
            pre_dependency_tree_sha256="dep",
            post_dependency_tree_sha256="dep",
            execution_attempt_id="att1",
            orchestrator_invocation_id="inv1",
            final_lock_id="lock1",
            run_sequence=i + 1,
            log_sha256="b" * 64,
            artifacts=[{"relative_path": f"a/{rid}.json", "sha256": "a" * 64}],
            config_sha256="cfg",
            stage_a_checkpoint_sha256="ckpt",
            vocab_sha256="vocab",
            bundle_bank_content_sha256=("f" * 64)
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
            runtime_imported_project_paths=["src/online2/v2/tokenizer.py"]
            if rid != "pytest"
            else [],
        )
        for i, rid in enumerate(REQUIRED_RUN_IDS)
    ]
    a7 = evaluate_a7_provenance({"active_execution_attempt_id": "att1", "runs": runs})
    assert a7["a7_pass"] is False
    assert any("missing_runtime_imports" in e for e in a7["errors"])
