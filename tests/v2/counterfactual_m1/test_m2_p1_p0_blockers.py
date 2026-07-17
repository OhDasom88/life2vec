"""P0 closure blocker regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    make_bundle_id,
    observations_and_uniques_from_rows,
    query_unique_bundles_from_bank,
    write_two_tier_bank,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    BANK_MODE_RECONSTRUCTION_EVAL,
    evaluate_a8_1_bank_integrity,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    evaluate_a2_functional_smoke,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    SELECTION_ORDER_VERSION,
    select_mg_keys_deterministic,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import (
    PROJECT_IMPORT_PREFIXES,
    project_paths_from_sys_modules,
)


def test_runtime_import_collection_includes_src_data_new():
    assert any(p.startswith("src/") for p in PROJECT_IMPORT_PREFIXES)
    # Import vocabulary so it appears in sys.modules
    import src.data_new.vocabulary  # noqa: F401

    root = Path(__file__).resolve().parents[3]
    paths = project_paths_from_sys_modules(root)
    assert any(p.startswith("src/data_new/") for p in paths), paths[:20]


def test_dirty_registry_vocabulary_fails_a6(tmp_path: Path, monkeypatch):
    import src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock as sl

    for ep in sl.P1_ENTRYPOINT_PATHS:
        p = tmp_path / ep
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#\n", encoding="utf-8")
    vocab = tmp_path / "src/data_new/vocabulary.py"
    vocab.parent.mkdir(parents=True, exist_ok=True)
    vocab.write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "src/online2/v2/finetune_v03/counterfactual").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/online2/v2/finetune_v03/counterfactual/x.py").write_text("1\n")

    def fake_git(cwd, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("status", "--porcelain"):
            return " M src/data_new/vocabulary.py"
        return ""

    monkeypatch.setattr(sl, "_git", fake_git)
    lock = sl.build_source_lock_v2(
        root=tmp_path,
        config_path=tmp_path / "conf/m1/cf_m2_prereq_smoke.yaml",
        config_sha256="dead",
        runtime_imported_project_paths=["src/data_new/vocabulary.py"],
        require_nonempty_runtime_imports=True,
    )
    assert lock["a6_pass"] is False
    assert lock.get("tracked_changes_in_scope") or not lock.get("cf_source_clean")


def test_all_runtime_project_imports_are_declared(tmp_path: Path, monkeypatch):
    import src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock as sl

    for ep in sl.P1_ENTRYPOINT_PATHS:
        p = tmp_path / ep
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#\n", encoding="utf-8")
    for rel in (
        "src/online2/v2/finetune_v03/counterfactual/x.py",
        "src/data_new/vocabulary.py",
        "src/online2/v2/finetune_v03/checkpoint.py",
        "src/online2/v2/tokenizer.py",
        "src/online2/v2/feature_schema.py",
        "scripts/online2_v2/cache_stage_a_event_embeddings.py",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#\n", encoding="utf-8")
    (tmp_path / "tests/v2/counterfactual_m1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/v2/counterfactual_m1/t.py").write_text("#\n")

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
        runtime_imported_project_paths=["src/data_new/vocabulary.py"],
        require_nonempty_runtime_imports=True,
    )
    assert lock["undeclared_runtime_dependencies"] == []
    assert lock["dependency_closure_complete"] is True


def test_orchestrator_static_gate_blocks_before_subprocess():
    src = Path("scripts/online2_v2/v03/run_p1_acceptance_recovery_v03.py").read_text()
    assert "orchestrator_static_gate.json" in src
    assert "static_gate_pass" in src
    assert "missing_git_commit" in src


def test_reconstruction_query_excludes_target_case():
    rows = [
        {
            "feature": "t",
            "tokens": ["A"],
            "roles": ["R"],
            "signature": "R",
            "case_id": "c_target",
            "event_id": "e1",
            "measurement_group_id": "mg1",
            "timestamp": "2020-01-01T00:00:00",
            "source_partition": "TRAINING",
        },
        {
            "feature": "t",
            "tokens": ["B"],
            "roles": ["R"],
            "signature": "R",
            "case_id": "c_other",
            "event_id": "e2",
            "measurement_group_id": "mg1",
            "timestamp": "2020-01-02T00:00:00",
            "source_partition": "TRAINING",
        },
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    out, _ex = query_unique_bundles_from_bank(
        obs,
        uniq,
        feature="t",
        case_id="c_target",
        mode=BANK_MODE_RECONSTRUCTION_EVAL,
        exclude_target_case=True,
    )
    toks = [tuple(u.get("tokens") or []) for u in out]
    assert ("A",) not in toks
    assert ("B",) in toks


def test_a8_1_fails_on_missing_consumer_or_query_hash(tmp_path: Path):
    rows = [
        {
            "feature": "t",
            "tokens": ["A"],
            "roles": ["R"],
            "signature": "R",
            "case_id": "c1",
            "event_id": "e1",
            "measurement_group_id": "mg1",
            "timestamp": "2020-01-01T00:00:00",
            "source_partition": "TRAINING",
        }
    ]
    obs, uniq = observations_and_uniques_from_rows(rows)
    bank_dir = tmp_path / "bank"
    man = write_two_tier_bank(bank_dir, observations=obs, uniques=uniq)
    result = evaluate_a8_1_bank_integrity(
        bank_dir=bank_dir,
        bank_meta={"bundle_bank_content_sha256": man["bundle_bank_content_sha256"]},
        consumer_bank_refs=[],
        query_records=[],
    )
    assert result["a8_1_pass"] is False
    assert any("missing_required_consumer" in e for e in result["errors"])


def test_a8_2_metrics_keys_from_evaluated_not_manifest_copy():
    src = Path("scripts/online2_v2/v03/run_mlm_reconstruction_eval_v03.py").read_text()
    assert "evaluated_mg_keys" in src
    assert "metrics_keys=metrics_keys_for_a8" in src or "metrics_keys=evaluated" in src
    assert "metrics_keys=selected_keys" not in src.split("exact_match_selection_metrics")[1][:400]


def test_q2_q3_from_manifest_recoverable_flags():
    src = Path("scripts/online2_v2/v03/run_mlm_reconstruction_eval_v03.py").read_text()
    assert "manifest_recoverable" in src
    assert "selected_bank_coverage_diagnostic" in src or "manifest_recoverable /" in src.replace(" ", "")


def test_q4_q5_recomputed_from_per_mg_jsonl():
    src = Path("scripts/online2_v2/v03/run_mlm_reconstruction_eval_v03.py").read_text()
    assert "mlm_reconstruction_per_mg.jsonl" in src
    assert "lift_from_per_mg" in src or "mean_recall_at_3_from_per_mg" in src


def test_selection_default_full_pool_hash_order():
    assert SELECTION_ORDER_VERSION == "hash_order_v1"
    pool = [
        {"case_id": "c", "event_id": f"e{i}", "measurement_group_id": "mg", "feature": "f"}
        for i in range(10)
    ]
    k1 = select_mg_keys_deterministic(pool, max_mg_per_feature=3, selection_seed=1)
    k2 = select_mg_keys_deterministic(pool, max_mg_per_feature=3, selection_seed=2)
    assert k1 != k2 or len(k1) == 0
    sel = Path("scripts/online2_v2/v03/run_mlm_reconstruction_selection_v03.py").read_text()
    assert "default=0" in sel


def test_a2_requires_explicit_functional_smoke_passed():
    ok = {
        "run_type": "FUNCTIONAL_DECODER_SMOKE",
        "cf_evaluation_performed": False,
        "decoder_forward_call_count": 1,
        "unique_scored_bundle_count": 3,
        # missing functional_smoke_passed
    }
    assert evaluate_a2_functional_smoke(ok)["a2_pass"] is False
    ok["functional_smoke_passed"] = True
    assert evaluate_a2_functional_smoke(ok)["a2_pass"] is True
