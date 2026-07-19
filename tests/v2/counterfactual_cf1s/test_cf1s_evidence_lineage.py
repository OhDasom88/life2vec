"""P0/P1 contract tests: selection, promotion, trace, lineage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
    PROMOTED_BUT_UNATTESTED_BY_RECEIPT,
    atomic_promote_directory,
    package_state_after_receipt,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_selection import (
    SEARCH_CONTROL_NO_MATERIAL,
    SEARCH_MATERIAL_STABLE,
    SEARCH_UNSTABLE_OR_MIXED,
    SELECTED_CONTROL_NO_MATERIAL,
    SELECTED_MATERIAL,
    classify_search_materiality,
    select_after_search,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_trace import (
    GlobalForwardTrace,
    TraceKind,
)


def test_search_materiality_classes():
    thr = 0.001
    assert (
        classify_search_materiality({"0": 0.01, "1": 0.02}, locked_threshold=thr)["class"]
        == SEARCH_MATERIAL_STABLE
    )
    assert (
        classify_search_materiality({"0": 0.01, "1": 0.02}, locked_threshold=thr)[
            "expected_effect_direction"
        ]
        == "RISK_INCREASE"
    )
    assert (
        classify_search_materiality({"0": -0.01, "1": -0.02}, locked_threshold=thr)[
            "expected_effect_direction"
        ]
        == "RISK_DECREASE"
    )
    assert (
        classify_search_materiality({"0": 0.0, "1": 0.0}, locked_threshold=thr)["class"]
        == SEARCH_CONTROL_NO_MATERIAL
    )
    assert (
        classify_search_materiality({"0": 0.0, "1": 0.0}, locked_threshold=thr)[
            "expected_effect_direction"
        ]
        is None
    )
    assert (
        classify_search_materiality({"0": 0.01, "1": -0.02}, locked_threshold=thr)["class"]
        == SEARCH_UNSTABLE_OR_MIXED
    )
    assert (
        classify_search_materiality({"0": 0.01, "1": 0.0}, locked_threshold=thr)["class"]
        == SEARCH_UNSTABLE_OR_MIXED
    )


def test_selection_deterministic_and_control_cap():
    rows = [
        {
            "candidate_id": "P_ctrl",
            "event_ids": ["a", "b"],
            "fold_deltas": {"0": 0.0, "1": 0.0},
        },
        {
            "candidate_id": "P_mat",
            "event_ids": ["c", "d"],
            "fold_deltas": {"0": -0.01, "1": -0.02},
        },
        {
            "candidate_id": "P_mix",
            "event_ids": ["e", "f"],
            "fold_deltas": {"0": 0.01, "1": -0.02},
        },
    ]
    # Permute input order
    m1 = select_after_search(pair_rows=rows, locked_threshold=0.001)
    m2 = select_after_search(pair_rows=list(reversed(rows)), locked_threshold=0.001)
    assert m1["selection_manifest_sha256"] == m2["selection_manifest_sha256"]
    assert m1["selected_candidate_ids"] == ["P_mat", "P_ctrl"]
    assert m1["selected"][0]["disposition"] == SELECTED_MATERIAL
    assert m1["selected"][0]["expected_effect_direction"] == "RISK_DECREASE"
    assert m1["selected"][1]["disposition"] == SELECTED_CONTROL_NO_MATERIAL
    assert m1["selected"][1]["expected_effect_direction"] is None
    assert any(e["candidate_id"] == "P_mix" for e in m1["excluded"])


def test_trace_state_machine_and_counts():
    tr = GlobalForwardTrace()
    assert tr.begin_operation(
        kind=TraceKind.PIPELINE_INVOCATION,
        invocation_id="p1",
        phase="SEARCH",
        scope="search",
        extra={"forward_kind": "BASELINE_FORWARD"},
    )
    tr.start(kind=TraceKind.PIPELINE_INVOCATION, invocation_id="p1", phase="SEARCH", scope="search")
    tr.complete(
        kind=TraceKind.PIPELINE_INVOCATION,
        invocation_id="p1",
        phase="SEARCH",
        scope="search",
        extra={"forward_kind": "BASELINE_FORWARD"},
    )
    # Denial before side effect
    assert not tr.begin_operation(
        kind=TraceKind.CHECKPOINT_FORWARD,
        invocation_id="c_deny",
        phase="SEARCH",
        scope="search",
        allow=False,
        denial_code="TEST",
    )
    counts = tr.to_forward_counts()
    assert counts["baseline_forward_count"] == 1
    assert counts["denied_count"] == 1
    assert counts["checkpoint_forward_count"] == 0
    with pytest.raises(CoreContractError):
        tr.start(
            kind=TraceKind.CHECKPOINT_FORWARD,
            invocation_id="c_deny",
            phase="SEARCH",
            scope="search",
        )


def test_atomic_promote_rename_only(tmp_path):
    q = tmp_path / "quarantine" / "run1"
    f = tmp_path / "final" / "run1"
    q.mkdir(parents=True)
    (q / "a.json").write_text('{"ok":true}\n', encoding="utf-8")
    meta = atomic_promote_directory(quarantine_dir=q, final_dir=f)
    assert meta["promotion_method"] == "renameat2(RENAME_NOREPLACE)"
    assert f.is_dir()
    assert not q.exists()
    assert (f / "a.json").is_file()
    # overwrite forbidden
    q2 = tmp_path / "quarantine" / "run2"
    q2.mkdir(parents=True)
    (q2 / "b.json").write_text("x\n", encoding="utf-8")
    with pytest.raises(CoreContractError, match="already exists"):
        atomic_promote_directory(quarantine_dir=q2, final_dir=f)


def test_receipt_failure_state():
    assert (
        package_state_after_receipt(promoted=True, receipt_valid=False)
        == PROMOTED_BUT_UNATTESTED_BY_RECEIPT
    )


def test_evidence_lineage_preserves_old_packages():
    lineage = Path("/home/dasom/life2vec/outputs/cf1s_core/EVIDENCE_LINEAGE.json")
    assert lineage.is_file()
    data = json.loads(lineage.read_text(encoding="utf-8"))
    assert data["current_claim_status"]["highest_claim_level"] == "NONE"
    old = Path(
        "/home/dasom/life2vec/outputs/cf1s_core/development3_evidence/20260719T052936Z"
    )
    assert old.is_dir()
    # byte-immutable: runner_summary still present
    assert (old / "runner_summary.json").is_file()
    for pkg in data["invalidated_packages"]:
        assert pkg["artifact_status"] == "INVALIDATED_FOR_FINAL_CLAIM"
        if pkg["superseded_by"] is not None:
            successor = Path(
                "/home/dasom/life2vec/outputs/cf1s_core/development3_evidence"
            ) / str(pkg["superseded_by"])
            assert successor.is_dir()
