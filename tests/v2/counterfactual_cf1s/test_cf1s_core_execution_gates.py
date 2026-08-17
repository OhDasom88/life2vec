"""Cold-build selection, legacy runner gates, cohort manifests, Problem20 sealing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
    CoreContractError,
)

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_execution import (
    compare_cold_build_equivalence,
    select_fold_checkpoints,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_preflight import (
    core_preflight,
    validate_cohort_manifests,
)
from src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s_core import (
    run_cf1s_core_cohort_contract,
)

ROOT = Path(__file__).resolve().parents[3]


def test_select_fold_checkpoints_excludes_nonrequested():
    selected = select_fold_checkpoints(
        checkpoint_paths=["a.pt", "b.pt", "c.pt"],
        fold_ids=[0, 1, 2],
        requested_fold_ids=[0, 1],
    )
    assert selected["selected_fold_ids"] == [0, 1]
    assert selected["nonrequested_fold_ids_excluded"] == [2]


def test_cold_build_equivalence_membership_and_tensors():
    mean = torch.randn(2, 4)
    mx = torch.randn(2, 4)
    core = {
        "valid_event_ids": ["E0", "E1"],
        "window_ids": ["window_target_E0", "window_target_E1"],
        "target_membership": {"E0": ["window_target_E0"], "E1": ["window_target_E1"]},
        "context_membership": {"E0": ["window_target_E0", "window_target_E1"]},
        "event_mean": mean,
        "event_max": mx,
    }
    ref = {
        "valid_event_ids": ["E0", "E1"],
        "window_ids": ["window_target_E0", "window_target_E1"],
        "target_membership": {"E0": ["window_target_E0"], "E1": ["window_target_E1"]},
        "context_membership": {"E0": ["window_target_E0", "window_target_E1"]},
        "event_mean": mean.clone(),
        "event_max": mx.clone(),
        "reference_calls_core_cold_build_method": False,
    }
    eq = compare_cold_build_equivalence(core, ref)
    assert eq["cold_rebuild_equivalence_test_pass"]
    assert eq["core_reference_event_to_window_membership_equal"]
    bad = dict(ref)
    bad["context_membership"] = {"E0": ["window_target_E0"]}
    eq2 = compare_cold_build_equivalence(core, bad)
    assert eq2["cold_rebuild_equivalence_test_pass"] is False


def test_cohort_manifests_disjoint_and_problem20_no_labels():
    out = validate_cohort_manifests(
        root=ROOT,
        development_path=ROOT
        / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_MANIFEST.json",
        primary32_path=ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_PRIMARY32_MANIFEST.json",
        problem20_path=ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_PROBLEM20_MANIFEST.json",
    )
    assert out["ok"]
    sealed = json.loads(
        (
            ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_PROBLEM20_LABELS_SEALED.json"
        ).read_text(encoding="utf-8")
    )
    assert sealed["sealed"] is True
    assert len(sealed["cases"]) == 20
    runtime = json.loads(
        (
            ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_PROBLEM20_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    assert runtime["ground_truth_value_in_runtime_manifest"] is False
    for c in runtime["cases"]:
        assert "label" not in c


def test_legacy_runner_requires_flag_and_blocks_production():
    py = sys.executable
    r = subprocess.run(
        [py, str(ROOT / "scripts/online2_v2/v03/run_cf1s_v03.py"), "--mode", "fixture"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "FROZEN_REFERENCE_ONLY" in (r.stderr + r.stdout)
    r2 = subprocess.run(
        [
            py,
            str(ROOT / "scripts/online2_v2/v03/run_cf1s_v03.py"),
            "--allow-legacy-fixture-only",
            "--mode",
            "production",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r2.returncode != 0
    assert "Legacy production mode is blocked" in (r2.stderr + r2.stdout)


def test_problem20_blocked_without_authorization():
    # Problem20's authorization marker genuinely does not exist in this repo
    # (Problem20 execution is never reached by this pipeline), so this
    # subprocess check against real repo state remains valid.
    py = sys.executable
    r2 = subprocess.run(
        [
            py,
            str(ROOT / "scripts/online2_v2/v03/run_cf1s_core_v03.py"),
            "--cohort",
            "problem20",
            "--mode",
            "contract",
            "--skip-preflight",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r2.returncode != 0


def test_assert_primary32_authorized_gate(tmp_path):
    # Unlike Problem20, Primary32's authorization artifact is expected to
    # exist for real once Development3 + Validation20 both reach FINAL PASS —
    # so this checks the assert_primary32_authorized() contract directly
    # against isolated fixtures, rather than a subprocess against real repo
    # state (which would start failing the moment Primary32 is legitimately
    # authorized, exactly as intended).
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_readiness import (
        assert_primary32_authorized,
    )

    missing = tmp_path / "missing.json"
    with pytest.raises(CoreContractError, match="lock readiness missing"):
        assert_primary32_authorized(missing)

    def _write(doc: dict) -> Path:
        path = tmp_path / "auth.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    valid = {
        "artifact_kind": "CF1S_PRIMARY32_EXECUTION_AUTHORIZATION_V1",
        "scope": ["PRIMARY32", "TWO_EVENT_ONLY"],
        "primary32_execution_authorized": True,
        "threshold_change_authorized": False,
        "source_development_final_verdict_sha256": "a" * 64,
        "source_validation20_final_verdict_sha256": "b" * 64,
    }
    assert assert_primary32_authorized(_write(valid)) == valid

    with pytest.raises(CoreContractError, match="not authoritative"):
        assert_primary32_authorized(_write({**valid, "artifact_kind": "OTHER"}))
    with pytest.raises(CoreContractError, match="scope mismatch"):
        assert_primary32_authorized(_write({**valid, "scope": ["PRIMARY32"]}))
    with pytest.raises(CoreContractError, match="not authorized"):
        assert_primary32_authorized(
            _write({**valid, "primary32_execution_authorized": False})
        )
    with pytest.raises(CoreContractError, match="threshold changes"):
        assert_primary32_authorized(
            _write({**valid, "threshold_change_authorized": True})
        )
    with pytest.raises(CoreContractError, match="Development3 FINAL link"):
        assert_primary32_authorized(
            _write({**valid, "source_development_final_verdict_sha256": None})
        )
    with pytest.raises(CoreContractError, match="Validation20 FINAL link"):
        assert_primary32_authorized(
            _write({**valid, "source_validation20_final_verdict_sha256": None})
        )


def test_core_contract_cohort_pipeline_runs():
    out = run_cf1s_core_cohort_contract(
        case_ids=[
            "F420458_2025-02-16_2025-03-01",
            "F385790_2025-03-20_2025-04-02",
            "F279269_2024-10-17_2024-10-30",
        ],
        search_fold_ids=[0, 1],
        holdout_fold_ids=[2],
        locked_cohort_case_count=3,
        minimum_evaluable_case_count=3,
        minimum_supported_case_count=0,
        minimum_supported_fraction=0.0,
    )
    assert out["execution_status"] == "PASS"
    assert out["noise_ceiling_pass"]
    assert len(out["per_case"]) == 3


def test_core_preflight_writes_legacy_manifest(tmp_path):
    import yaml

    cfg = yaml.safe_load((ROOT / "conf/m1/cf1s_core_smoke.yaml").read_text(encoding="utf-8"))
    # Path/boolean-only preflight is no longer sufficient for authorization.
    with pytest.raises(CoreContractError, match="measured checks missing"):
        core_preflight(cfg, root=ROOT, apply_runtime=False)
    legacy = ROOT / "outputs/cf1s/CF1S_LEGACY_PATH_MANIFEST.json"
    assert legacy.exists()
    doc = json.loads(legacy.read_text(encoding="utf-8"))
    assert doc["status"] == "FROZEN_REFERENCE_ONLY"
    ready = json.loads(
        (ROOT / "outputs/cf1s/CF1S_PRODUCTION_READINESS.json").read_text(encoding="utf-8")
    )
    assert ready.get("deprecated_for_core_authorization") is True
