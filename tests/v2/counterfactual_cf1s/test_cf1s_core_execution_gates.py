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


def test_primary32_and_problem20_blocked_without_authorization():
    py = sys.executable
    r = subprocess.run(
        [
            py,
            str(ROOT / "scripts/online2_v2/v03/run_cf1s_core_v03.py"),
            "--cohort",
            "primary32",
            "--mode",
            "contract",
            "--skip-preflight",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
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
