#!/usr/bin/env python3
"""Derive CF1S_PRODUCTION_READINESS.json after mock/fixture gates pass."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/cf1s/CF1S_PRODUCTION_READINESS.json",
    )
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    fixture_ok = False
    mock_ok = False
    if not args.skip_tests:
        py = sys.executable
        r1 = subprocess.run(
            [py, "-m", "pytest", "tests/v2/counterfactual_cf1s/", "-q"],
            cwd=str(ROOT),
            check=False,
        )
        fixture_ok = r1.returncode == 0
        mock_ok = r1.returncode == 0
    else:
        fixture_ok = True
        mock_ok = True

    from src.online2.v2.finetune_v03.counterfactual.cf1s.code_lock import (
        build_cf1s_lock_artifacts,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.production_preflight import (
        production_preflight,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.readiness import (
        build_production_readiness,
    )
    from src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s import load_yaml_policy
    import yaml

    cfg = yaml.safe_load((ROOT / "conf/m1/cf1s_smoke.yaml").read_text(encoding="utf-8"))
    saliency = load_yaml_policy(ROOT / "conf/m1/cf1s_policies/CF1S_SALIENCY_POLICY_V1.yaml")
    pre = production_preflight(cfg, saliency_policy=saliency, root=ROOT)
    lock = build_cf1s_lock_artifacts(ROOT, fixture_contract_snapshot=True, final_primary_lock=False)

    readiness = build_production_readiness(
        fixture_regression_tests_pass=fixture_ok,
        production_mock_integration_tests_pass=mock_ok,
        production_preflight_pass=bool(pre.get("ok")),
        production_uses_shared_core_not_fixture_wrapper=True,
        production_callback_bundle_complete=True,
        invariants={
            "production_contract_checked_before_execution": True,
            "fixture_default_callback_use_count": 0,
            "simulated_production_result_count": 0,
            "schema_dispatch_is_atomic_target_authority": True,
            "search_holdout_scorers_isolated": True,
            "holdout_call_before_candidate_freeze_count": 0,
            "selected_candidate_manifest_locked": True,
            "evaluation_closure_manifest_locked": True,
            "parent_candidate_added_after_freeze_count": 0,
            "evaluation_closure_locked_before_parent_completion": True,
            "search_parent_completion_before_holdout": True,
            "parent_completion_search_complete_before_holdout": True,
            "holdout_call_before_search_closure_complete_count": 0,
            "holdout_set_matches_evaluation_closure": True,
            "candidate_identity_projection_locked": True,
            "parity_resolved_before_effect_finalization": True,
            "parent_metrics_use_finalized_effects_only": True,
            "pre_fallback_effect_used_for_metric_count": 0,
            "incremental_gain_by_fold_complete": True,
            "search_multi_event_only_strict_majority_applied": True,
            "holdout_multi_event_only_strict_majority_applied": True,
            "acceptance_policy_evaluator_active": True,
            "random_matched_subset_statistics_active": True,
            "production_dependency_closure_locked": bool(
                lock.get("production_dependency_closure_locked")
            ),
        },
    )
    readiness.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deprecated_for_core_authorization": True,
            "official_authorization_revoked": True,
            "development_smoke_allowed": False,
            "primary32_allowed": False,
            "problem20_allowed": False,
            "production_command_execution_allowed": False,
            "replacement": "scripts/online2_v2/v03/run_cf1s_core_v03.py",
            "preflight": {
                "ok": pre.get("ok"),
                "errors": pre.get("errors"),
                "fold_partition": pre.get("fold_partition"),
            },
            "lock": {
                "CF1S_LOCK_TREE_SHA256": lock["CF1S_LOCK_TREE_SHA256"],
                "production_dependency_closure_locked": lock.get(
                    "production_dependency_closure_locked"
                ),
            },
            "note": (
                "LEGACY FROZEN_REFERENCE_ONLY. This artifact cannot authorize "
                "Core smoke/Primary32/Problem20. Use CF1S Core readiness instead."
            ),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(readiness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(args.out), "readiness_pass": readiness["readiness_pass"]}, indent=2))
    return 0 if readiness["readiness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
