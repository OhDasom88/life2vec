#!/usr/bin/env python3
"""Run CF-1S fixture-contract smoke or production development smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _synthetic_events(feature_ids: List[str], n_times: int = 6) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for fi, feature in enumerate(feature_ids):
        for t in range(n_times):
            events.append(
                {
                    "event_id": f"E{fi}_{t}",
                    "feature_id": feature,
                    "feature": feature,
                    "mg_id": f"MG{fi}_{t}",
                    "event_time": f"2025-01-01T{t:02d}:00:00",
                    "event_time_epoch": float(t * 3600),
                    "same_time_group_id": f"2025-01-01T{t:02d}:00:00",
                    "observed_raw": float((t + fi) % 2),
                    "raw_reconstructable": True,
                    "atomic_schema_edit_possible": True,
                    "token_indices": [fi * 100 + t],
                    "member_token_count": 1,
                    "absolute_saliency": float(10 - t) + 0.1 * fi,
                    "signed_saliency": float(10 - t),
                    "fold_stable": t < 3,
                    "method_stable": True,
                    "perturbation_stable": True,
                    "saliency_evaluable": True,
                }
            )
            if t == 0:
                events.append(
                    {
                        **events[-1],
                        "token_indices": [fi * 100 + t, fi * 100 + t + 50],
                        "member_token_count": 2,
                        "absolute_saliency": 0.01,
                    }
                )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf1s_smoke.yaml")
    parser.add_argument(
        "--mode",
        choices=["fixture", "production"],
        default="fixture",
        help="fixture=contract smoke with synthetic events; production=real data bridge",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-legacy-fixture-only",
        action="store_true",
        help="Required. Legacy broad path is FROZEN_REFERENCE_ONLY and cannot issue official results.",
    )
    args = parser.parse_args()

    if not args.allow_legacy_fixture_only:
        raise SystemExit(
            "CF-1S legacy runner is FROZEN_REFERENCE_ONLY. "
            "Pass --allow-legacy-fixture-only for fixture regression only, "
            "or use scripts/online2_v2/v03/run_cf1s_core_v03.py for official Core runs."
        )
    if args.mode == "production":
        raise SystemExit(
            "Legacy production mode is blocked. Official production uses "
            "run_cf1s_core_v03.py only; legacy readiness cannot authorize Core."
        )

    from src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s import (
        load_yaml_policy,
        lock_manifest,
        run_cf1s_case_fixture,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.status import overall_cf1s_status

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = _load_yaml(cfg_path)
    edit = load_yaml_policy(ROOT / cfg["edit_policy_path"])
    search = load_yaml_policy(ROOT / cfg["search_policy_path"])
    saliency = load_yaml_policy(ROOT / cfg["saliency_policy_path"])
    acceptance = load_yaml_policy(ROOT / cfg["acceptance_policy_path"])
    cases = list(cfg.get("development_cases") or [])
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    package_name = (
        f"CF1S_FIXTURE_CONTRACT_SMOKE_{ts}"
        if args.mode == "fixture"
        else f"CF1S_PRODUCTION_DEV_SMOKE_{ts}"
    )
    out_dir = args.out or (ROOT / "outputs/cf1s" / package_name)
    out_dir.mkdir(parents=True, exist_ok=False)

    per_case = []
    if args.mode == "fixture":
        whitelist = list((edit.get("primary_control_like_whitelist") or [])[:3]) or [
            "circulation_fan",
            "fcu_fan",
            "co2_supply",
        ]
        for case_id in cases:
            result = run_cf1s_case_fixture(
                case_id=case_id,
                events=_synthetic_events(whitelist),
                edit_policy=edit,
                search_policy=search,
                saliency_policy=saliency,
                acceptance_policy=acceptance,
                arm="CONTROL_LIKE",
                cutoff_time="2025-01-01T05:00:00",
                model_effects_evaluated=False,
                root=ROOT,
            )
            case_path = out_dir / "per_case" / f"{case_id}.json"
            case_path.parent.mkdir(parents=True, exist_ok=True)
            case_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            per_case.append(result)

    parent = ROOT / str(cfg.get("parent_cf0r_correction_package") or "")
    parent_sha = ""
    if parent.exists():
        checksums = parent / "CHECKSUMS.sha256"
        if checksums.exists():
            parent_sha = _sha_file(checksums)

    manifest = lock_manifest(
        root=ROOT,
        parent_cf0r_correction_sha256=parent_sha,
        fixture_contract_snapshot=(args.mode == "fixture"),
        final_primary_lock=False,
    )

    # Cohort status: aggregate development cases, never first-case-only.
    statuses = [
        c.get("primary_multi_event_edit_feasibility_status") for c in per_case
    ]
    if not statuses:
        dev_status = "NOT_EVALUABLE"
    elif all(s == "NOT_EVALUABLE" for s in statuses):
        dev_status = "NOT_EVALUABLE"
    elif any(s == "SUPPORTED" for s in statuses):
        dev_status = "SUPPORTED"
    elif any(s == "LIMITED" for s in statuses):
        dev_status = "LIMITED"
    elif any(s == "NOT_SUPPORTED" for s in statuses):
        dev_status = "NOT_SUPPORTED"
    else:
        dev_status = "NOT_EVALUABLE"

    cohort = overall_cf1s_status(
        problem20_blind_status="NOT_EVALUABLE",
        primary_internal_validation_32_status="NOT_EVALUABLE",
        development_3_status=dev_status,
        problem20_min_evaluable_met=False,
    )
    summary = {
        "mode": args.mode,
        "package_role": "CF1S_LEGACY_FIXTURE_REFERENCE_ONLY",
        "legacy_broad_path_status": "FROZEN_REFERENCE_ONLY",
        "official_authorization": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(per_case),
        "development_only": True,
        "primary_execution_allowed": False,
        "problem20_execution_allowed": False,
        "final_primary_lock": False,
        "fixture_contract_snapshot": True,
        "manifest": manifest,
        "cohort_status": cohort,
        "development_case_statuses": statuses,
        "action_authorization": False,
        "recommendation_authorization": False,
        "cf1b_authorization": False,
        "case_ids": cases,
        "replacement_runner": "scripts/online2_v2/v03/run_cf1s_core_v03.py",
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "config_snapshot.yaml").write_text(
        cfg_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # Persist lock manifests
    for name in ("CODE_MANIFEST", "POLICY_MANIFEST", "TEST_MANIFEST"):
        (out_dir / f"{name}.json").write_text(
            json.dumps(manifest[name], indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": summary["package_role"] + "_COMPLETE",
                "out": str(out_dir),
                "mode": args.mode,
                "primary_32_allowed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
