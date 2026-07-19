#!/usr/bin/env python3
"""CF-1S Core official runner — development / primary32 / problem20 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf1s_core_smoke.yaml")
    parser.add_argument(
        "--cohort",
        choices=["development", "primary32", "problem20"],
        default="development",
    )
    parser.add_argument(
        "--mode",
        choices=["contract", "production"],
        default="contract",
        help="contract=pure acceptance/identity dry-run; production=real cold-rebuild path",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="for unit tests only; production/smoke must not use this",
    )
    parser.add_argument(
        "--allow-development-smoke",
        action="store_true",
        help="explicitly allow development smoke after tests/equivalence/preflight evidence",
    )
    args = parser.parse_args()

    # Generic production is never served by this runner — block before any output path.
    if args.mode != "contract":
        raise SystemExit(
            "generic production mode blocked on run_cf1s_core_v03.py before output creation; "
            "use run_cf1s_core_development_production_v03.py with authorization artifact"
        )

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = _load_yaml(cfg_path)
    core_flags = cfg.get("cf1s_core") or {}

    if args.cohort == "primary32":
        from src.online2.v2.finetune_v03.counterfactual.cf1s.core_readiness import (
            assert_primary32_authorized,
        )

        lock = (
            ROOT
            / "outputs/cf1s_core/CF1S_PRIMARY32_EXECUTION_AUTHORIZATION.json"
        )
        assert_primary32_authorized(lock)
        if not core_flags.get("primary32_allowed") and not json.loads(lock.read_text()).get(
            "primary32_execution_authorized"
        ):
            raise SystemExit("primary32_allowed=false")

    if args.cohort == "problem20":
        # Must not receive sealed label path
        if "problem20_labels_sealed_path" in sys.argv or any(
            "LABELS_SEALED" in a for a in sys.argv
        ):
            raise SystemExit("Problem20 blind runner must not receive sealed label path")
        marker = ROOT / "outputs/cf1s_core/CF1S_PROBLEM20_AUTHORIZATION.json"
        if not marker.exists() or not json.loads(marker.read_text()).get(
            "problem20_execution_authorized"
        ):
            raise SystemExit("problem20 not authorized")

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_preflight import (
        core_preflight,
        write_core_preflight,
    )
    from src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s_core import (
        run_cf1s_core_cohort_contract,
    )

    if not args.skip_preflight:
        preflight = core_preflight(cfg, root=ROOT, apply_runtime=True)
        preflight_path = ROOT / "outputs/cf1s_core/CF1S_CORE_PREFLIGHT.json"
        write_core_preflight(preflight, preflight_path)
    else:
        preflight_path = ROOT / "outputs/cf1s_core/CF1S_CORE_PREFLIGHT.json"

    if args.cohort == "development":
        man = _load_manifest(ROOT / str(cfg["development_manifest_path"]))
        locked = 3
        min_eval = 3
        min_sup = 0
        min_frac = 0.0
    elif args.cohort == "primary32":
        man = _load_manifest(ROOT / str(cfg["primary32_manifest_path"]))
        locked, min_eval, min_sup, min_frac = 32, 24, 8, 0.25
    else:
        man = _load_manifest(ROOT / str(cfg["problem20_manifest_path"]))
        locked, min_eval, min_sup, min_frac = 20, 15, 5, 0.25
        # Ensure no ground-truth values
        for c in man.get("cases") or []:
            if "label" in c:
                raise SystemExit("problem20 runtime manifest leaked label values")

    case_ids = list(man["ordered_case_ids"])
    critic = cfg.get("critic") or {}
    search_folds = list(critic.get("search_fold_ids") or [0, 1])
    holdout_folds = list(critic.get("holdout_fold_ids") or [2])

    # Contract mode only reaches here.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or (
        ROOT / "outputs/cf1s_core" / "contract" / f"CF1S_CORE_{args.cohort.upper()}_CONTRACT_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    cohort_result = run_cf1s_core_cohort_contract(
        case_ids=case_ids,
        search_fold_ids=search_folds,
        holdout_fold_ids=holdout_folds,
        locked_cohort_case_count=locked,
        minimum_evaluable_case_count=min_eval,
        minimum_supported_case_count=min_sup,
        minimum_supported_fraction=min_frac,
    )

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_output import (
        build_contract_case_envelope,
        build_runner_summary,
        write_case_proposal_json,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_edit_proposal import (
        ExecutionReadiness,
        EvaluationCoverage,
    )

    per_dir = out_dir / "per_case"
    per_dir.mkdir(parents=True, exist_ok=True)
    envelopes = []
    for case in cohort_result["per_case"]:
        (per_dir / f"{case['case_id']}.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        env = build_contract_case_envelope(case["case_id"])
        env["contract_fixture_expected_status"] = case.get(
            "primary_multi_event_edit_feasibility_status"
        )
        env["contract_validation_status"] = case.get("execution_status") or case.get(
            "case_status"
        )
        write_case_proposal_json(env, per_dir / f"{case['case_id']}.edit_proposals.json")
        envelopes.append(env)

    proposal_summary = build_runner_summary(
        case_envelopes=envelopes,
        execution_readiness=ExecutionReadiness.BLOCKED,
        evaluation_coverage=EvaluationCoverage.NONE,
        scientifically_evaluable_case_count=0,
        locked_cohort_case_count=locked,
    )

    summary = {
        "runner": "run_cf1s_core_v03.py",
        "cohort": args.cohort,
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_ids": case_ids,
        "execution_status": cohort_result["execution_status"],
        "case_scientific_statuses": cohort_result["case_scientific_statuses"],
        "thresholds": cohort_result["thresholds"],
        "noise_ceiling_pass": cohort_result["noise_ceiling_pass"],
        "cohort_result": cohort_result["cohort"],
        "primary32_allowed": False,
        "problem20_allowed": False,
        "final_primary_lock": False,
        "preflight_path": str(preflight_path),
        "config_sha256": _sha_file(cfg_path),
        "reencode_mode": "FULL_SEQUENCE_COLD_REBUILD",
        "selector": "SALIENCY_TOP_K",
        "arm": "EDITABLE_CONTROL_LIKE",
        "edit_proposal_summary": proposal_summary,
    }
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "config_snapshot.yaml").write_text(
        cfg_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(json.dumps({"status": "CF1S_CORE_RUN_COMPLETE", "out": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
