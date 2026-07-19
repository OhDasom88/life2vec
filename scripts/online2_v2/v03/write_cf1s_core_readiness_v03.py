#!/usr/bin/env python3
"""Write evidence-based CF1S_CORE_DEVELOPMENT_SMOKE_READINESS.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _tree_sha(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        if p.is_file():
            h.update(p.name.encode())
            h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--pytest-log", type=Path, required=True)
    parser.add_argument(
        "--cold-equivalence",
        type=Path,
        default=ROOT / "outputs/cf1s_core/CF1S_CORE_COLD_REBUILD_EQUIVALENCE.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/cf1s_core/CF1S_CORE_DEVELOPMENT_SMOKE_READINESS.json",
    )
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        apply_runtime_lock,
        required_observed_gate,
        sha256_file,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_readiness import (
        build_development_smoke_readiness,
    )

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
    cold = {"required_value": True, "observed_value": None, "evidence_artifact": None}
    if args.cold_equivalence.exists():
        cold_doc = json.loads(args.cold_equivalence.read_text(encoding="utf-8"))
        cold = required_observed_gate(
            required_value=True,
            observed_value=bool(cold_doc.get("cold_rebuild_equivalence_test_pass")),
            evidence_artifact=str(args.cold_equivalence),
            evidence_sha256=hashlib.sha256(args.cold_equivalence.read_bytes()).hexdigest(),
        )

    preflight_path = Path(args.preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    runtime_state = apply_runtime_lock(seed=0)
    preflight["runtime_lock"] = runtime_state
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    runtime = required_observed_gate(
        required_value=True,
        observed_value=bool(runtime_state.get("lock_satisfied")),
        evidence_artifact=str(preflight_path),
        evidence_sha256=sha256_file(preflight_path),
    )
    preflight["deterministic_runtime_state_verified"] = runtime
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Refresh evidence sha after final preflight write.
    runtime["evidence_sha256"] = sha256_file(preflight_path)
    preflight["deterministic_runtime_state_verified"] = runtime
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    code_sha = _tree_sha(
        list((ROOT / "src/online2/v2/finetune_v03/counterfactual/cf1s").glob("core_*.py"))
        + [ROOT / "src/online2/v2/finetune_v03/counterfactual/pipeline_cf1s_core.py"]
    )
    policy_sha = _tree_sha(
        [
            ROOT / "conf/m1/cf1s_policies/CF1S_CORE_ACCEPTANCE_POLICY.yaml",
            ROOT / "conf/m1/cf1s_core_smoke.yaml",
        ]
    )
    readiness = build_development_smoke_readiness(
        execution_status=str(summary.get("execution_status") or "FAIL"),
        case_scientific_statuses=list(summary.get("case_scientific_statuses") or []),
        noise_ceiling_pass=bool(summary.get("noise_ceiling_pass")),
        preflight_path=preflight_path,
        test_log_path=Path(args.pytest_log),
        package_dir=run_dir,
        code_sha256=code_sha,
        policy_sha256=policy_sha,
        cold_rebuild_equivalence=cold,
        runtime_verified=runtime,
    )
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(readiness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "readiness": readiness["development_readiness_status"],
                "readiness_issued": readiness.get("readiness_issued"),
                "primary32_allowed": False,
            },
            indent=2,
        )
    )
    return 0 if readiness.get("readiness_issued") else 1


if __name__ == "__main__":
    raise SystemExit(main())
