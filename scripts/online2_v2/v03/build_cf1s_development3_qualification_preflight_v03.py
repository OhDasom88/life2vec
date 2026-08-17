#!/usr/bin/env python3
"""Build the DEVELOPMENT3 qualification preflight artifact — the 6
REQUIRED_MEASURED_CHECKS derived from real qualification pytest results
(JUnit XML) plus the real Search-baseline calibration measurement, never
invented values. Consumed as pre_stable_lock["artifact_sources"]
["qualification_preflight_sha256"]."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]

# (check_name, junit test name, junit classname, source test file, producer module)
TEST_DERIVED_CHECKS = [
    (
        "cold_rebuild_equivalence",
        "test_cold_build_equivalence_membership_and_tensors",
        "tests.v2.counterfactual_cf1s.test_cf1s_core_execution_gates",
        "tests/v2/counterfactual_cf1s/test_cf1s_core_execution_gates.py",
        "src/online2/v2/finetune_v03/counterfactual/cf1s/core_production.py",
    ),
    (
        "official_batch_field_parity",
        "test_stale_batch_field_parity_fails",
        "tests.v2.counterfactual_cf1s.test_cf1s_core_production_mock",
        "tests/v2/counterfactual_cf1s/test_cf1s_core_production_mock.py",
        "src/online2/v2/finetune_v03/counterfactual/cf1s/core_production.py",
    ),
    (
        "stage_a_critic_pairing",
        "test_two_phase_verifier_end_to_end",
        "tests.v2.counterfactual_cf1s.test_cf1s_verifier_contract_v2",
        "tests/v2/counterfactual_cf1s/test_cf1s_verifier_contract_v2.py",
        "src/online2/v2/finetune_v03/counterfactual/cf1s/core_verifier.py",
    ),
    (
        "fold_isolation",
        "test_pure_repeat_noise_and_forced_identity_all_folds",
        "tests.v2.counterfactual_cf1s.test_cf1s_core_family_identity",
        "tests/v2/counterfactual_cf1s/test_cf1s_core_family_identity.py",
        "src/online2/v2/finetune_v03/counterfactual/cf1s/core_trace.py",
    ),
    (
        "risk_logit_contract",
        "test_observation_median_never_mixes_risk_and_logit",
        "tests.v2.counterfactual_cf1s.test_cf1s_verifier_contract_v2",
        "tests/v2/counterfactual_cf1s/test_cf1s_verifier_contract_v2.py",
        "src/online2/v2/finetune_v03/counterfactual/cf1s/core_scientific.py",
    ),
]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/cf1s_core/qualification/CF1S_DEVELOPMENT3_QUALIFICATION_PREFLIGHT.json",
    )
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
        canonical_json_sha256,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        CoreContractError,
        sha256_file,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_preflight import (
        validate_measured_preflight_contract,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )

    junit_path = args.junit_xml if args.junit_xml.is_absolute() else ROOT / args.junit_xml
    calib_path = (
        args.calibration_artifact
        if args.calibration_artifact.is_absolute()
        else ROOT / args.calibration_artifact
    )
    if not junit_path.is_file():
        raise CoreContractError(f"qualification JUnit XML missing: {junit_path}")
    if not calib_path.is_file():
        raise CoreContractError(f"calibration artifact missing: {calib_path}")

    root_el = ET.parse(junit_path).getroot()
    suite_el = root_el if root_el.tag == "testsuite" else root_el.find("testsuite")
    if suite_el is None:
        raise CoreContractError("qualification JUnit XML missing <testsuite>")
    testcases = {
        (tc.get("classname"), tc.get("name")): tc for tc in suite_el.iter("testcase")
    }

    checks = []
    for check_name, test_name, classname, test_relpath, producer_relpath in TEST_DERIVED_CHECKS:
        tc = testcases.get((classname, test_name))
        if tc is None:
            raise CoreContractError(
                f"required preflight evidence test not found in qualification run: "
                f"{classname}::{test_name}"
            )
        if tc.find("failure") is not None or tc.find("error") is not None:
            raise CoreContractError(
                f"required preflight evidence test did not pass: {classname}::{test_name}"
            )
        test_path = ROOT / test_relpath
        producer_path = ROOT / producer_relpath
        checks.append(
            {
                "check_name": check_name,
                "target_sha256": sha256_file(test_path),
                "producer_code_sha256": sha256_file(producer_path),
                "tolerance": 0.0,
                "status": "PASS",
                "observations": {
                    "evidence_kind": "QUALIFICATION_JUNIT_TESTCASE",
                    "junit_relpath": str(junit_path.relative_to(ROOT)),
                    "junit_classname": classname,
                    "junit_test_name": test_name,
                    "junit_time_seconds": float(tc.get("time") or 0.0),
                    "junit_outcome": "passed",
                },
            }
        )

    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check_name": "deterministic_runtime",
            "target_sha256": sha256_file(
                ROOT
                / "src/online2/v2/finetune_v03/counterfactual/cf1s/core_runtime.py"
            ),
            "producer_code_sha256": sha256_file(calib_path),
            "tolerance": 0.0,
            "status": "PASS" if calib.get("cohort_max_search_baseline_pure_repeat_risk_noise") == 0.0 else "FAIL",
            "observations": {
                "evidence_kind": "MEASURED_SEARCH_BASELINE_REPEAT_NOISE",
                "calibration_artifact_relpath": str(calib_path.relative_to(ROOT)),
                "calibration_artifact_sha256": calib["calibration_artifact_sha256"],
                "cohort_max_search_baseline_pure_repeat_risk_noise": calib[
                    "cohort_max_search_baseline_pure_repeat_risk_noise"
                ],
                "cohort_max_search_identity_reconstruction_error": calib[
                    "cohort_max_search_identity_reconstruction_error"
                ],
                "case_ids": calib["case_ids"],
            },
        }
    )

    validated = validate_measured_preflight_contract(checks)
    artifact = {
        "artifact_kind": "CF1S_DEVELOPMENT3_QUALIFICATION_PREFLIGHT_V1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": validated,
        "checks_sha256": canonical_json_sha256(validated),
    }
    artifact["qualification_preflight_sha256"] = canonical_json_sha256(artifact)

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in existing.items() if k != "created_at"}
        comparable_new = {k: v for k, v in artifact.items() if k != "created_at"}
        if comparable != comparable_new:
            raise CoreContractError(f"existing preflight differs; refuse overwrite: {out}")
        print(json.dumps({"status": "EXISTING_VALID", "out": str(out)}, indent=2))
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(artifact) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "out": str(out),
                "qualification_preflight_sha256": artifact["qualification_preflight_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
