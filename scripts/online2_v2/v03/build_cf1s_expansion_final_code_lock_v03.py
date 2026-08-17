#!/usr/bin/env python3
"""Build EXPANSION_FINAL_CODE_LOCK (E-LOCK): the code/test/policy/node-ID lock
fixed after Development3 FINAL PASS + cohort-aware verifier/lock/auth/runner
implementation + cross-cohort (§7) failure injection + full regression.
Required before Validation20/Primary32 qualification/pilot may begin."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-root",
        type=Path,
        default=ROOT / "tests/v2/counterfactual_cf1s",
    )
    parser.add_argument(
        "--development3-final-code-lock",
        type=Path,
        default=ROOT / "outputs/cf1s_core/locks/DEVELOPMENT3_FINAL_CODE_LOCK.json",
    )
    parser.add_argument(
        "--development3-final-verdict",
        type=Path,
        default=ROOT
        / "outputs/cf1s_core/sidecars/development3-20260719T142507Z/FINAL_VERDICT.json",
    )
    parser.add_argument(
        "--node-id-manifest-out",
        type=Path,
        default=ROOT / "outputs/cf1s_core/QUALIFICATION_TEST_NODE_IDS.json",
    )
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        default=ROOT / "outputs/cf1s_core/qualification",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/cf1s_core/locks/EXPANSION_FINAL_CODE_LOCK.json",
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
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
        default_code_tree_paths,
        default_policy_tree_paths,
        default_test_tree_paths,
        tree_manifest_sha,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_verifier import (
        compute_verifier_code_sha256,
    )

    # 0. Prerequisite: Development3 FINAL must be all-PASS.
    dev3_lock_path = (
        args.development3_final_code_lock
        if args.development3_final_code_lock.is_absolute()
        else ROOT / args.development3_final_code_lock
    )
    dev3_verdict_path = (
        args.development3_final_verdict
        if args.development3_final_verdict.is_absolute()
        else ROOT / args.development3_final_verdict
    )
    if not dev3_lock_path.is_file():
        raise CoreContractError(f"DEVELOPMENT3_FINAL_CODE_LOCK missing: {dev3_lock_path}")
    if not dev3_verdict_path.is_file():
        raise CoreContractError(f"Development3 FINAL_VERDICT missing: {dev3_verdict_path}")
    dev3_verdict = json.loads(dev3_verdict_path.read_text(encoding="utf-8"))
    if dev3_verdict.get("verdict_kind") != "CF1S_FINAL_VERIFIER_VERDICT_V1":
        raise CoreContractError("not a canonical Development3 FINAL verdict")
    gates = dev3_verdict.get("gate_results") or {}
    if set(gates) != {f"G{i}" for i in range(1, 13)} or any(
        v != "PASS" for v in gates.values()
    ):
        raise CoreContractError(
            "E-LOCK requires Development3 FINAL G1-G12 all PASS; refusing"
        )
    if not dev3_verdict.get("final_report_allowed"):
        raise CoreContractError("Development3 FINAL final_report_allowed=false")
    dev3_lock = json.loads(dev3_lock_path.read_text(encoding="utf-8"))

    test_root = args.test_root if args.test_root.is_absolute() else ROOT / args.test_root

    # 1. Canonical qualification node-ID manifest (file-level).
    test_paths = [p for p in default_test_tree_paths(ROOT) if p.is_file()]
    node_manifest = {
        "version": "CF1S_TEST_NODE_ID_MANIFEST_V1",
        "nodes": sorted(str(p.relative_to(ROOT)) for p in test_paths),
    }
    node_manifest_path = (
        args.node_id_manifest_out
        if args.node_id_manifest_out.is_absolute()
        else ROOT / args.node_id_manifest_out
    )
    node_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    node_manifest_path.write_text(
        canonical_json_dumps(node_manifest) + "\n", encoding="utf-8"
    )

    # 2. Real full regression, including §7 cross-cohort failure injection tests.
    qual_dir = (
        args.qualification_dir
        if args.qualification_dir.is_absolute()
        else ROOT / args.qualification_dir
    )
    qual_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    junit_path = qual_dir / f"expansion_qualification_junit_{ts}.xml"
    log_path = qual_dir / f"expansion_qualification_pytest_{ts}.log"

    collect = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_root), "--collect-only", "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    canonical_test_selection = sorted(
        line.strip() for line in collect.stdout.splitlines() if "::" in line
    )
    if not canonical_test_selection:
        raise CoreContractError("qualification collection produced no test node IDs")
    if "tests/v2/counterfactual_cf1s/test_cf1s_cohort_isolation_v2.py::test_dev3_authorization_rejected_by_validation20_consumer" not in canonical_test_selection:
        raise CoreContractError(
            "required §7 cross-cohort failure-injection tests missing from selection"
        )

    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_root), "-q", f"--junitxml={junit_path}"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    log_path.write_text(run.stdout + run.stderr, encoding="utf-8")
    if not junit_path.is_file():
        raise CoreContractError("qualification run did not produce JUnit XML result")

    root_el = ET.parse(junit_path).getroot()
    suite_el = root_el if root_el.tag == "testsuite" else root_el.find("testsuite")
    if suite_el is None:
        raise CoreContractError("qualification JUnit XML missing <testsuite>")
    total = int(suite_el.get("tests", "0"))
    failed = int(suite_el.get("failures", "0")) + int(suite_el.get("errors", "0"))
    passed = total - failed - int(suite_el.get("skipped", "0"))
    exit_code = run.returncode
    if exit_code != 0 or failed != 0:
        raise CoreContractError(
            f"qualification regression FAILED: exit_code={exit_code} "
            f"failed={failed} total={total}; refusing to build E-LOCK"
        )

    junit_sha = sha256_file(junit_path)
    node_manifest_sha = sha256_file(node_manifest_path)
    canonical_test_selection_sha = canonical_json_sha256(canonical_test_selection)

    code_sha, _ = tree_manifest_sha(default_code_tree_paths(ROOT), root=ROOT, required=True)
    test_sha, _ = tree_manifest_sha(default_test_tree_paths(ROOT), root=ROOT, required=True)
    policy_sha, _ = tree_manifest_sha(default_policy_tree_paths(ROOT), root=ROOT, required=True)
    verifier_code_sha = compute_verifier_code_sha256()

    lock = {
        "artifact_kind": "EXPANSION_FINAL_CODE_LOCK",
        "schema_version": "CF1S_FINAL_CODE_LOCK_V1",
        "scope": ["DEVELOPMENT3", "VALIDATION20", "PRIMARY32"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_development3_final_code_lock_sha256": dev3_lock["final_code_lock_sha256"],
        "source_development3_final_verdict_sha256": dev3_verdict["verdict_sha256"],
        "code_tree_sha256": code_sha,
        "test_tree_sha256": test_sha,
        "policy_tree_sha256": policy_sha,
        "verifier_code_sha256": verifier_code_sha,
        "qualification_node_id_manifest_sha256": node_manifest_sha,
        "qualification_node_id_manifest_relpath": str(
            node_manifest_path.relative_to(ROOT)
        ),
        "canonical_test_selection_sha256": canonical_test_selection_sha,
        "canonical_test_selection_count": len(canonical_test_selection),
        "qualification_result": {
            "exit_code": exit_code,
            "total_count": total,
            "passed_count": passed,
            "failed_count": failed,
        },
        "qualification_junit_sha256": junit_sha,
        "qualification_junit_relpath": str(junit_path.relative_to(ROOT)),
        "execution_scope": "TWO_EVENT_ONLY",
        "three_event_execution_status": "OUT_OF_SCOPE",
    }
    lock["final_code_lock_sha256"] = canonical_json_sha256(lock)

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in existing.items() if k != "created_at"}
        comparable_new = {k: v for k, v in lock.items() if k != "created_at"}
        if comparable != comparable_new:
            raise CoreContractError(
                f"existing E-LOCK differs from freshly-computed lock; refuse overwrite: {out}"
            )
        print(json.dumps({"status": "EXISTING_VALID", "out": str(out)}, indent=2))
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(lock) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "out": str(out),
                "final_code_lock_sha256": lock["final_code_lock_sha256"],
                "qualification_result": lock["qualification_result"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
