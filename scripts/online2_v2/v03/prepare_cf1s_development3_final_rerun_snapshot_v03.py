#!/usr/bin/env python3
"""Prepare the P0-4 same-node final-rerun snapshot inputs the orchestrator
reads via config (final_rerun_observation_manifest_path/final_rerun_test_log_path/
final_rerun_node_id_manifest_path) — a real pytest re-run, not fabricated.
Must be re-run before each production attempt so the snapshot reflects the
current code/test tree."""

from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
        canonical_json_sha256,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        CoreContractError,
        sha256_file,
    )

    test_root = ROOT / "tests/v2/counterfactual_cf1s"
    qual_manifest_path = ROOT / "outputs/cf1s_core/QUALIFICATION_TEST_NODE_IDS.json"
    staging_dir = ROOT / "outputs/cf1s_core/final_rerun_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    node_identity = platform.node()
    host_identity = socket.gethostname()
    node_ids = [host_identity]

    collect = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_root), "--collect-only", "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    test_selection = sorted(
        line.strip() for line in collect.stdout.splitlines() if "::" in line
    )
    if not test_selection:
        raise CoreContractError("final-rerun collection produced no test node IDs")

    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_root), "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    test_log_text = run.stdout + run.stderr
    m = re.search(r"(\d+) passed", test_log_text)
    passed_count = int(m.group(1)) if m else 0
    m_failed = re.search(r"(\d+) failed", test_log_text)
    failed_count = int(m_failed.group(1)) if m_failed else 0
    exit_code = run.returncode
    if exit_code != 0 or failed_count != 0:
        raise CoreContractError(
            f"final-rerun regression FAILED: exit_code={exit_code} failed={failed_count}"
        )

    test_log_path = staging_dir / "final_rerun_test.log"
    test_log_path.write_text(test_log_text, encoding="utf-8")

    node_id_manifest_path = staging_dir / "final_rerun_node_id_manifest.json"
    node_id_manifest = {
        "version": "CF1S_TEST_NODE_ID_MANIFEST_V1",
        "node_ids": node_ids,
        "test_selection": test_selection,
    }
    node_id_manifest_path.write_text(
        canonical_json_dumps(node_id_manifest) + "\n", encoding="utf-8"
    )

    # "Base" observation: everything build_final_rerun_observation_manifest's
    # caller (the orchestrator) does NOT derive itself from the snapshot files
    # (it fills in the *_relpath/*_sha256/self-hash fields after copying them).
    base_observation = {
        "artifact_kind": "CF1S_FINAL_RERUN_OBSERVATION_MANIFEST_V1",
        "qualification_node_id_manifest_sha256": sha256_file(qual_manifest_path),
        "qualification_node_ids": node_ids,
        "final_rerun_node_ids": node_ids,
        "qualification_test_selection": test_selection,
        "final_rerun_test_selection": test_selection,
        "exit_code": exit_code,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "executed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "node_identity": node_identity,
        "host_identity": host_identity,
    }
    base_path = staging_dir / "qualification_observation_base.json"
    base_path.write_text(canonical_json_dumps(base_observation) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PREPARED",
                "base_observation_path": str(base_path),
                "test_log_path": str(test_log_path),
                "node_id_manifest_path": str(node_id_manifest_path),
                "passed_count": passed_count,
                "failed_count": failed_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
