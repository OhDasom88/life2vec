#!/usr/bin/env python3
"""Run counterfactual_m1 suite and atomically bind source hashes to log + JUnit.

Writes reports/p0_evidence_binding/test_run_manifest.json then (optionally) gate + STATUS.
Fixed argv only — no suite shrink / filter flags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.counterfactual.evaluation.p0_evidence_binding_gate import (  # noqa: E402
    DEFAULT_GATE_REL,
    DEFAULT_STATUS_REL,
    DEFAULT_TEST_RUN_MANIFEST_REL,
    write_p0_evidence_binding_gate,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.source_lock import (  # noqa: E402
    SCOPE_VERSION_V2,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.test_run_manifest import (  # noqa: E402
    FIXED_JUNIT_REL,
    FIXED_LOG_REL,
    build_test_run_manifest,
    current_dependency_tree_sha256,
    fixed_pytest_command,
    write_test_run_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--pytest-log",
        type=Path,
        default=None,
        help=f"Default: {FIXED_LOG_REL}",
    )
    p.add_argument(
        "--junit-xml",
        type=Path,
        default=None,
        help=f"Default: {FIXED_JUNIT_REL}",
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_TEST_RUN_MANIFEST_REL}",
    )
    p.add_argument("--skip-gate", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.pytest_log) if args.pytest_log else root / FIXED_LOG_REL
    junit_path = Path(args.junit_xml) if args.junit_xml else root / FIXED_JUNIT_REL
    man_out = (
        Path(args.manifest_out)
        if args.manifest_out
        else root / DEFAULT_TEST_RUN_MANIFEST_REL
    )
    if not log_path.is_absolute():
        log_path = root / log_path
    if not junit_path.is_absolute():
        junit_path = root / junit_path
    if not man_out.is_absolute():
        man_out = root / man_out

    log_path = log_path.resolve()
    junit_path = junit_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.parent.mkdir(parents=True, exist_ok=True)

    # Delete stale evidence so JUnit/log cannot be reused from a prior filtered run
    if log_path.is_file():
        log_path.unlink()
    if junit_path.is_file():
        junit_path.unlink()

    cmd = fixed_pytest_command(
        python_executable=sys.executable,
        junit_xml_absolute=junit_path,
    )

    pre_sha = current_dependency_tree_sha256(root)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    post_sha = current_dependency_tree_sha256(root)

    # Manifest records the exact subprocess argv that was executed
    manifest = build_test_run_manifest(
        root=root,
        test_command=cmd,
        pre_dependency_tree_sha256=pre_sha,
        post_dependency_tree_sha256=post_sha,
        exit_code=int(proc.returncode),
        pytest_log_path=log_path,
        junit_xml_path=junit_path,
        source_scope_version=SCOPE_VERSION_V2,
    )
    write_test_run_manifest(man_out, manifest)

    print(
        {
            "test_run_manifest": str(man_out),
            "exit_code": proc.returncode,
            "test_command": cmd,
            "pre_dependency_tree_sha256": pre_sha,
            "post_dependency_tree_sha256": post_sha,
            "passed_count": manifest.get("passed_count"),
            "failed_count": manifest.get("failed_count"),
        },
        flush=True,
    )

    if not args.skip_gate:
        gate = write_p0_evidence_binding_gate(
            root / DEFAULT_GATE_REL,
            root=root,
            test_run_manifest_path=man_out,
            status_path=root / DEFAULT_STATUS_REL,
        )
        print(
            {
                "gate": str(root / DEFAULT_GATE_REL),
                "ready_for_final_locked_rerun": gate.get("ready_for_final_locked_rerun"),
                "failure_reasons": gate.get("failure_reasons"),
            },
            flush=True,
        )

    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
