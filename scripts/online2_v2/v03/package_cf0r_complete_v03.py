#!/usr/bin/env python3
"""Package official CF-0R completion artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


SNAPSHOT_FILES = [
    "scripts/online2_v2/v03/run_cf_funnel_audit_v03.py",
    "scripts/online2_v2/v03/merge_cf0r_composite_v03.py",
    "scripts/online2_v2/v03/run_cf0r_epsilon_sensitivity_v03.py",
    "conf/m1/cf_m2_prereq_smoke.yaml",
    "src/online2/v2/finetune_v03/counterfactual/pipeline_m2.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/funnel_accounting.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/mlm_preflight.py",
    "src/online2/v2/finetune_v03/counterfactual/candidates/path_a_schema_dispatch.py",
    "src/online2/v2/finetune_v03/counterfactual/candidates/path_a_generator.py",
    "src/transformer/transformer.py",
    "src/online2/v2/tokenizer.py",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--composite-dir",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--report-root",
        type=Path,
        default=ROOT / "reports/cf0r_schema_recovery",
    )
    ap.add_argument("--cf0r-commit", type=str, required=True)
    ap.add_argument(
        "--parent-archive-sha256",
        type=str,
        default="3134c5c5f5a801e58cc484c2fe948108ed15160b273cb52f1f14f4c2d4f72c34",
    )
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.report_root) / f"CF0R_COMPLETE_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    # Copy composite summary artifacts
    for name in (
        "funnel_audit_summary.json",
        "audit_manifest.json",
        "FUNNEL_AUDIT_REPORT.md",
        "epsilon_sensitivity_diagnostic.json",
    ):
        src = Path(args.composite_dir) / name
        if src.exists():
            shutil.copy2(src, out / name)
    per_src = Path(args.composite_dir) / "per_case"
    if per_src.exists():
        shutil.copytree(per_src, out / "per_case", dirs_exist_ok=True)

    # Code snapshot
    snap = out / "code_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    for rel in SNAPSHOT_FILES:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Tests log
    test_log = out / "test_log.txt"
    proc = subprocess.run(
        [
            str(Path.home() / "miniconda3/envs/life2vec/bin/python"),
            "-m",
            "pytest",
            "tests/v2/counterfactual_m1/test_cf0_funnel_accounting.py",
            "tests/v2/counterfactual_m1/test_cf0r_schema_dispatch.py",
            "tests/v2/counterfactual_m1/test_cf0r_decoder_warning.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    test_log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    summary = json.loads((out / "funnel_audit_summary.json").read_text(encoding="utf-8"))
    source_rev = {
        "parent_cf0_source_commit": summary.get("parent_cf0_source_commit"),
        "cf0r_source_commit": args.cf0r_commit,
        "push_allowed": False,
        "merged_result_type": "READ_ONLY_COMPOSITE_47_PLUS_8",
        "single_execution_lineage": False,
        "parent_archive_sha256": args.parent_archive_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completion_gate": summary.get("completion_gate"),
    }
    (out / "SOURCE_REVISION.json").write_text(
        json.dumps(source_rev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    readme = out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# CF-0R Schema Recovery Complete Package",
                "",
                f"- created_at: `{source_rev['created_at']}`",
                f"- cf0r_source_commit: `{args.cf0r_commit}`",
                f"- merged_result_type: `READ_ONLY_COMPOSITE_47_PLUS_8`",
                f"- completion: `{summary.get('completion_gate', {}).get('status')}`",
                f"- push_allowed: `false`",
                "",
                "## Guards",
                "",
                "1. Observational sensitivity candidates may be `model_valid_cf` but never `recommendation_valid_cf`.",
                "2. `NO_SCHEMA_SUPPORTED_CANDIDATE` is locus-level skip (not candidate conservation).",
                "3. Parent 47 and recovery 8 are different source commits (composite lineage).",
                "4. `current_operational_epsilon=0.01` is UNCHANGED_NOT_RETUNED; epsilon grid is diagnostic only.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # CHECKSUMS for all packaged files
    lines = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            rel = path.relative_to(out).as_posix()
            lines.append(f"{_sha256(path)}  {rel}")
    (out / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # tarball
    archive = Path(args.report_root) / f"{out.name}.tar.gz"
    subprocess.check_call(["tar", "-czf", str(archive), "-C", str(out.parent), out.name])
    archive_sha = _sha256(archive)
    (Path(args.report_root) / f"{out.name}.tar.gz.sha256").write_text(
        f"{archive_sha}  {archive.name}\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": "CF0R_PACKAGE_COMPLETE",
                "package_dir": str(out),
                "archive": str(archive),
                "archive_sha256": archive_sha,
                "test_exit_code": proc.returncode,
                "completion_gate": summary.get("completion_gate"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
