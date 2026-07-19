#!/usr/bin/env python3
"""Package a CF-1S official result directory with checksums and lock manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]

SNAPSHOT_FILES = (
    "conf/m1/cf1s_policies/CF1S_EDIT_POLICY_V1.yaml",
    "conf/m1/cf1s_policies/CF1S_SALIENCY_POLICY_V1.yaml",
    "conf/m1/cf1s_policies/CF1S_SEARCH_POLICY_V1.yaml",
    "conf/m1/cf1s_policies/CF1S_ACCEPTANCE_POLICY_V1.yaml",
    "conf/m1/cf1s_smoke.yaml",
    "src/online2/v2/finetune_v03/counterfactual/pipeline_cf1s.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/__init__.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/edit_policy.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/locus_universe.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/selectors.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/beam.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/interaction.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/fold_effect.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/status.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/identity.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/code_lock.py",
    "src/online2/v2/finetune_v03/counterfactual/cf1s/production_bridge.py",
    "src/online2/v2/finetune_v03/counterfactual/candidates/cf1s_composer.py",
    "src/online2/v2/finetune_v03/counterfactual/retokenization/multi_event_retokenizer.py",
    "src/online2/v2/finetune_v03/counterfactual/evaluation/cf1s_funnel_accounting.py",
    "scripts/online2_v2/v03/run_cf1s_v03.py",
    "scripts/online2_v2/v03/package_cf1s_v03.py",
    "tests/v2/counterfactual_cf1s/conftest.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_universe_selectors.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_interaction_fold_status.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_composer_retokenizer.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_pipeline_identity.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_budget_parent.py",
    "tests/v2/counterfactual_cf1s/test_cf1s_beam_budget.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory: Path) -> None:
    lines: List[str] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            lines.append(f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}")
    (directory / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--parent-cf0r-correction",
        type=Path,
        default=ROOT
        / "reports/cf0r_schema_recovery/CF0R_GOVERNANCE_CORRECTION_20260718T231902Z",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (
        ROOT / "reports/cf1s" / f"CF1S_MULTI_EVENT_SEQUENCE_EDIT_FEASIBILITY_{ts}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    shutil.copytree(args.source_run, out)

    snap = out / "code_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in SNAPSHOT_FILES:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    parent_sha = ""
    parent_manifest = args.parent_cf0r_correction / "CORRECTION_MANIFEST.json"
    if parent_manifest.exists():
        parent_sha = file_sha256(parent_manifest)

    package_manifest = {
        "package_role": "CF1S_MULTI_EVENT_SEQUENCE_EDIT_FEASIBILITY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_cf0r_correction_package": str(
            args.parent_cf0r_correction.relative_to(ROOT)
        )
        if args.parent_cf0r_correction.is_relative_to(ROOT)
        else str(args.parent_cf0r_correction),
        "parent_cf0r_correction_manifest_sha256": parent_sha,
        "action_authorization": False,
        "trajectory_authorization": False,
        "recommendation_authorization": False,
        "cf1b_authorization": False,
        "epsilon_change_allowed": False,
        "push_allowed": False,
        "snapshot_files": copied,
        "labels": [
            "MULTI_EVENT_MODEL_SENSITIVITY",
            "SEQUENCE_EDIT_FEASIBILITY",
            "NON_CAUSAL",
            "NOT_AN_ACTION",
            "NOT_RECOMMENDATION",
        ],
    }
    (out / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(package_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_checksums(out)
    archive = Path(str(out) + ".tar.gz")
    shutil.make_archive(str(out), "gztar", root_dir=out.parent, base_dir=out.name)
    (Path(str(archive) + ".sha256")).write_text(
        f"{file_sha256(archive)}  {archive.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "CF1S_PACKAGE_COMPLETE",
                "package_dir": str(out),
                "archive": str(archive),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
