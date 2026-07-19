#!/usr/bin/env python3
"""Build a reference-only Train35 aggregate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _read(path: Path) -> dict:
    resolved = path if path.is_absolute() else ROOT / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-evidence-root", required=True)
    parser.add_argument("--primary32-evidence-root", required=True)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=ROOT
        / "conf/m1/cf1s_policies/cohorts/CF1S_DEVELOPMENT3_MANIFEST.json",
    )
    parser.add_argument(
        "--primary32-manifest",
        type=Path,
        default=ROOT
        / "conf/m1/cf1s_policies/cohorts/CF1S_PRIMARY32_MANIFEST.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_expansion import (
        build_train35_reference_report,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )

    development = _read(args.development_manifest)
    primary = _read(args.primary32_manifest)
    report = build_train35_reference_report(
        development3_evidence_root_sha256=args.development_evidence_root,
        primary32_evidence_root_sha256=args.primary32_evidence_root,
        development3_case_ids=development["ordered_case_ids"],
        primary32_case_ids=primary["ordered_case_ids"],
    )
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if _read(out) != report:
            raise SystemExit("existing Train35 report differs; refuse overwrite")
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(report) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
