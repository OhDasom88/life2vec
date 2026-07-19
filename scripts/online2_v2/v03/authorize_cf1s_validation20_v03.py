#!/usr/bin/env python3
"""Authorize Validation20 from an all-PASS Development3 completion."""

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
    parser.add_argument("--development-final-verdict", type=Path, required=True)
    parser.add_argument("--development-completion", type=Path, required=True)
    parser.add_argument("--validation20-manifest", type=Path, required=True)
    parser.add_argument(
        "--problem20-manifest",
        type=Path,
        default=ROOT
        / "conf/m1/cf1s_policies/cohorts/CF1S_PROBLEM20_MANIFEST.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/cf1s_core/CF1S_VALIDATION20_EXECUTION_AUTHORIZATION.json",
    )
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        sha256_file,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_expansion import (
        authorize_validation20,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )

    problem_path = (
        args.problem20_manifest
        if args.problem20_manifest.is_absolute()
        else ROOT / args.problem20_manifest
    )
    authorization = authorize_validation20(
        development_final_verdict=_read(args.development_final_verdict),
        development_completion=_read(args.development_completion),
        validation_manifest=_read(args.validation20_manifest),
        problem20_manifest_sha256=sha256_file(problem_path),
    )
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = _read(out)
        if existing != authorization:
            raise SystemExit("existing Validation20 authorization differs; refuse overwrite")
        print(json.dumps({"status": "EXISTING_VALID", "out": str(out)}, indent=2))
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(authorization) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(json.dumps({"status": "AUTHORIZED", "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
