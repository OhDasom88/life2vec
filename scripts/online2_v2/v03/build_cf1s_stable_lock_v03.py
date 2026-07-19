#!/usr/bin/env python3
"""Build a full non-null CF1S stable-lock manifest from declared source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-sources",
        type=Path,
        required=True,
        help="JSON object mapping every non-code/test/policy SHA field to a source file.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_authorization import (
        repository_relative,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        CoreContractError,
        sha256_file,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_locks import (
        REQUIRED_STABLE_LOCK_SHA_FIELDS,
        compute_stable_lock_manifest,
        validate_stable_lock_manifest,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )

    source_path = (
        args.artifact_sources
        if args.artifact_sources.is_absolute()
        else ROOT / args.artifact_sources
    )
    declared = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(declared, dict):
        raise CoreContractError("artifact-sources must be a JSON object")
    required = set(REQUIRED_STABLE_LOCK_SHA_FIELDS) - {
        "code_tree_sha256",
        "test_tree_sha256",
        "policy_tree_sha256",
    }
    if set(declared) != required:
        missing = sorted(required - set(declared))
        extra = sorted(set(declared) - required)
        raise CoreContractError(
            f"artifact source keys mismatch; missing={missing} extra={extra}"
        )
    resolved = {}
    normalized_sources = {}
    for field, raw in declared.items():
        path = Path(str(raw))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise CoreContractError(f"artifact source missing for {field}: {path}")
        resolved[field] = sha256_file(path)
        try:
            normalized_sources[field] = repository_relative(path, ROOT)
        except ValueError:
            normalized_sources[field] = str(path.resolve())

    manifest = compute_stable_lock_manifest(
        root=ROOT,
        qualification_test_log_sha=resolved["qualification_test_log_sha256"],
        qualification_preflight_sha=resolved["qualification_preflight_sha256"],
        qualification_test_node_id_manifest_sha=resolved[
            "qualification_node_id_manifest_sha256"
        ],
        artifact_sha256=resolved,
        artifact_sources=normalized_sources,
        runtime_versions={},
        risk_head_contract={},
    )
    validate_stable_lock_manifest(manifest)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != manifest:
            raise CoreContractError("existing stable lock differs; refuse overwrite")
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(manifest) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(json.dumps({"status": "WRITTEN", "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
