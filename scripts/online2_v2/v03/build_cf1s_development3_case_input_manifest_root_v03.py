#!/usr/bin/env python3
"""Build a cohort's case-input manifest root: a canonical hash over each
case's real raw-input artifact (embedding sidecar parquet), so the stable lock
binds to actual case-input bytes rather than trusting the case manifest alone.
Despite the filename (kept for D-LOCK code-tree continuity), this now supports
any cohort via --cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

COHORT_MANIFEST_CONFIG_KEYS = {
    "DEVELOPMENT3": "development_manifest_path",
    "VALIDATION20": "validation20_manifest_path",
    "PRIMARY32": "primary32_manifest_path",
}
COHORT_OUT_RELPATHS = {
    "DEVELOPMENT3": "outputs/cf1s_core/CF1S_DEVELOPMENT3_CASE_INPUT_MANIFEST_ROOT.json",
    "VALIDATION20": "outputs/cf1s_core/CF1S_VALIDATION20_CASE_INPUT_MANIFEST_ROOT.json",
    "PRIMARY32": "outputs/cf1s_core/CF1S_PRIMARY32_CASE_INPUT_MANIFEST_ROOT.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", choices=sorted(COHORT_MANIFEST_CONFIG_KEYS), default="DEVELOPMENT3"
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
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )

    import yaml

    cfg = yaml.safe_load((ROOT / "conf/m1/cf1s_core_smoke.yaml").read_text(encoding="utf-8"))
    man_path = ROOT / str(cfg[COHORT_MANIFEST_CONFIG_KEYS[args.cohort]])
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    case_ids = list(manifest["ordered_case_ids"])
    emb_dir = Path(cfg["embeddings_dir"])

    cases = []
    for case_id in sorted(case_ids):
        sidecar_path = emb_dir / f"{case_id}.parquet"
        if not sidecar_path.is_file():
            raise CoreContractError(f"case input sidecar missing: {sidecar_path}")
        cases.append(
            {
                "case_id": case_id,
                "input_artifact_relpath": str(sidecar_path.relative_to(ROOT)),
                "input_artifact_sha256": sha256_file(sidecar_path),
            }
        )

    root_manifest = {
        "artifact_kind": f"CF1S_{args.cohort}_CASE_INPUT_MANIFEST_ROOT_V1",
        "cohort": args.cohort,
        "case_count": len(cases),
        "cases": cases,
    }
    root_manifest["case_input_manifest_root_sha256"] = canonical_json_sha256(root_manifest)

    out = ROOT / COHORT_OUT_RELPATHS[args.cohort]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != root_manifest:
            raise CoreContractError(f"existing case-input manifest root differs; refuse overwrite: {out}")
        print(json.dumps({"status": "EXISTING_VALID", "out": str(out)}, indent=2))
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(root_manifest) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "out": str(out),
                "case_input_manifest_root_sha256": root_manifest[
                    "case_input_manifest_root_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
