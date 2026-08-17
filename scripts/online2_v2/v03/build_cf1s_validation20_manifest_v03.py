#!/usr/bin/env python3
"""Build CF1S_VALIDATION20_MANIFEST.json: Problem20 cases with real
input_artifact_sha256 bound (embedding sidecar hash) and provenance back to
the Problem20 manifest, for authorize_validation20()'s selection-blind chain.
selection_rule is trivial (all 20 Problem20 cases, fixed before any execution)
since Problem20 already locks exactly 20 cases."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
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

    problem20_path = ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_PROBLEM20_MANIFEST.json"
    problem20 = json.loads(problem20_path.read_text(encoding="utf-8"))
    problem20_sha = sha256_file(problem20_path)
    case_ids = list(problem20["ordered_case_ids"])
    if len(case_ids) != 20 or len(set(case_ids)) != 20:
        raise CoreContractError("Validation20 requires exactly 20 unique Problem20 cases")

    emb_dir = ROOT / "outputs/online2/v2_finetune_v02/event_embeddings"
    cases = []
    for row in problem20.get("cases") or []:
        case_id = str(row["case_id"])
        sidecar = emb_dir / f"{case_id}.parquet"
        if not sidecar.is_file():
            raise CoreContractError(f"case input sidecar missing: {sidecar}")
        cases.append(
            {
                "case_id": case_id,
                "source_label_row_identity_sha256": row.get(
                    "source_label_row_identity_sha256"
                ),
                "input_artifact_sha256": sha256_file(sidecar),
                "farm_id": row.get("farm_id"),
                "period_start": row.get("period_start"),
                "period_end": row.get("period_end"),
                "ground_truth_value_in_runtime_manifest": False,
            }
        )

    manifest = {
        "manifest_id": "CF1S_VALIDATION20_MANIFEST",
        "locked_cohort_case_count": 20,
        "ordered_case_ids": case_ids,
        "cases": cases,
        "ground_truth_value_in_runtime_manifest": False,
        "source_problem20_manifest_sha256": problem20_sha,
        "selection_rule": "ALL_PROBLEM20_CASES_LOCKED_BEFORE_EXECUTION",
    }

    out = ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_VALIDATION20_MANIFEST.json"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != manifest:
            raise CoreContractError(f"existing Validation20 manifest differs; refuse overwrite: {out}")
        print(json.dumps({"status": "EXISTING_VALID", "out": str(out)}, indent=2))
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(manifest) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "out": str(out),
                "case_count": len(case_ids),
                "source_problem20_manifest_sha256": problem20_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
