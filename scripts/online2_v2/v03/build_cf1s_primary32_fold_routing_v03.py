#!/usr/bin/env python3
"""Build CF1S_PRIMARY32_FOLD_ROUTING_V1.json: 32 finetune-unseen,
selection-blind-to-Fold2 cases, same search/reevaluation fold structure as
Development3/Validation20 (search=[0,1], reevaluation=[2]), training_seen=false."""

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
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )

    primary32_path = ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_PRIMARY32_MANIFEST.json"
    primary32 = json.loads(primary32_path.read_text(encoding="utf-8"))
    case_ids = list(primary32["ordered_case_ids"])
    if len(case_ids) != 32:
        raise CoreContractError("Primary32 requires exactly 32 cases")

    routing = {
        "manifest_id": "CF1S_PRIMARY32_FOLD_ROUTING_V1",
        "cohort": "primary32",
        "scope_name": "SELECTION_BLIND_FOLD2_REEVALUATION",
        "default_search_checkpoint_ids": [0, 1],
        "default_selection_checkpoint_ids": [0, 1],
        "default_reevaluation_checkpoint_ids": [2],
        "cases": {
            case_id: {
                "training_seen": False,
                "is_training_holdout": True,
                "holdout_role": "SELECTION_BLIND_REEVALUATION",
                "holdout_blind_to_selection": True,
                "scientifically_independent_holdout": False,
                "search_checkpoint_ids": [0, 1],
                "selection_checkpoint_ids": [0, 1],
                "reevaluation_checkpoint_ids": [2],
                "forbidden_checkpoint_ids": [],
            }
            for case_id in case_ids
        },
    }

    out = ROOT / "conf/m1/cf1s_policies/cohorts/CF1S_PRIMARY32_FOLD_ROUTING_V1.json"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != routing:
            raise CoreContractError(f"existing fold routing differs; refuse overwrite: {out}")
        print(json.dumps({"status": "EXISTING_VALID", "out": str(out)}, indent=2))
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(routing) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out.parent)
    print(json.dumps({"status": "WRITTEN", "out": str(out), "case_count": len(case_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
