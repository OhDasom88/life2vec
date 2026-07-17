#!/usr/bin/env python3
"""Build immutable 2-tier MLM bundle bank (observations + unique)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_new.vocabulary import RegistryVocabulary
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    observations_and_uniques_from_rows,
    write_two_tier_bank,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_loader import (
    iter_event_bundle_rows,
    load_continuous_feature_names,
    resolve_training_case_ids,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    BANK_MODE_RECONSTRUCTION_EVAL,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
    ap.add_argument(
        "--mode",
        default="index_neutral",
        help="Base bank is mode-neutral index; query mode lives on consumer artifacts",
    )
    ap.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap for debug only; 0 = full training + problem case",
    )
    args = ap.parse_args()

    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        paths = M1Paths(Path(cfg["output_root"]))
        vocab = RegistryVocabulary(
            registry_path=str(cfg.get("vocabulary_path") or cfg["tokenizer_path"]),
            registry_version="v2",
        )
        continuous = load_continuous_feature_names(Path(cfg["feature_schema_path"]))
        train_ids = list(resolve_training_case_ids(Path(cfg["labels_path"])))
        if int(args.max_cases) > 0:
            train_ids = train_ids[: int(args.max_cases)]
        problem_case = str(cfg.get("case_id") or "")
        case_ids = sorted(set(train_ids) | ({problem_case} if problem_case else set()))
        train_set = set(str(x) for x in train_ids)
        problem_set = {problem_case} if problem_case else set()
        overlap = sorted(train_set & problem_set)
        if overlap:
            write_json(
                paths.artifacts / "mlm_bundle_bank_meta.json",
                {
                    "ok": False,
                    "reason": "training_problem_case_overlap",
                    "training_problem_case_overlap_count": len(overlap),
                    "overlap_case_ids": overlap[:50],
                },
            )
            return 1
        allowed_cases = train_set | problem_set
        unk = int(vocab.token2index.get("[UNK]", 0))
        rows = list(
            iter_event_bundle_rows(
                Path(cfg["events_path"]),
                case_ids=case_ids,
                vocab_token2index=vocab.token2index,
                unk_id=unk,
                continuous_features=continuous,
                include_image=bool(cfg.get("include_image", False)),
            )
        )
        unknown_cases = []
        for row in rows:
            cid = str(row.get("case_id") or "")
            if cid not in allowed_cases:
                unknown_cases.append(cid)
                continue
            if cid in train_set:
                row["source_partition"] = "TRAINING"
            elif cid in problem_set:
                row["source_partition"] = "PROBLEM"
            else:
                unknown_cases.append(cid)
        if unknown_cases:
            write_json(
                paths.artifacts / "mlm_bundle_bank_meta.json",
                {
                    "ok": False,
                    "reason": "unknown_case_id_outside_train_problem",
                    "unknown_partition_row_count": len(unknown_cases),
                    "unknown_case_ids": sorted(set(unknown_cases))[:50],
                },
            )
            return 1
        try:
            observations, uniques = observations_and_uniques_from_rows(rows)
        except ValueError as e:
            write_json(
                paths.artifacts / "mlm_bundle_bank_meta.json",
                {"ok": False, "reason": "partition_assignment_failed", "error": str(e)},
            )
            return 1
        if not observations:
            write_json(
                paths.artifacts / "mlm_bundle_bank_meta.json",
                {"ok": False, "reason": "no_bank_rows"},
            )
            return 1

        bank_dir = paths.artifacts / "mlm_bundle_bank"
        manifest = write_two_tier_bank(
            bank_dir, observations=observations, uniques=uniques, mode=args.mode
        )
        manifest["training_problem_case_overlap_count"] = 0
        manifest["unknown_partition_row_count"] = 0
        manifest["source_partition_assignment_complete"] = True
        # rewrite manifest file with partition fields
        man_path = bank_dir / "mlm_bundle_bank_manifest.json"
        man_path.write_text(
            __import__("json").dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        out = {
            "ok": True,
            "n_training_cases": len(train_ids),
            "n_case_ids": len(case_ids),
            "problem_case_id": problem_case or None,
            "training_problem_case_overlap_count": 0,
            "unknown_partition_row_count": 0,
            "source_partition_assignment_complete": True,
            **manifest,
        }
        write_json(paths.artifacts / "mlm_bundle_bank_meta.json", out)
        write_json(paths.artifacts / "mlm_bundle_bank_index.json", out)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if manifest.get("a8_1_relational_pass") else 1
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
