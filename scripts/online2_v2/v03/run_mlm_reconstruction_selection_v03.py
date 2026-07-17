#!/usr/bin/env python3
"""Reconstruction selection only — freeze manifest then exit (A8-2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_new.vocabulary import RegistryVocabulary
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    annotate_bundles_with_token_ids,
    build_bank_query_set_manifest,
    load_two_tier_bank,
    query_unique_bundles_from_bank,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_RECONSTRUCTION_EVAL,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_loader import (
    load_continuous_feature_names,
    resolve_training_case_ids,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.constrained_mlm import (
    score_bundles_joint,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.mlm_window_lift import (
    annotate_sentence_tokens,
    extract_value_bundle,
    lift_local_mask_to_window,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    DEFAULT_EXPANSION_SCHEDULE,
    MIN_RECONSTRUCTION_CANDIDATES,
    build_selected_mg_flag_rows,
    build_selection_chain_document,
    frozen_final_manifest_sha256,
    mg_key,
    run_expansion_stages,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths
from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import _load_case_events
from src.online2.v2.finetune_v03.counterfactual.stage_a_loader import load_stage_a_module


def build_eligible_pool(cfg: Dict[str, Any], *, limit_cases: int = 0) -> List[Dict[str, Any]]:
    continuous = load_continuous_feature_names(Path(cfg["feature_schema_path"]))
    train_ids = list(resolve_training_case_ids(Path(cfg["labels_path"])))
    if int(limit_cases) > 0:
        train_ids = train_ids[: int(limit_cases)]
    pool: List[Dict[str, Any]] = []
    for cid in train_ids:
        try:
            _sa, events, _farm, _ps = _load_case_events(cfg, cid)
        except Exception:
            continue
        for ev in events:
            ann = annotate_sentence_tokens(list(ev.sentence_tokens))
            seen = set()
            for g, f in zip(ann.group_ids, ann.features):
                if g in seen:
                    continue
                seen.add(g)
                if str(f).lower() not in continuous and str(f) not in continuous:
                    continue
                v_toks, _, v_roles = extract_value_bundle(ann, measurement_group_id=g)
                if not v_toks:
                    continue
                pool.append(
                    {
                        "case_id": cid,
                        "event_id": str(ev.event_id),
                        "measurement_group_id": str(g),
                        "feature": str(f),
                        "original_tokens": list(v_toks),
                        "original_roles": list(v_roles),
                        "timestamp": pd.Timestamp(ev.timestamp),
                    }
                )
    return pool


def evaluate_recoverable(
    selected_rows: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    encoder,
    stage_a,
    vocab,
    abspos_ts,
    device: torch.device,
    observations,
    uniques,
) -> int:
    mask_id = int(vocab.token2index["[MASK]"])
    unk = int(vocab.token2index.get("[UNK]", 0))
    n_ok = 0
    for row in selected_rows:
        try:
            _sa, events, _farm, period_start = _load_case_events(cfg, row["case_id"])
        except Exception:
            continue
        eid_map = {str(ev.event_id): i for i, ev in enumerate(events)}
        if row["event_id"] not in eid_map:
            continue
        tidx = eid_map[row["event_id"]]
        ev = events[tidx]
        ann = annotate_sentence_tokens(
            list(ev.sentence_tokens),
            vocab_token2index=vocab.token2index,
            unk_id=unk,
        )
        bundles, _executed = query_unique_bundles_from_bank(
            observations,
            uniques,
            feature=row["feature"],
            expected_roles=row["original_roles"],
            target_event_id=row["event_id"],
            target_timestamp=row["timestamp"],
            case_id=row["case_id"],
            mode=BANK_MODE_RECONSTRUCTION_EVAL,
        )
        bundles = annotate_bundles_with_token_ids(
            bundles, vocab_token2index=vocab.token2index, unk_id=unk
        )
        if len(bundles) < MIN_RECONSTRUCTION_CANDIDATES:
            continue
        window = stage_a.construct_target_window(events, tidx, max_length=1024)
        x, pad, _L = stage_a.window_to_tensors(
            events,
            window,
            vocab=vocab,
            abspos_reference=abspos_ts,
            case_t0=period_start,
            max_length=1024,
        )
        try:
            lift = lift_local_mask_to_window(
                local_token_ids=ann.token_ids,
                local_group_ids=ann.group_ids,
                local_roles=ann.roles,
                local_tokens=ann.tokens,
                target_group_id=row["measurement_group_id"],
                mask_id=mask_id,
                window_input_ids_4ch=x,
                window_padding_mask=pad,
                target_token_start=int(window.target_token_start),
            )
            scored = score_bundles_joint(
                encoder,
                input_ids_4ch=lift.input_ids_4ch,
                padding_mask=lift.padding_mask.long(),
                mask_result=lift.mask_result,
                bundles=bundles,
                device=device,
            )
        except Exception:
            continue
        if not scored:
            continue
        # recoverable if original among scored
        orig = list(row["original_tokens"])
        if any(list(c.tokens) == orig for c in scored):
            n_ok += 1
    return n_ok


def main() -> int:
    exit_code = 1
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
        ap.add_argument(
            "--limit-cases",
            type=int,
            default=0,
            help="0 = full training pool; >0 debug-only case cap",
        )
        args = ap.parse_args()
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        out_root = Path(cfg["output_root"])
        paths = M1Paths(out_root)
        manifests = out_root / "manifests" / "mlm_selection"
        reports = out_root / "reports"
        manifests.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)

        device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        stage_a = load_stage_a_module()
        vocab = RegistryVocabulary(
            registry_path=str(cfg.get("vocabulary_path") or cfg["tokenizer_path"]),
            registry_version="v2",
        )
        encoder, _hp, abspos = stage_a.load_frozen_encoder(Path(cfg["stage_a_ckpt"]), vocab, device)
        if abspos is None:
            abspos = stage_a.load_abspos_reference(Path(cfg["abspos_reference_path"]))
        abspos_ts = pd.Timestamp(abspos)

        observations, uniques, bank_manifest = load_two_tier_bank(
            paths.artifacts / "mlm_bundle_bank"
        )

        pool = build_eligible_pool(cfg, limit_cases=int(args.limit_cases))
        mlm_cfg = cfg.get("mlm") or {}
        schedule = mlm_cfg.get("deterministic_expansion_schedule") or list(DEFAULT_EXPANSION_SCHEDULE)
        min_rec = int(mlm_cfg.get("min_recoverable_mg", 20))
        selection_seed = int(mlm_cfg.get("selection_seed", 20250717))

        def recoverable_fn(selected_rows):
            return evaluate_recoverable(
                selected_rows,
                cfg=cfg,
                encoder=encoder,
                stage_a=stage_a,
                vocab=vocab,
                abspos_ts=abspos_ts,
                device=device,
                observations=observations,
                uniques=uniques,
            )

        expansion = run_expansion_stages(
            pool,
            recoverable_fn=recoverable_fn,
            schedule=schedule,
            min_recoverable_mg=min_rec,
            out_dir=manifests,
            selection_seed=selection_seed,
        )

        # Freeze recoverable flags NOW (selection process) — metrics must not rewrite
        key_set = set(expansion["selected_mg_keys"])
        selected_rows = [r for r in pool if mg_key(r) in key_set]
        recoverable_keys: List[str] = []
        for row in selected_rows:
            n = evaluate_recoverable(
                [row],
                cfg=cfg,
                encoder=encoder,
                stage_a=stage_a,
                vocab=vocab,
                abspos_ts=abspos_ts,
                device=device,
                observations=observations,
                uniques=uniques,
            )
            if n >= 1:
                recoverable_keys.append(mg_key(row))
        flags = build_selected_mg_flag_rows(
            expansion["selected_mg_keys"], recoverable_keys=recoverable_keys
        )
        final_doc = dict(expansion["final"])
        final_doc.pop("selection_manifest_chain_hash", None)
        final_doc["selected_mg_flags"] = flags
        final_doc["frozen"] = True
        final_doc["frozen_by"] = "run_mlm_reconstruction_selection_v03"
        final_sha = frozen_final_manifest_sha256(final_doc)
        final_doc["selection_manifest_final_sha256"] = final_sha
        final_path = manifests / "selection_manifest_final.json"
        final_path.write_text(
            json.dumps(final_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        recoverable_count = sum(1 for f in flags if f.get("is_recoverable"))
        chain = build_selection_chain_document(
            stage_artifacts=expansion["stages"],
            selection_manifest_final_sha256=final_sha,
            final_stage=int(final_doc.get("final_stage") or 0),
            stopped_reason=str(final_doc.get("stopped_reason") or ""),
            recoverable_mg_count=recoverable_count,
        )
        write_json(manifests / "selection_manifest_chain.json", chain)
        write_json(reports / "selection_manifest_chain.json", chain)

        bank_query = {}
        if selected_rows:
            executed_entries = []
            for row in selected_rows:
                _b, executed = query_unique_bundles_from_bank(
                    observations,
                    uniques,
                    feature=row["feature"],
                    expected_roles=row["original_roles"],
                    target_event_id=row["event_id"],
                    target_timestamp=row["timestamp"],
                    case_id=row["case_id"],
                    mode=BANK_MODE_RECONSTRUCTION_EVAL,
                )
                executed_entries.append(
                    {"mg_key": mg_key(row), "executed_query_manifest": executed}
                )
            bank_query = build_bank_query_set_manifest(
                run_id="reconstruction_selection",
                bank_manifest=bank_manifest,
                mode=BANK_MODE_RECONSTRUCTION_EVAL,
                executed_entries=executed_entries,
            )
            bank_query["expected_mg_keys"] = [mg_key(r) for r in selected_rows]
            bank_query["query_set_manifest"] = {
                "run_id": bank_query["run_id"],
                "query_count": bank_query["query_count"],
                "query_records": bank_query["query_records"],
                "query_set_sha256": bank_query["query_set_sha256"],
            }
            write_json(reports / "bank_query_record_reconstruction_selection.json", bank_query)
            write_json(
                reports / "bank_query_set_manifest_reconstruction_selection.json",
                bank_query["query_set_manifest"],
            )

        out = {
            "ok": True,
            "n_selected": len(expansion["selected_mg_keys"]),
            "recoverable_mg_count": recoverable_count,
            "selection_manifest_final_sha256": final_sha,
            "selection_manifest_chain_hash": chain.get("selection_manifest_chain_hash"),
            "frozen": True,
            "bank_query_record": bank_query,
            **{k: bank_query[k] for k in bank_query if k != "query_manifest"},
        }
        write_json(reports / "reconstruction_selection_result.json", out)
        print(json.dumps(out, indent=2))
        exit_code = 0
        return exit_code
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
