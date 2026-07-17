#!/usr/bin/env python3
"""Dedicated A2 functional decoder smoke — CF evaluation forbidden."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_new.vocabulary import RegistryVocabulary
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_index import (
    annotate_bundles_with_token_ids,
    wrap_executed_query_record,
    load_two_tier_bank,
    query_unique_bundles_from_bank,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.bank_integrity import (
    BANK_MODE_DEPLOYMENT,
    as_sequence_list,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_loader import (
    load_continuous_feature_names,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.constrained_mlm import score_bundles_joint
from src.online2.v2.finetune_v03.counterfactual.candidates.mlm_window_lift import (
    annotate_sentence_tokens,
    extract_value_bundle,
    lift_local_mask_to_window,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
    evaluate_a2_functional_smoke,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths
from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import _load_case_events
from src.online2.v2.finetune_v03.counterfactual.stage_a_loader import load_stage_a_module


def main() -> int:
    exit_code = 1
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
        ap.add_argument("--bank-dir", type=Path, default=None)
        args = ap.parse_args()
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        paths = M1Paths(Path(cfg["output_root"]))
        device = torch.device(str(cfg.get("device", "cpu")))
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        stage_a = load_stage_a_module()
        vocab = RegistryVocabulary(
            registry_path=str(cfg.get("vocabulary_path") or cfg["tokenizer_path"]),
            registry_version="v2",
        )
        encoder, _hp, abspos = stage_a.load_frozen_encoder(Path(cfg["stage_a_ckpt"]), vocab, device)
        if abspos is None:
            abspos = stage_a.load_abspos_reference(Path(cfg["abspos_reference_path"]))
        abspos_ts = __import__("pandas").Timestamp(abspos)

        case_id = str(cfg["case_id"])
        _sa, events, _farm, period_start = _load_case_events(cfg, case_id)
        continuous = load_continuous_feature_names(Path(cfg["feature_schema_path"]))
        mask_id = int(vocab.token2index["[MASK]"])
        unk = int(vocab.token2index.get("[UNK]", 0))

        # Pick first continuous MG with value tokens
        eligible = None
        tidx = None
        for i, ev in enumerate(events):
            ann = annotate_sentence_tokens(
                list(ev.sentence_tokens), vocab_token2index=vocab.token2index, unk_id=unk
            )
            seen = set()
            for g, f in zip(ann.group_ids, ann.features):
                if g in seen:
                    continue
                seen.add(g)
                if str(f).lower() not in continuous and str(f) not in continuous:
                    continue
                v_toks, v_ids, v_roles = extract_value_bundle(ann, measurement_group_id=g)
                if not v_toks:
                    continue
                eligible = {
                    "event_id": str(ev.event_id),
                    "measurement_group_id": str(g),
                    "feature": str(f),
                    "tokens": v_toks,
                    "roles": v_roles,
                    "token_ids": v_ids,
                    "ann": ann,
                    "timestamp": ev.timestamp,
                }
                tidx = i
                break
            if eligible:
                break

        result = {
            "run_type": "FUNCTIONAL_DECODER_SMOKE",
            "cf_evaluation_performed": False,
            "decoder_forward_call_count": 0,
            "unique_scored_bundle_count": 0,
            "eligible_mg_key": None,
            "functional_smoke_passed": False,
        }
        if not eligible or tidx is None:
            write_json(paths.artifacts / "mlm_functional_smoke_result.json", result)
            print(json.dumps(result, indent=2))
            return 1

        bank_dir = Path(args.bank_dir or paths.artifacts / "mlm_bundle_bank")
        manifest_path = bank_dir / "mlm_bundle_bank_manifest.json"
        if not manifest_path.exists():
            result.update(
                {
                    "functional_smoke_passed": False,
                    "reason": "missing_persisted_bank",
                }
            )
            write_json(paths.artifacts / "mlm_functional_smoke_result.json", result)
            print(json.dumps(result, indent=2))
            return 1

        observations, uniques, bank_manifest = load_two_tier_bank(bank_dir)
        bundles, executed = query_unique_bundles_from_bank(
            observations,
            uniques,
            feature=eligible["feature"],
            expected_roles=eligible["roles"],
            target_event_id=eligible["event_id"],
            target_timestamp=__import__("pandas").Timestamp(eligible["timestamp"]),
            case_id=case_id,
            mode=BANK_MODE_DEPLOYMENT,
        )
        bundles = annotate_bundles_with_token_ids(
            bundles, vocab_token2index=vocab.token2index, unk_id=unk
        )
        bank_query = wrap_executed_query_record(
            run_id="functional_smoke",
            bank_manifest=bank_manifest,
            executed_query_manifest=executed,
        )
        result["bank_query_record"] = bank_query
        result.update(bank_query)

        # Ensure at least original + one other if possible
        if not any(as_sequence_list(b.get("tokens")) == list(eligible["tokens"]) for b in bundles):
            bundles = list(bundles) + [
                {
                    "feature": eligible["feature"],
                    "tokens": list(eligible["tokens"]),
                    "token_ids": list(eligible["token_ids"]),
                    "roles": list(eligible["roles"]),
                    "is_original": True,
                }
            ]

        window = stage_a.construct_target_window(events, tidx, max_length=1024)
        x, pad, _L = stage_a.window_to_tensors(
            events,
            window,
            vocab=vocab,
            abspos_reference=abspos_ts,
            case_t0=period_start,
            max_length=1024,
        )
        ann = eligible["ann"]
        lift = lift_local_mask_to_window(
            local_token_ids=ann.token_ids,
            local_group_ids=ann.group_ids,
            local_roles=ann.roles,
            local_tokens=ann.tokens,
            target_group_id=eligible["measurement_group_id"],
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
        n_unique = len({tuple(c.tokens) for c in scored})
        result.update(
            {
                "decoder_forward_call_count": 1 if scored else 0,
                "unique_scored_bundle_count": n_unique,
                "eligible_mg_key": (
                    f"{case_id}|{eligible['event_id']}|"
                    f"{eligible['measurement_group_id']}|{eligible['feature']}"
                ),
                "n_bundles_input": len(bundles),
            }
        )
        # Set substantive pass flag before A2 eval (A2 requires the explicit flag).
        result["functional_smoke_passed"] = bool(
            int(result.get("decoder_forward_call_count") or 0) >= 1
            and int(result.get("unique_scored_bundle_count") or 0) >= 2
            and result.get("cf_evaluation_performed") is not True
            and str(result.get("run_type")) == "FUNCTIONAL_DECODER_SMOKE"
        )
        a2 = evaluate_a2_functional_smoke(result)
        result["a2_eval"] = a2
        result["functional_smoke_passed"] = bool(a2["a2_pass"])
        write_json(paths.artifacts / "mlm_functional_smoke_result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        exit_code = 0 if result["functional_smoke_passed"] else 1
        return exit_code
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
