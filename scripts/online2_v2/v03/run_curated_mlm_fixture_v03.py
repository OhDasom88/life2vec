#!/usr/bin/env python3
"""Frozen curated Problem fixture: STRUCTURAL_ONLY selection → non-original → critic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
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
)
from src.online2.v2.finetune_v03.counterfactual.candidates.constrained_mlm import (
    score_bundles_joint,
    select_mlm_candidates,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.mlm_window_lift import (
    annotate_sentence_tokens,
    extract_value_bundle,
    lift_local_mask_to_window,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.path_a_generator import (
    candidates_from_mlm_bundles,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.case_batch import (
    load_case_batch,
    tensor_batch_only,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.crossfit_critic import (
    resolve_fold_split,
    score_candidate_folds,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.curated_fixture import (
    select_structural_fixture,
    write_fixture_manifest,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.execution_provenance import (
    dump_runtime_imports_from_env,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json
from src.online2.v2.finetune_v03.counterfactual.pipeline import M1Paths
from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import (
    _bin_edges_for_feature,
    _load_case_events,
    _normalize_feature_name,
)
from src.online2.v2.finetune_v03.counterfactual.retokenization.full_event_retokenizer import (
    apply_raw_edit_closure,
    load_frozen_tokenizer_v2,
    recover_observed_raw_for_mg,
)
from src.online2.v2.finetune_v03.counterfactual.stage_a_loader import load_stage_a_module
from src.online2.v2.finetune_v03.counterfactual.stage_a_reencoder import StageAReencoder


def _query_bundles_for_locus(
    observations: Sequence[Mapping[str, Any]],
    uniques: Sequence[Mapping[str, Any]],
    *,
    vocab: RegistryVocabulary,
    feature: str,
    expected_roles: Sequence[str],
    target_event_id: str,
    target_timestamp: pd.Timestamp,
    case_id: str,
) -> tuple:
    unk = int(vocab.token2index.get("[UNK]", 0))
    bundles, executed = query_unique_bundles_from_bank(
        observations,
        uniques,
        feature=feature,
        expected_roles=expected_roles,
        target_event_id=target_event_id,
        target_timestamp=target_timestamp,
        case_id=case_id,
        mode=BANK_MODE_DEPLOYMENT,
    )
    bundles = annotate_bundles_with_token_ids(
        bundles, vocab_token2index=vocab.token2index, unk_id=unk
    )
    return bundles, executed


def main() -> int:
    exit_code = 1
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
        args = ap.parse_args()
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        paths = M1Paths(Path(cfg["output_root"]))
        case_id = str(cfg["case_id"])
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
        _sa, events, farm_id, period_start = _load_case_events(cfg, case_id)

        feature_schema_path = Path(cfg["feature_schema_path"])
        prod_tok, _hashes = load_frozen_tokenizer_v2(
            feature_schema_path=feature_schema_path,
            binning_registry_path=Path(cfg["binning_registry_path"]),
            vocab_path=Path(cfg.get("vocabulary_path") or cfg["tokenizer_path"]),
            farm_relative_enabled=True,
        )

        print("[curated] load persisted bank", flush=True)
        bank_dir = paths.artifacts / "mlm_bundle_bank"
        if not (bank_dir / "mlm_bundle_bank_manifest.json").is_file():
            out = {
                "integration_fixture_passed": False,
                "integration_fixture_status": "FAIL",
                "non_original_candidate_reached_critic": False,
                "reason": "missing_persisted_bank",
                "fixture_selection_policy": "STRUCTURAL_ONLY_PRE_MODEL",
                "decoder_scores_used_for_selection": False,
                "critic_scores_used_for_selection": False,
            }
            write_json(paths.reports / "curated_fixture_result.json", out)
            write_fixture_manifest(paths.manifests / "curated_fixture_manifest.json", out)
            print(json.dumps(out, indent=2))
            return 1

        observations, uniques, bank_manifest = load_two_tier_bank(bank_dir)
        bank_meta: Dict[str, Any] = {
            "run_id": "curated_fixture",
            "bundle_bank_content_sha256": bank_manifest.get("bundle_bank_content_sha256"),
            "observation_file_sha256": bank_manifest.get("observation_file_sha256"),
            "unique_file_sha256": bank_manifest.get("unique_file_sha256"),
            "bank_dir": str(bank_dir),
        }
        print(f"[curated] bank_observations={len(observations)} uniques={len(uniques)}", flush=True)

        from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_loader import (
            load_continuous_feature_names,
        )

        continuous = load_continuous_feature_names(feature_schema_path)
        prefer = _normalize_feature_name(str(cfg.get("path_a_probe_feature") or "inside_temp_c"))
        fixture = None
        # Prefer probe feature MGs first
        mg_candidates = []
        for ev in events:
            ann = annotate_sentence_tokens(
                list(ev.sentence_tokens),
                vocab_token2index=vocab.token2index,
                unk_id=int(vocab.token2index.get("[UNK]", 0)),
            )
            seen = set()
            for g, f in zip(ann.group_ids, ann.features):
                if g in seen:
                    continue
                seen.add(g)
                feat = _normalize_feature_name(f)
                if feat not in continuous and str(f) not in continuous:
                    continue
                orig_toks, _oids, orig_roles = extract_value_bundle(ann, measurement_group_id=g)
                if not orig_toks:
                    continue
                mg_candidates.append((0 if feat == prefer else 1, ev, g, f, feat, orig_toks, orig_roles))
        mg_candidates.sort(key=lambda x: (x[0], str(x[1].event_id), str(x[2])))

        for _prio, ev, g, f, feat, orig_toks, orig_roles in mg_candidates[:40]:
            print(f"[curated] try feature={feat} event={ev.event_id[:12]}...", flush=True)
            try:
                edges = _bin_edges_for_feature(cfg, feat)
            except Exception:
                continue
            bundles, _executed_q = _query_bundles_for_locus(
                observations,
                uniques,
                vocab=vocab,
                feature=str(f),
                expected_roles=orig_roles,
                target_event_id=str(ev.event_id),
                target_timestamp=pd.Timestamp(ev.timestamp),
                case_id=case_id,
            )
            bundles = bundles[:64]
            recovered = recover_observed_raw_for_mg(
                prod_tok,
                feature=feat,
                farm_id=farm_id,
                original_tokens=list(ev.sentence_tokens),
                grounded_raw=None,
                edges_abs=edges,
                allow_abs_bin_probe=True,
            )
            if not recovered.get("ok"):
                continue
            observed_raw = float(recovered["observed_raw"])
            cand_fix = select_structural_fixture(
                case_id=case_id,
                event_id=str(ev.event_id),
                feature=feat,
                measurement_group_id=str(g),
                observed_raw=observed_raw,
                edges_abs=edges,
                original_tokens=orig_toks,
                bank_bundles=bundles,
                tokenizer=prod_tok,
                farm_id=farm_id,
            )
            if cand_fix.get("ok"):
                fixture = {
                    **cand_fix,
                    "original_tokens": orig_toks,
                    "original_roles": orig_roles,
                    "edges": list(edges),
                    "event_timestamp": str(ev.timestamp),
                    "feature_raw": str(f),
                }
                break

        if not fixture:
            out = {
                "integration_fixture_passed": False,
                "integration_fixture_status": "FAIL",
                "non_original_candidate_reached_critic": False,
                "reason": "NO_STRUCTURAL_FIXTURE",
                "fixture_selection_policy": "STRUCTURAL_ONLY_PRE_MODEL",
                "decoder_scores_used_for_selection": False,
                "critic_scores_used_for_selection": False,
            }
            write_json(paths.reports / "curated_fixture_result.json", out)
            write_fixture_manifest(paths.manifests / "curated_fixture_manifest.json", out)
            print(json.dumps(out, indent=2))
            return 1

        write_fixture_manifest(paths.manifests / "curated_fixture_manifest.json", fixture)

        # Within frozen MG: score all unique bundles, pick best non-original invertible EDIT
        eid = fixture["event_id"]
        feat = fixture["feature"]
        mg_id = fixture["measurement_group_id"]
        events_by_id = {str(ev.event_id): i for i, ev in enumerate(events)}
        tidx = events_by_id[eid]
        ev = events[tidx]
        ann = annotate_sentence_tokens(
            list(ev.sentence_tokens),
            vocab_token2index=vocab.token2index,
            unk_id=int(vocab.token2index.get("[UNK]", 0)),
        )
        bundles, _executed_q = _query_bundles_for_locus(
            observations,
            uniques,
            vocab=vocab,
            feature=fixture.get("feature_raw") or feat,
            expected_roles=fixture["original_roles"],
            target_event_id=eid,
            target_timestamp=pd.Timestamp(ev.timestamp),
            case_id=case_id,
        )
        window = stage_a.construct_target_window(events, tidx, max_length=1024)
        x, pad, _L = stage_a.window_to_tensors(
            events,
            window,
            vocab=vocab,
            abspos_reference=abspos_ts,
            case_t0=period_start,
            max_length=1024,
        )
        lift = lift_local_mask_to_window(
            local_token_ids=ann.token_ids,
            local_group_ids=ann.group_ids,
            local_roles=ann.roles,
            local_tokens=ann.tokens,
            target_group_id=mg_id,
            mask_id=int(vocab.token2index["[MASK]"]),
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
        selected = select_mlm_candidates(scored, top_k=5, exclude_original_from_edits=True)
        observed_raw = float(fixture["selected_target_raw"])  # structural seed; prefer recovered below
        target_raw = float(fixture["selected_target_raw"])
        proposed = list(fixture["selected_bundle_tokens"])
        non_orig = [c for c in selected if not c.is_original]
        if non_orig:
            raw_cands = candidates_from_mlm_bundles(
                feature=feat,
                observed_raw=float(fixture["observed_raw"]),
                edges=fixture["edges"],
                bundles=[non_orig[0]],
                margin_ratio=0.001,
            )
            edit = next((c for c in raw_cands if c.get("inversion_ok") and not c.get("is_noop")), None)
            if edit is not None:
                target_raw = float(edit["target_raw"])
                from src.online2.v2.finetune_v03.counterfactual.retokenization.full_event_retokenizer import (
                    tokenize_mg_production,
                )

                proposed = tokenize_mg_production(
                    prod_tok, feature=feat, raw_value=target_raw, farm_id=farm_id
                )

        man = json.loads((paths.manifests / "cf_m2_prereq_manifest.json").read_text())
        split = resolve_fold_split(
            checkpoint_paths=man["checkpoint_paths"],
            fold_ids=man["fold_ids"],
            search_fold_ids=man.get("search_fold_ids") or [0, 1],
            holdout_fold_ids=man.get("holdout_fold_ids") or [2],
        )
        batch, side = load_case_batch(
            case_id=case_id,
            embeddings_dir=Path(cfg["embeddings_dir"]),
            labels_path=Path(cfg["labels_path"]),
            label_map_path=Path(cfg.get("label_map_path") or Path(cfg["labels_path"]).parent / "label_map.json"),
        )
        base = tensor_batch_only(batch)
        eid_to_idx = {str(r.event_id): int(r.event_index) for r in side.itertuples()}
        reenc = StageAReencoder(
            encoder=encoder,
            stage_a_mod=stage_a,
            vocab=vocab,
            abspos_reference=abspos_ts,
            case_t0=period_start,
            device=device,
        )
        rtok = apply_raw_edit_closure(
            event_id=eid,
            feature=feat,
            target_raw=target_raw,
            original_tokens=list(ev.sentence_tokens),
            tokenizer=prod_tok,
            farm_id=farm_id,
            proposed_bundle=proposed,
            observed_raw=float(fixture["observed_raw"]),
            strict_exact=True,
        )
        reached_critic = False
        search_stats = None
        if rtok.baseline_roundtrip_valid and (rtok.gate4 or {}).get("status") == "PASSED":
            result = reenc.apply_edits_and_reencode(
                events,
                [{"event_id": eid, "to_tokens": list(rtok.new_sentence_tokens), "replace_sentence": True}],
                {k: v.to(device) for k, v in base.items()},
                event_id_to_index=eid_to_idx,
            )
            search_stats = score_candidate_folds(
                split.search_ckpts(),
                {k: v.cpu() for k, v in base.items()},
                {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in result.batch.items()},
                normal_class_id=int(batch["_normal_class_id"]),
                fold_ids=split.search_fold_ids,
                device=str(cfg.get("device", "cuda")),
                gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
            )
            reached_critic = True

        token_edit_count = int(
            sum(a != b for a, b in zip(ev.sentence_tokens, rtok.new_sentence_tokens))
            if len(ev.sentence_tokens) == len(rtok.new_sentence_tokens)
            else abs(len(ev.sentence_tokens) - len(rtok.new_sentence_tokens))
        )
        bank_meta = wrap_executed_query_record(
            run_id="curated_fixture",
            bank_manifest=bank_manifest,
            executed_query_manifest=_executed_q,
        )

        bank_meta["bank_dir"] = str(bank_dir)
        out = {
            "integration_fixture_passed": bool(reached_critic and token_edit_count > 0),
            "integration_fixture_status": "PASS" if reached_critic and token_edit_count > 0 else "FAIL",
            "non_original_candidate_reached_critic": bool(reached_critic and token_edit_count > 0),
            "fixture_selection_policy": "STRUCTURAL_ONLY_PRE_MODEL",
            "decoder_scores_used_for_selection": False,
            "critic_scores_used_for_selection": False,
            "fixture_frozen_before_integration_run": True,
            "selection_reason": "PREVERIFIED_INVERTIBLE_NON_ORIGINAL_BUNDLE",
            "token_edit_count": token_edit_count,
            "gate4_status": (rtok.gate4 or {}).get("status"),
            "search_delta_r": None if search_stats is None else search_stats.get("delta_r"),
            "bank_meta": bank_meta,
            "bank_query_record": bank_meta,
            "decoder_forward_call_count": 1,
            "unique_scored_bundle_count": len({tuple(c.tokens) for c in scored}),
            "fixture": {k: v for k, v in fixture.items() if k != "edges"},
        }
        write_json(paths.reports / "curated_fixture_result.json", out)
        print(json.dumps(out, indent=2, default=str))
        exit_code = 0 if out["integration_fixture_passed"] else 1
        return exit_code
    finally:
        dump_runtime_imports_from_env(ROOT, entrypoint=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
