"""Path A constrained-MLM orchestration: mask → lift → score → invert."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

from ..evaluation.bank_integrity import BANK_MODE_DEPLOYMENT, as_sequence_list
from .bundle_bank_index import (
    annotate_bundles_with_token_ids,
    build_bank_query_record,
    load_two_tier_bank,
    query_unique_bundles_from_bank,
    wrap_executed_query_record,
)
from .bundle_bank_loader import (
    load_continuous_feature_names,
)
from .constrained_mlm import score_bundles_joint, select_mlm_candidates
from .mlm_window_lift import annotate_sentence_tokens, extract_value_bundle, lift_local_mask_to_window
from .path_a_generator import candidates_from_mlm_bundles


def is_continuous_feature(feature: str, continuous: set) -> bool:
    f = str(feature)
    return f in continuous or f.lower() in continuous


def _default_bank_dir(cfg: Mapping[str, Any]) -> Path:
    out = Path(cfg["output_root"])
    return out / "artifacts" / "mlm_bundle_bank"


def run_constrained_mlm_for_locus(
    *,
    cfg: Mapping[str, Any],
    encoder,
    stage_a_mod,
    vocab,
    abspos_reference: pd.Timestamp,
    case_t0: pd.Timestamp,
    events: list,
    target_event_idx: int,
    event_id: str,
    feature: str,
    measurement_group_id: str,
    observed_raw: float,
    edges_abs: Sequence[float],
    case_id: str,
    device: torch.device,
    bank_rows_cache: Optional[List[Dict[str, Any]]] = None,
    bank_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute constrained MLM candidate generation for one locus.

    Returns dict with candidates, funnel stats, and decoder call evidence.
    Persisted two-tier bank is load-only (no corpus rebuild).
    """
    mlm_cfg = dict(cfg.get("mlm") or {})
    if not mlm_cfg.get("enabled", True):
        return {
            "mlm_executed": False,
            "reason": "mlm_disabled",
            "candidates": [],
            "decoder_forward_call_count": 0,
            "unique_scored_bundle_count": 0,
        }

    continuous = load_continuous_feature_names(Path(cfg["feature_schema_path"]))
    if mlm_cfg.get("continuous_features_only", True) and not is_continuous_feature(
        feature, continuous
    ):
        return {
            "mlm_executed": True,
            "reason": "non_continuous_feature",
            "candidates": [],
            "decoder_forward_call_count": 0,
            "unique_scored_bundle_count": 0,
            "n_eligible_loci": 0,
        }

    mask_id = int(vocab.token2index["[MASK]"])
    unk_id = int(vocab.token2index.get("[UNK]", 0))
    ev = events[target_event_idx]
    ann = annotate_sentence_tokens(
        list(ev.sentence_tokens),
        vocab_token2index=vocab.token2index,
        unk_id=unk_id,
    )
    # Prefer provided mg id; else derive from feature
    mg_id = str(measurement_group_id or f"mg:{feature}")
    if mg_id not in set(ann.group_ids):
        # try normalized feature match
        for g, f in zip(ann.group_ids, ann.features):
            if str(f).lower() == str(feature).lower():
                mg_id = str(g)
                break

    orig_toks, orig_ids, orig_roles = extract_value_bundle(ann, measurement_group_id=mg_id)
    if not orig_toks:
        return {
            "mlm_executed": True,
            "reason": "no_value_tokens_in_mg",
            "candidates": [],
            "decoder_forward_call_count": 0,
            "unique_scored_bundle_count": 0,
            "n_eligible_loci": 0,
            "measurement_group_id": mg_id,
        }

    # Bundle bank — persisted load-only
    bdir = Path(bank_dir) if bank_dir is not None else _default_bank_dir(cfg)
    try:
        observations, uniques, bank_manifest = load_two_tier_bank(bdir)
    except FileNotFoundError as e:
        return {
            "mlm_executed": True,
            "reason": "missing_persisted_bank",
            "error": str(e),
            "candidates": [],
            "decoder_forward_call_count": 0,
            "unique_scored_bundle_count": 0,
            "n_eligible_loci": 0,
            "measurement_group_id": mg_id,
        }

    target_ts = pd.Timestamp(ev.timestamp)
    bank_mode = BANK_MODE_DEPLOYMENT
    bundles, executed = query_unique_bundles_from_bank(
        observations,
        uniques,
        feature=feature,
        expected_roles=orig_roles,
        target_event_id=event_id,
        target_timestamp=target_ts,
        case_id=case_id,
        mode=bank_mode,
    )
    bundles = annotate_bundles_with_token_ids(
        bundles, vocab_token2index=vocab.token2index, unk_id=unk_id
    )
    bank_meta = wrap_executed_query_record(
        run_id="natural_path_a",
        bank_manifest=bank_manifest,
        executed_query_manifest=executed,
    )
    bank_meta["bank_dir"] = str(bdir)
    bank_meta["legacy_bank_rows_cache_ignored"] = bank_rows_cache is not None
    # Always include observed original for NO_OP reconstruction ranking
    if not any(b.get("is_original") for b in bundles):
        bundles.append(
            {
                "feature": feature,
                "tokens": list(orig_toks),
                "token_ids": list(orig_ids),
                "roles": list(orig_roles),
                "signature": tuple(orig_roles),
                "event_id": event_id,
                "is_original": True,
            }
        )
    else:
        for b in bundles:
            b["is_original"] = as_sequence_list(b.get("tokens")) == list(orig_toks)

    window = stage_a_mod.construct_target_window(events, target_event_idx, max_length=1024)
    x, pad_mask, _L = stage_a_mod.window_to_tensors(
        events,
        window,
        vocab=vocab,
        abspos_reference=abspos_reference,
        case_t0=case_t0,
        max_length=1024,
    )
    lift = lift_local_mask_to_window(
        local_token_ids=ann.token_ids,
        local_group_ids=ann.group_ids,
        local_roles=ann.roles,
        local_tokens=ann.tokens,
        target_group_id=mg_id,
        mask_id=mask_id,
        window_input_ids_4ch=x,
        window_padding_mask=pad_mask,
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
    top_k = int(mlm_cfg.get("bundles_per_locus", 3))
    selected = select_mlm_candidates(
        scored,
        top_k=top_k,
        exclude_original_from_edits=bool(mlm_cfg.get("exclude_original_from_edits", True)),
    )
    margin = float(mlm_cfg.get("raw_interval_interior_margin_ratio", 0.001))
    raw_cands = candidates_from_mlm_bundles(
        feature=feature,
        observed_raw=observed_raw,
        edges=edges_abs,
        bundles=selected,
        margin_ratio=margin,
    )

    inversion_failures = sum(1 for c in raw_cands if c.get("rejected"))
    valid_edits = [
        c
        for c in raw_cands
        if c.get("source") == "constrained_mlm"
        and c.get("inversion_ok")
        and not c.get("is_noop")
        and c.get("target_raw") is not None
        and abs(float(c["target_raw"]) - float(observed_raw)) > 1e-12
    ]
    # annotate ranks / scores
    for i, c in enumerate(raw_cands):
        c["bundle_rank"] = i
        c["target_event_excluded"] = True
        c["mlm_bundle_score"] = c.get("mlm_score")
        c["measurement_group_id"] = mg_id

    from ..evaluation.funnel_accounting import (
        SELECTION_PRUNED,
        SELECTION_TOPK,
        TERMINAL_FAILED,
        empty_funnel_counts,
    )

    topk_records = []
    for i, c in enumerate(raw_cands):
        cand_id = f"mlm::{mg_id}::{i}"
        if c.get("rejected") or not c.get("inversion_ok"):
            reason = str(c.get("inversion_reason_code") or "NON_INVERTIBLE_BUNDLE")
            code_map = {
                "role_signature_mismatch": "ROLE_SIGNATURE_MISMATCH",
                "ROLE_SIGNATURE_MISMATCH": "ROLE_SIGNATURE_MISMATCH",
                "empty_interval": "EMPTY_INTERVAL_INTERSECTION",
                "EMPTY_INTERVAL_INTERSECTION": "EMPTY_INTERVAL_INTERSECTION",
                "collapsed_noop": "TARGET_RAW_COLLAPSED_TO_NOOP",
                "TARGET_RAW_COLLAPSED_TO_NOOP": "TARGET_RAW_COLLAPSED_TO_NOOP",
            }
            term_reason = code_map.get(reason, "NON_INVERTIBLE_BUNDLE")
            topk_records.append(
                {
                    "candidate_id": cand_id,
                    "selection_status": SELECTION_TOPK,
                    "terminal_status": TERMINAL_FAILED,
                    "terminal_failure_reason": term_reason,
                    "last_completed_stage": "RAW_INVERSION",
                    "bundle_tokens": as_sequence_list(c.get("proposed_token_bundle"))
                    or as_sequence_list(c.get("tokens")),
                }
            )
        elif c.get("is_noop") or (
            c.get("target_raw") is not None
            and abs(float(c["target_raw"]) - float(observed_raw)) <= 1e-12
        ):
            topk_records.append(
                {
                    "candidate_id": cand_id,
                    "selection_status": SELECTION_TOPK,
                    "terminal_status": TERMINAL_FAILED,
                    "terminal_failure_reason": "TARGET_RAW_COLLAPSED_TO_NOOP",
                    "last_completed_stage": "RAW_INVERSION",
                    "bundle_tokens": as_sequence_list(c.get("proposed_token_bundle"))
                    or as_sequence_list(c.get("tokens")),
                }
            )
        else:
            topk_records.append(
                {
                    "candidate_id": cand_id,
                    "selection_status": SELECTION_TOPK,
                    "terminal_status": "PENDING",
                    "terminal_failure_reason": None,
                    "last_completed_stage": "RAW_INVERSION",
                    "pending_downstream": True,
                    "target_raw": c.get("target_raw"),
                    "bundle_tokens": as_sequence_list(c.get("proposed_token_bundle"))
                    or as_sequence_list(c.get("tokens")),
                }
            )

    n_scored_unique = len({tuple(c.tokens) for c in scored})
    n_not_selected = max(0, n_scored_unique - len(selected))
    funnel = empty_funnel_counts()
    funnel["n_bank_unique"] = len(bundles)
    funnel["n_scored_unique"] = n_scored_unique
    funnel["n_topk"] = len(topk_records)
    funnel["n_not_selected_topk"] = n_not_selected
    funnel["n_inversion_terminal_fail"] = sum(
        1
        for r in topk_records
        if r.get("terminal_status") == TERMINAL_FAILED
        and r.get("terminal_failure_reason")
        in {
            "ROLE_SIGNATURE_MISMATCH",
            "NON_INVERTIBLE_BUNDLE",
            "EMPTY_INTERVAL_INTERSECTION",
            "TARGET_RAW_COLLAPSED_TO_NOOP",
        }
    )
    funnel["n_inversion_ok"] = funnel["n_topk"] - funnel["n_inversion_terminal_fail"]
    funnel["pending_downstream_stages"] = any(r.get("pending_downstream") for r in topk_records)
    funnel["topk_records"] = topk_records
    funnel["pruned_count"] = n_not_selected
    funnel["selection_status_note"] = SELECTION_PRUNED

    bank_meta = {
        **bank_meta,
        "n_unique_fingerprints": len(bundles),
    }

    return {
        "mlm_executed": True,
        "reason": "ok",
        "candidates": raw_cands,
        "valid_edit_candidates": valid_edits,
        "scored_bundles": [
            {
                "tokens": c.tokens,
                "token_ids": c.token_ids,
                "score": c.score,
                "is_original": c.is_original,
                "source": c.source,
            }
            for c in scored
        ],
        "bank_meta": bank_meta,
        "lift_meta": {
            "masked_window_positions": lift.window_masked_indices,
            "modified_channels": lift.modified_channels,
            "unchanged_auxiliary_channels": lift.unchanged_auxiliary_channels,
            "unchanged_non_target_positions": lift.unchanged_non_target_positions,
            "unchanged_attention_mask": lift.unchanged_attention_mask,
        },
        "decoder_forward_call_count": 1 if scored else 0,
        "unique_scored_bundle_count": n_scored_unique,
        "n_eligible_loci": 1,
        "inversion_failures": inversion_failures,
        "n_valid_mlm_edits": len(valid_edits),
        "measurement_group_id": mg_id,
        "original_tokens": orig_toks,
        "original_roles": orig_roles,
        "funnel": funnel,
        "topk_records": topk_records,
    }
