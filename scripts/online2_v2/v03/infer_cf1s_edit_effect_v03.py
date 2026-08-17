#!/usr/bin/env python3
"""Lightweight CF1S inference: apply a proposed multi-event raw-value edit to
a real sequence and report the model's predicted risk/logit delta.

This is NOT part of the governance/verification pipeline (no hash-chain trace,
no repeated noise measurement, no candidate search/selection, no signing or
promotion). It reuses the same validated grounding -> retokenization ->
cold-rebuild -> forward machinery as the governance path so the transformation
itself is identical to what was scientifically verified across Development3/
Validation20/Primary32 — it just skips the audit ceremony for a single,
already-specified edit, so it runs in seconds instead of minutes.

Edits file format (JSON list):
  [{"event_id": "...", "feature_id": "substrate_temp_c", "target_raw": 18.5}, ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _abnormal_logit(row: Mapping[str, Any]) -> float:
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        CoreContractError,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_scientific import (
        check_risk_logit_contract,
    )

    logits = list(row.get("logit") or [])
    if len(logits) != 1:
        raise CoreContractError("abnormal binary logit must contain exactly one value")
    value = float(logits[0])
    check_risk_logit_contract(float(row["risk"]), value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--edits",
        type=Path,
        required=True,
        help="JSON file: list of {event_id, feature_id, target_raw}",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "conf/m1/cf1s_core_smoke.yaml"
    )
    parser.add_argument(
        "--fold-ids",
        type=str,
        default=None,
        help="Comma-separated checkpoint fold indices to score (default: all discovered)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_sha256,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
        CoreContractError,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_execution import (
        production_apply_edits_and_score,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_production import (
        make_production_fold_forward_fn,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
        ValidatedRawAtomicEdit,
        assert_change_set_closed,
        build_allowed_change_set,
        build_identity_transaction,
        build_validated_raw_transaction,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime import (
        apply_exact_byte_deterministic_runtime,
        build_stage_a_reencoder,
        discover_critic_checkpoints,
        load_case_events,
        parse_case_id,
    )
    from src.online2.v2.finetune_v03.counterfactual.evaluation.case_batch import (
        load_embedding_sidecar,
    )
    from src.online2.v2.finetune_v03.counterfactual.grounding.locus_raw import (
        ground_locus_raw,
        is_exact,
    )
    from src.online2.v2.finetune_v03.counterfactual.retokenization.full_event_retokenizer import (
        apply_raw_edit_closure,
        load_frozen_tokenizer_v2,
    )

    import pandas as pd

    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    case_id = args.case_id

    requested_edits: List[Dict[str, Any]] = json.loads(
        (args.edits if args.edits.is_absolute() else ROOT / args.edits).read_text(
            encoding="utf-8"
        )
    )
    if not requested_edits:
        raise SystemExit("--edits must contain at least one edit")

    apply_exact_byte_deterministic_runtime(before_cuda_init=not torch.cuda.is_initialized())

    events = load_case_events(cfg, case_id)
    for ev in events:
        if not getattr(ev, "sentence_tokens", None):
            ev.sentence_tokens = list(getattr(ev, "tokens", None) or [])
            ev.token_count = len(ev.sentence_tokens)
    by_event_id = {str(getattr(ev, "event_id", "")): ev for ev in events}

    side = load_embedding_sidecar(Path(cfg["embeddings_dir"]), case_id)
    sidecar: Dict[str, Any] = {}
    for row in side.itertuples(index=False):
        d = row._asdict() if hasattr(row, "_asdict") else dict(zip(side.columns, row))
        sidecar[str(d.get("event_id"))] = {
            "view": d.get("view", "UNKNOWN"),
            "zone": d.get("zone", 0),
            "case_age_hours": float(d.get("case_age_hours") or 0.0),
            "local_hour": int(float(d.get("local_hour") or 0)) % 24,
        }

    payload = [
        {
            "event_id": str(getattr(e, "event_id", "")),
            "tokens": list(getattr(e, "sentence_tokens", None) or []),
        }
        for e in events
    ]
    original_caseevents_sha = canonical_json_sha256(payload)

    cells = pd.read_parquet(Path(cfg["cells_path"]))
    farm_id, _start, _end = parse_case_id(case_id)
    tokenizer, _hashes = load_frozen_tokenizer_v2(
        feature_schema_path=Path(cfg["feature_schema_path"]),
        binning_registry_path=Path(cfg["binning_registry_path"]),
        vocab_path=Path(cfg["vocabulary_path"]),
        farm_relative_enabled=True,
    )

    atomics: List[ValidatedRawAtomicEdit] = []
    for i, req in enumerate(requested_edits):
        event_id = str(req["event_id"])
        feature = str(req["feature_id"])
        target_raw = float(req["target_raw"])
        ev = by_event_id.get(event_id)
        if ev is None:
            raise CoreContractError(f"event_id not found in case: {event_id}")
        tokens = list(getattr(ev, "sentence_tokens", None) or [])
        ts = getattr(ev, "timestamp", None)
        zone = getattr(ev, "zone_id", None)
        if zone is None:
            zone = getattr(ev, "zone", None)
        g = ground_locus_raw(
            cells=cells,
            farm_id=farm_id,
            feature=feature,
            zone_id=str(zone),
            timestamp=ts,
            event_id=event_id,
        )
        if not is_exact(g) or g.get("observed_raw") is None:
            raise CoreContractError(
                f"edit not groundable (no exact raw match): {event_id}/{feature}"
            )
        observed = float(g["observed_raw"])
        ret = apply_raw_edit_closure(
            event_id=event_id,
            feature=feature,
            target_raw=target_raw,
            original_tokens=tokens,
            tokenizer=tokenizer,
            farm_id=farm_id,
            observed_raw=observed,
            strict_exact=True,
        )
        new_tokens = list(ret.new_sentence_tokens or [])
        if not ret.baseline_roundtrip_valid or not ret.unchanged_outside_mg or not new_tokens:
            raise CoreContractError(
                f"edit rejected by gate0/gate4 closure: {event_id}/{feature} -> {target_raw}"
            )
        if new_tokens == tokens:
            raise CoreContractError(
                f"edit is a no-op after retokenization (same discretization bin): "
                f"{event_id}/{feature} {observed} -> {target_raw}"
            )
        grounding_sha = canonical_json_sha256(
            {
                "event_id": event_id,
                "feature": feature,
                "observed_raw": observed,
                "mg_id": g.get("measurement_group_id"),
                "status": g.get("status"),
            }
        )
        schema_sha = canonical_json_sha256({"feature": feature, "projection": "FeatureSpec_v1"})
        retok_sha = canonical_json_sha256(
            {
                "event_id": event_id,
                "original_tokens": tokens,
                "new_tokens": new_tokens,
                "target_raw": target_raw,
            }
        )
        atomics.append(
            ValidatedRawAtomicEdit(
                case_id=case_id,
                event_id=event_id,
                mg_id=str(g.get("measurement_group_id") or f"MG_{event_id}"),
                feature_id=feature,
                raw_field_path=f"raw.{feature}",
                operation="SET_RAW",
                canonical_value=target_raw,
                to_tokens=tuple(new_tokens),
                grounding_sha256=grounding_sha,
                schema_projection_sha256=schema_sha,
                retokenization_sha256=retok_sha,
                submitted_order=i,
            )
        )

    targeted = [a.raw_field_path for a in atomics]
    token_paths = [f"tokens.{a.event_id}" for a in atomics]
    allowed = build_allowed_change_set(
        targeted_raw_fields=targeted, tokenizer_derived_tokens=token_paths
    )
    required = list(targeted) + list(token_paths)
    assert_change_set_closed(
        allowed=allowed, actual_changed_paths=required, required_changed_paths=required
    )
    edit_tx = build_validated_raw_transaction(
        case_id=case_id,
        atomics=atomics,
        allowed_change_set=allowed,
        original_caseevents_sha=original_caseevents_sha,
        edited_caseevents_sha="pending_materialization",
        transaction_mode="CANDIDATE",
        identity=False,
        skip_edited_sha_required=True,
    )
    identity_tx = build_identity_transaction(
        case_id=case_id, original_caseevents_sha=original_caseevents_sha
    )

    reencoder = build_stage_a_reencoder(cfg, case_id=case_id)
    ckpts = discover_critic_checkpoints(cfg)
    all_fold_ids = list(range(len(ckpts)))
    requested_fold_ids = (
        [int(x) for x in args.fold_ids.split(",")] if args.fold_ids else all_fold_ids
    )
    fold_forward = make_production_fold_forward_fn(
        gpu_fraction=float(cfg.get("gpu_memory_fraction") or 0.4)
    )

    baseline = production_apply_edits_and_score(
        events=events,
        validated_transaction=identity_tx,
        reencoder=reencoder,
        fold_forward_fn=fold_forward,
        checkpoint_paths=ckpts,
        fold_ids=all_fold_ids,
        requested_fold_ids=requested_fold_ids,
        sidecar_by_event_id=sidecar,
        scope="search",
        transaction_mode="FORCED_IDENTITY",
        batch_size=args.batch_size,
    )
    edited = production_apply_edits_and_score(
        events=events,
        validated_transaction=edit_tx,
        reencoder=reencoder,
        fold_forward_fn=fold_forward,
        checkpoint_paths=ckpts,
        fold_ids=all_fold_ids,
        requested_fold_ids=requested_fold_ids,
        sidecar_by_event_id=sidecar,
        scope="search",
        transaction_mode="CANDIDATE",
        batch_size=args.batch_size,
    )

    by_fold: Dict[str, Any] = {}
    for fid in requested_fold_ids:
        key = str(fid)
        base_row = baseline["scores"]["by_fold"][key]
        edit_row = edited["scores"]["by_fold"][key]
        base_risk = float(base_row["risk"])
        edit_risk = float(edit_row["risk"])
        by_fold[key] = {
            "baseline_risk": base_risk,
            "edited_risk": edit_risk,
            "delta_risk": edit_risk - base_risk,
            "baseline_logit": _abnormal_logit(base_row),
            "edited_logit": _abnormal_logit(edit_row),
        }
    aggregate_delta = sum(v["delta_risk"] for v in by_fold.values()) / max(len(by_fold), 1)

    report = {
        "artifact_kind": "CF1S_INFERENCE_EDIT_EFFECT_V1",
        "interpretation_label": "INFERENCE_ONLY — not a governance-verified cohort result",
        "case_id": case_id,
        "requested_edits": requested_edits,
        "by_fold": by_fold,
        "aggregate_delta_risk": aggregate_delta,
        "reencode_mode": edited["reencode_mode"],
        "fold_ids_scored": requested_fold_ids,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
