"""Search-fold attribution → validated multi-event candidate construction (no production_bridge)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError
from .core_raw_transaction import (
    ValidatedRawAtomicEdit,
    assert_change_set_closed,
    build_allowed_change_set,
    build_canonical_atomic_evidence_manifest,
    build_validated_raw_transaction,
)
from .core_runtime import discover_critic_checkpoints, parse_case_id


def compute_search_event_preselect(
    cfg: Mapping[str, Any],
    *,
    case_id: str,
    top_k: int = 8,
    global_trace: Optional[Any] = None,
) -> pd.DataFrame:
    from ..attribution.token_ixg_v03 import preselect_events_search_folds

    ckpts = discover_critic_checkpoints(cfg)
    search_ids = list((cfg.get("critic") or {}).get("search_fold_ids") or [0, 1])
    search_ckpts = [ckpts[i] for i in search_ids if i < len(ckpts)]
    if len(search_ckpts) < 2:
        raise CoreContractError("search preselect requires fold 0/1 checkpoints")
    return preselect_events_search_folds(
        case_id=case_id,
        search_ckpts=search_ckpts,
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=Path(cfg["label_map_path"]),
        device=str(cfg.get("device") or "cuda"),
        gpu_fraction=float(cfg.get("gpu_memory_fraction") or 0.4),
        top_k=int(top_k),
        use_binary=True,
        global_trace=global_trace,
        trace_context={"phase": "ATTRIBUTION"},
    )


def _load_cells(cfg: Mapping[str, Any]) -> pd.DataFrame:
    path = Path(cfg["cells_path"])
    return pd.read_parquet(path)


def _caseevents_sha(events: Sequence[Any]) -> str:
    payload = []
    for ev in events:
        payload.append(
            {
                "event_id": str(getattr(ev, "event_id", "")),
                "tokens": list(
                    getattr(ev, "sentence_tokens", None)
                    or getattr(ev, "tokens", None)
                    or []
                ),
            }
        )
    return canonical_json_sha256(payload)


def _find_editable_atomics_for_events(
    cfg: Mapping[str, Any],
    *,
    case_id: str,
    events: Sequence[Any],
    event_ids: Sequence[str],
    cells: pd.DataFrame,
    prefer_features: Sequence[str] = ("substrate_temp_c", "inside_temp_c", "substrate_ec_ds_m"),
) -> List[Dict[str, Any]]:
    """Ground + retokenize one atomic edit per event for a shared feature when possible."""
    from ..grounding.locus_raw import ground_locus_raw, is_exact
    from ..retokenization.full_event_retokenizer import (
        apply_raw_edit_closure,
        load_frozen_tokenizer_v2,
    )

    farm_id, _start, _end = parse_case_id(case_id)
    tokenizer, _hashes = load_frozen_tokenizer_v2(
        feature_schema_path=Path(cfg["feature_schema_path"]),
        binning_registry_path=Path(cfg["binning_registry_path"]),
        vocab_path=Path(cfg["vocabulary_path"]),
        farm_relative_enabled=True,
    )
    by_id = {str(getattr(ev, "event_id", "")): ev for ev in events}
    atomics: List[Dict[str, Any]] = []
    chosen_feature: Optional[str] = None

    for feat in prefer_features:
        trial: List[Dict[str, Any]] = []
        ok = True
        for eid in event_ids:
            ev = by_id.get(str(eid))
            if ev is None:
                ok = False
                break
            tokens = list(
                getattr(ev, "sentence_tokens", None) or getattr(ev, "tokens", None) or []
            )
            ts = getattr(ev, "timestamp", None)
            zone = getattr(ev, "zone_id", None)
            if zone is None:
                zone = getattr(ev, "zone", None)
            g = ground_locus_raw(
                cells=cells,
                farm_id=farm_id,
                feature=feat,
                zone_id=str(zone),
                timestamp=ts,
                event_id=str(eid),
            )
            if not is_exact(g) or g.get("observed_raw") is None:
                ok = False
                break
            observed = float(g["observed_raw"])
            target = observed - abs(observed) * 0.05 if observed != 0 else observed - 0.1
            ret = apply_raw_edit_closure(
                event_id=str(eid),
                feature=feat,
                target_raw=float(target),
                original_tokens=tokens,
                tokenizer=tokenizer,
                farm_id=farm_id,
                observed_raw=observed,
                strict_exact=True,
            )
            new_tokens = list(ret.new_sentence_tokens or [])
            if (
                not ret.baseline_roundtrip_valid
                or not ret.unchanged_outside_mg
                or not new_tokens
                or new_tokens == tokens
            ):
                ok = False
                break
            grounding_sha = canonical_json_sha256(
                {
                    "event_id": str(eid),
                    "feature": feat,
                    "observed_raw": observed,
                    "mg_id": g.get("measurement_group_id"),
                    "status": g.get("status"),
                }
            )
            schema_sha = canonical_json_sha256({"feature": feat, "projection": "FeatureSpec_v1"})
            retok_sha = canonical_json_sha256(
                {
                    "event_id": str(eid),
                    "original_tokens": tokens,
                    "new_tokens": new_tokens,
                    "target_raw": float(target),
                }
            )
            raw_path = f"raw.{feat}"
            trial.append(
                {
                    "event_id": str(eid),
                    "feature_id": feat,
                    "mg_id": str(g.get("measurement_group_id") or f"MG_{eid}"),
                    "edit_direction": "DOWN",
                    "observed_raw": observed,
                    "target_raw": float(target),
                    "to_tokens": new_tokens,
                    "raw_field_path": raw_path,
                    "grounding_sha256": grounding_sha,
                    "schema_projection_sha256": schema_sha,
                    "retokenization_sha256": retok_sha,
                    "outside_mg_unchanged": bool(ret.unchanged_outside_mg),
                    "event_time_epoch": float(pd.Timestamp(ts).timestamp()) if ts is not None else 0.0,
                }
            )
        if ok and len(trial) == len(event_ids):
            chosen_feature = feat
            atomics = trial
            break

    if not atomics:
        return []
    for a in atomics:
        a["family_feature_id"] = chosen_feature
    return atomics


def _atomic_to_validated(
    *,
    case_id: str,
    atomic: Mapping[str, Any],
    submitted_order: int,
) -> ValidatedRawAtomicEdit:
    return ValidatedRawAtomicEdit(
        case_id=case_id,
        event_id=str(atomic["event_id"]),
        mg_id=str(atomic["mg_id"]),
        feature_id=str(atomic["feature_id"]),
        raw_field_path=str(atomic["raw_field_path"]),
        operation="SET_RAW",
        canonical_value=float(atomic["target_raw"]),
        to_tokens=tuple(atomic["to_tokens"]),
        grounding_sha256=str(atomic["grounding_sha256"]),
        schema_projection_sha256=str(atomic["schema_projection_sha256"]),
        retokenization_sha256=str(atomic["retokenization_sha256"]),
        submitted_order=int(submitted_order),
    )


def _build_tx_from_atomics(
    *,
    case_id: str,
    atomics_raw: Sequence[Mapping[str, Any]],
    original_caseevents_sha: str,
) -> Dict[str, Any]:
    validated_atomics = [
        _atomic_to_validated(case_id=case_id, atomic=a, submitted_order=i)
        for i, a in enumerate(atomics_raw)
    ]
    targeted = [a.raw_field_path for a in validated_atomics]
    token_paths = [f"tokens.{a.event_id}" for a in validated_atomics]
    allowed = build_allowed_change_set(
        targeted_raw_fields=targeted,
        tokenizer_derived_tokens=token_paths,
    )
    required = list(targeted) + list(token_paths)
    assert_change_set_closed(
        allowed=allowed,
        actual_changed_paths=required,
        required_changed_paths=required,
    )
    # edited_caseevents_sha filled after materialization; pending placeholder for construction
    tx = build_validated_raw_transaction(
        case_id=case_id,
        atomics=validated_atomics,
        allowed_change_set=allowed,
        original_caseevents_sha=original_caseevents_sha,
        edited_caseevents_sha="pending_materialization",
        transaction_mode="CANDIDATE",
        identity=False,
        skip_edited_sha_required=True,
    )
    atomic_manifest = build_canonical_atomic_evidence_manifest(validated_atomics)
    return {
        "validated_transaction": tx,
        "atomic_evidence_manifest": atomic_manifest,
        "gate0_status": "PASS",
        "gate4_status": "PASS",
        "outside_mg_unchanged": all(bool(a.get("outside_mg_unchanged")) for a in atomics_raw),
        "allowed_change_set_sha": allowed.sha256,
        "actual_changed_paths_sha": canonical_json_sha256(sorted(required)),
        "required_changed_paths_sha": canonical_json_sha256(sorted(required)),
    }


def build_multievent_candidates_for_case(
    cfg: Mapping[str, Any],
    *,
    case_id: str,
    events: Sequence[Any],
    max_two_event: int = 2,
    top_k_events: int = 8,
    global_trace: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Construct up to max_two_event AB bundles from search preselect + grounded retokenization.
    Parents A/B are emitted as separate candidates for dependency closure.
    expected_effect_direction is intentionally unset until Search eligibility.
    """
    pre = compute_search_event_preselect(
        cfg,
        case_id=case_id,
        top_k=top_k_events,
        global_trace=global_trace,
    )
    if pre is None or len(pre) < 2:
        return []
    ordered_ids = [
        str(x)
        for x in pre.sort_values("absolute_attribution", ascending=False)["event_id"].tolist()
    ]
    cells = _load_cells(cfg)
    original_sha = _caseevents_sha(events)

    candidates: List[Dict[str, Any]] = []
    pairs: List[Tuple[str, str]] = []
    for i in range(len(ordered_ids) - 1):
        pairs.append((ordered_ids[i], ordered_ids[i + 1]))
        if len(pairs) >= max_two_event * 3:
            break

    for a_id, b_id in pairs:
        atomics = _find_editable_atomics_for_events(
            cfg,
            case_id=case_id,
            events=events,
            event_ids=[a_id, b_id],
            cells=cells,
        )
        if len(atomics) != 2:
            continue
        for atomic in atomics:
            cid = f"ATOMIC::{atomic['event_id']}::{atomic['feature_id']}"
            if any(c["candidate_id"] == cid for c in candidates):
                continue
            built = _build_tx_from_atomics(
                case_id=case_id,
                atomics_raw=[atomic],
                original_caseevents_sha=original_sha,
            )
            tx = built["validated_transaction"]
            candidates.append(
                {
                    "candidate_id": cid,
                    "event_ids": [atomic["event_id"]],
                    "kind": "ATOMIC",
                    "expected_effect_direction": None,  # Search-derived later
                    "validated_transaction": tx,
                    "transaction_sha": tx.canonical_validated_transaction_sha,
                    "atomics": [atomic],
                    "atomic_evidence_manifest": built["atomic_evidence_manifest"],
                    "gate0_status": built["gate0_status"],
                    "gate4_status": built["gate4_status"],
                    "outside_mg_unchanged": built["outside_mg_unchanged"],
                    "allowed_change_set_sha": built["allowed_change_set_sha"],
                    "actual_changed_paths_sha": built["actual_changed_paths_sha"],
                    "required_changed_paths_sha": built["required_changed_paths_sha"],
                }
            )
        built_pair = _build_tx_from_atomics(
            case_id=case_id,
            atomics_raw=atomics,
            original_caseevents_sha=original_sha,
        )
        tx_pair = built_pair["validated_transaction"]
        bundle_id = f"PAIR::{a_id}::{b_id}::{atomics[0]['feature_id']}"
        atomic_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("kind") == "ATOMIC"
            and set(candidate.get("event_ids") or []).issubset({a_id, b_id})
        ]
        atomic_candidates = sorted(
            atomic_candidates,
            key=lambda candidate: (
                tuple(candidate.get("event_ids") or []),
                str(candidate.get("candidate_id")),
            ),
        )
        if len(atomic_candidates) != 2:
            raise CoreContractError(
                f"pair {bundle_id} requires exactly two canonical atomic parents"
            )
        parent_candidate_ids = [str(candidate["candidate_id"]) for candidate in atomic_candidates]
        parent_transaction_shas = [
            str(candidate["transaction_sha"]) for candidate in atomic_candidates
        ]
        candidates.append(
            {
                "candidate_id": bundle_id,
                "event_ids": [a_id, b_id],
                "kind": "PAIR",
                "expected_effect_direction": None,
                "validated_transaction": tx_pair,
                "transaction_sha": tx_pair.canonical_validated_transaction_sha,
                "parent_candidate_ids": parent_candidate_ids,
                "parent_transaction_shas": parent_transaction_shas,
                "atomics": atomics,
                "atomic_evidence_manifest": built_pair["atomic_evidence_manifest"],
                "gate0_status": built_pair["gate0_status"],
                "gate4_status": built_pair["gate4_status"],
                "outside_mg_unchanged": built_pair["outside_mg_unchanged"],
                "allowed_change_set_sha": built_pair["allowed_change_set_sha"],
                "actual_changed_paths_sha": built_pair["actual_changed_paths_sha"],
                "required_changed_paths_sha": built_pair["required_changed_paths_sha"],
                "universe_component_sha": canonical_json_sha256(
                    {"a": a_id, "b": b_id, "feature": atomics[0]["feature_id"]}
                ),
            }
        )
        if sum(1 for c in candidates if c["kind"] == "PAIR") >= max_two_event:
            break
    return candidates
