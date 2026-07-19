"""CF-1S Core production transaction: cold rebuild + fold-subset scoring."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError, sha256_bytes, sha256_file, sha256_json
from .core_manifest import FoldForwardTrace
from .core_raw_transaction import ValidatedRawTransaction, reject_unchecked_token_dict


def tensor_bytes_sha256(t: torch.Tensor) -> str:
    arr = t.detach().float().cpu().contiguous().numpy()
    return sha256_bytes(arr.tobytes())


def select_fold_checkpoints(
    *,
    checkpoint_paths: Sequence[Any],
    fold_ids: Sequence[int],
    requested_fold_ids: Sequence[int],
) -> Dict[str, Any]:
    """Pre-select only requested fold checkpoints; forbid non-requested forwards."""
    paths = list(checkpoint_paths)
    fids = [int(x) for x in fold_ids]
    if len(paths) != len(fids):
        raise CoreContractError("checkpoint/fold length mismatch")
    req = [int(x) for x in requested_fold_ids]
    selected_paths = []
    selected_folds = []
    for p, f in zip(paths, fids):
        if f in req:
            selected_paths.append(p)
            selected_folds.append(f)
    missing = [f for f in req if f not in selected_folds]
    if missing:
        raise CoreContractError(f"requested folds missing from checkpoint set: {missing}")
    return {
        "requested_fold_ids": req,
        "selected_fold_ids": selected_folds,
        "selected_checkpoint_paths": selected_paths,
        "nonrequested_fold_ids_excluded": [f for f in fids if f not in set(req)],
    }


def score_requested_folds_only(
    *,
    fold_forward_fn: Callable[..., Mapping[str, Any]],
    selected_checkpoint_paths: Sequence[Any],
    selected_fold_ids: Sequence[int],
    diagnosis_batch: Optional[Mapping[str, Any]] = None,
    trace: Optional[FoldForwardTrace] = None,
    scope: str = "search",
) -> Dict[str, Any]:
    """Score only requested folds. fold_forward_fn(checkpoint_path, diagnosis_batch)."""
    if diagnosis_batch is None:
        raise CoreContractError("diagnosis_batch required for fold scoring")
    paths = list(selected_checkpoint_paths)
    folds = [int(x) for x in selected_fold_ids]
    if len(paths) != len(folds):
        raise CoreContractError("selected checkpoint/fold length mismatch")
    if len(folds) != len(set(folds)):
        raise CoreContractError("duplicate fold ids in selected set")
    by_fold: Dict[str, Any] = {}
    for path, fold_id in zip(paths, folds):
        if getattr(fold_forward_fn, "_cf1s_accepts_fold_id", False):
            out = fold_forward_fn(
                path,
                diagnosis_batch,
                _cf1s_fold_id=fold_id,
            )
        else:
            out = fold_forward_fn(path, diagnosis_batch)
        if "risk" not in out or "logit" not in out:
            raise CoreContractError("fold_forward_fn must return risk and logit")
        risk = float(out["risk"])
        logit = [float(x) for x in out["logit"]]
        if not math.isfinite(risk) or any(not math.isfinite(x) for x in logit):
            raise CoreContractError("non-finite risk/logit from fold forward")
        if trace is not None:
            trace.record(scope=scope, fold_ids=[fold_id])
        by_fold[str(int(fold_id))] = {
            "risk": risk,
            "logit": logit,
            "checkpoint": str(path),
            "checkpoint_sha256": (
                sha256_file(path)
                if hasattr(path, "is_file") and path.is_file()
                else canonical_json_sha256({"unresolved_checkpoint_path": str(path)})
            ),
            "semantic_critic_input_sha": out.get("semantic_critic_input_sha"),
        }
    return {
        "scope": scope,
        "requested_fold_ids": folds,
        "by_fold": by_fold,
        "forwarded_fold_count": len(folds),
    }


def apply_sentence_token_edits(
    events: list,
    edits: Sequence[Mapping[str, Any]],
    *,
    allow_noop: bool = False,
) -> Tuple[list, List[str]]:
    events_edit = deepcopy(events)
    id_to_idx = {
        str(getattr(ev, "event_id", "")): i for i, ev in enumerate(events_edit)
    }
    edited_ids: List[str] = []
    for ed in edits:
        eid = str(ed["event_id"])
        if eid not in id_to_idx:
            raise CoreContractError(f"edit event not in valid sequence: {eid}")
        i = id_to_idx[eid]
        new_tokens = list(ed.get("to_tokens") or ed.get("sentence_tokens") or [])
        if not new_tokens:
            raise CoreContractError(f"empty retokenization for event {eid}")
        old = list(getattr(events_edit[i], "sentence_tokens", []) or [])
        if old == new_tokens and not allow_noop:
            # normal candidates: no-op removed
            continue
        events_edit[i].sentence_tokens = list(new_tokens)
        events_edit[i].token_count = len(new_tokens)
        edited_ids.append(eid)
    if not edited_ids and not allow_noop:
        raise CoreContractError("NO_OP")
    return events_edit, edited_ids


def production_apply_edits_and_score(
    *,
    events: list,
    edits: Any = None,
    validated_transaction: Optional[ValidatedRawTransaction] = None,
    reencoder: Any,
    fold_forward_fn: Callable[..., Mapping[str, Any]],
    checkpoint_paths: Sequence[Any],
    fold_ids: Sequence[int],
    requested_fold_ids: Sequence[int],
    sidecar_by_event_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
    baseline_batch_template: Optional[Mapping[str, Any]] = None,
    trace: Optional[FoldForwardTrace] = None,
    scope: str = "search",
    transaction_mode: str = "CANDIDATE",
    batch_size: int = 8,
    # Rejected: completed diagnosis_batch must never bypass cold rebuild.
    diagnosis_batch: Any = None,
) -> Dict[str, Any]:
    """Core transaction: ValidatedRawTransaction → cold rebuild → fresh diagnosis batch → score."""
    if diagnosis_batch is not None:
        raise CoreContractError(
            "STALE_DIAGNOSIS_BATCH: completed diagnosis_batch injection forbidden; "
            "batch is always rebuilt from current cold Stage-A output"
        )
    tx = validated_transaction
    if tx is None:
        # Legacy kwarg name `edits` may carry a ValidatedRawTransaction only.
        if isinstance(edits, ValidatedRawTransaction):
            tx = edits
        else:
            reject_unchecked_token_dict(edits)
            raise CoreContractError("validated_transaction required")
    reject_unchecked_token_dict(tx)
    transaction_mode = tx.transaction_mode or transaction_mode
    allow_noop = bool(tx.identity) or transaction_mode == "FORCED_IDENTITY"
    apply_edits = tx.edits_for_apply()
    events_edit, edited_ids = apply_sentence_token_edits(
        events, apply_edits, allow_noop=allow_noop
    )
    cold = reencoder.cold_rebuild_full_sequence(events_edit, batch_size=batch_size)
    selected = select_fold_checkpoints(
        checkpoint_paths=checkpoint_paths,
        fold_ids=fold_ids,
        requested_fold_ids=requested_fold_ids,
    )
    from .core_canonical import semantic_critic_input_sha, stage_a_output_sha
    from .core_production import assert_batch_matches_cold, build_cold_diagnosis_batch

    sidecar = sidecar_by_event_id or {
        str(eid): {} for eid in cold.get("valid_event_ids") or []
    }
    batch = build_cold_diagnosis_batch(
        cold,
        sidecar_by_event_id=sidecar,
        baseline_batch=baseline_batch_template,
    )
    assert_batch_matches_cold(batch, cold)
    scores = score_requested_folds_only(
        fold_forward_fn=fold_forward_fn,
        selected_checkpoint_paths=selected["selected_checkpoint_paths"],
        selected_fold_ids=selected["selected_fold_ids"],
        diagnosis_batch=batch,
        trace=trace,
        scope=scope,
    )
    sa_sha = stage_a_output_sha(
        event_mean=cold["event_mean"],
        event_max=cold["event_max"],
        valid_event_ids=cold["valid_event_ids"],
    )
    return {
        "transaction_mode": transaction_mode,
        "canonical_validated_transaction_sha": tx.canonical_validated_transaction_sha,
        "canonical_atomic_set_sha": tx.canonical_atomic_set_sha,
        "allowed_change_set_sha": tx.allowed_change_set.sha256,
        "original_caseevents_sha": tx.original_caseevents_sha,
        "edited_caseevents_sha": tx.edited_caseevents_sha,
        "edited_event_ids": edited_ids,
        "reencode_mode": "FULL_EVENT_COVERAGE_COLD_REBUILD",
        "affected_window_only_mode": False,
        "untouched_embedding_cache_reuse_count": 0,
        "stale_stage_a_cache_read_count": 0,
        "all_valid_target_windows_reencoded": True,
        "all_valid_event_embeddings_reconstructed": True,
        "incremental_reencode_used": False,
        "parity_status": "NOT_APPLICABLE",
        "stage_a_output_sha": sa_sha,
        "stage_a_output_hash": {
            "event_mean": tensor_bytes_sha256(cold["event_mean"]),
            "event_max": tensor_bytes_sha256(cold["event_max"]),
        },
        "diagnosis_batch_sha": batch.get("diagnosis_batch_sha"),
        "semantic_critic_input_sha": semantic_critic_input_sha(batch),
        "valid_event_ids": cold["valid_event_ids"],
        "window_ids": cold["window_ids"],
        "target_membership": cold["target_membership"],
        "context_membership": cold["context_membership"],
        "cap_provenance": {
            "pre_cap_valid_event_count": cold["pre_cap_valid_event_count"],
            "post_cap_valid_event_count": cold["post_cap_valid_event_count"],
            "valid_event_cap_applied": cold["valid_event_cap_applied"],
            "excluded_tail_event_count": cold["excluded_tail_event_count"],
        },
        "fold_selection": selected,
        "scores": scores,
        "cold": {k: v for k, v in cold.items() if k not in {"event_mean", "event_max"}},
        "event_mean": cold["event_mean"],
        "event_max": cold["event_max"],
        "diagnosis_batch": {
            k: v
            for k, v in batch.items()
            if k
            not in {
                "event_mean",
                "event_max",
                "dino_vec",
                "dino_mask",
                "case_age_hours",
                "view_id",
                "zone_id",
                "local_hour",
                "padding_mask",
            }
        },
    }


def independent_cold_build_reference(
    *,
    events: list,
    stage_a_mod: Any,
    encoder: Any,
    vocab: Any,
    abspos_reference: Any,
    case_t0: Any,
    device: torch.device,
    max_length: int = 1024,
    special_token_budget: int = 6,
    batch_size: int = 8,
    valid_event_cap: int = 4096,
) -> Dict[str, Any]:
    """Independent reference path — must NOT call StageAReencoder.cold_rebuild_full_sequence."""
    max_target = int(max_length) - int(special_token_budget)
    capped = list(events)[: int(valid_event_cap)]
    valid_indices = []
    valid_event_ids = []
    window_ids = []
    target_membership: Dict[str, List[str]] = {}
    context_membership: Dict[str, List[str]] = {}
    xs_all = []
    masks_all = []
    spans_all = []
    for i, ev in enumerate(capped):
        tokens = list(getattr(ev, "sentence_tokens", None) or [])
        tc = int(getattr(ev, "token_count", len(tokens)) or len(tokens))
        eid = str(getattr(ev, "event_id", f"idx_{i}"))
        if tc < 1 or tc > max_target or not tokens:
            continue
        window = stage_a_mod.construct_target_window(capped, i, max_length=max_length)
        wid = f"window_target_{eid}"
        valid_indices.append(i)
        valid_event_ids.append(eid)
        window_ids.append(wid)
        target_membership[eid] = [wid]
        for ctx_i in window.event_indices:
            ctx_id = str(getattr(capped[int(ctx_i)], "event_id", f"idx_{ctx_i}"))
            context_membership.setdefault(ctx_id, []).append(wid)
        x, pad_mask, _L = stage_a_mod.window_to_tensors(
            capped,
            window,
            vocab=vocab,
            abspos_reference=abspos_reference,
            case_t0=case_t0,
            max_length=max_length,
        )
        xs_all.append(x)
        masks_all.append(pad_mask)
        spans_all.append((int(window.target_token_start), int(window.target_token_end)))

    means = []
    maxes = []
    bs = max(1, int(batch_size))
    for start in range(0, len(xs_all), bs):
        pooled = stage_a_mod.encode_batch_pool(
            encoder,
            xs_all[start : start + bs],
            masks_all[start : start + bs],
            spans_all[start : start + bs],
            device=device,
        )
        for mean_np, max_np in pooled:
            means.append(torch.as_tensor(mean_np, dtype=torch.float32))
            maxes.append(torch.as_tensor(max_np, dtype=torch.float32))
    return {
        "reference_calls_core_cold_build_method": False,
        "reference_uses_fresh_sequence_materialization": True,
        "reference_uses_empty_embedding_cache": True,
        "valid_event_ids": valid_event_ids,
        "window_ids": window_ids,
        "target_membership": target_membership,
        "context_membership": context_membership,
        "event_mean": torch.stack(means, dim=0) if means else torch.zeros(0, 1),
        "event_max": torch.stack(maxes, dim=0) if maxes else torch.zeros(0, 1),
    }


def compare_cold_build_equivalence(
    core: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    id_eq = list(core.get("valid_event_ids") or []) == list(reference.get("valid_event_ids") or [])
    win_eq = list(core.get("window_ids") or []) == list(reference.get("window_ids") or [])
    tgt_eq = dict(core.get("target_membership") or {}) == dict(
        reference.get("target_membership") or {}
    )
    ctx_eq = dict(core.get("context_membership") or {}) == dict(
        reference.get("context_membership") or {}
    )
    mean_ok = False
    max_ok = False
    if id_eq and torch.is_tensor(core.get("event_mean")) and torch.is_tensor(
        reference.get("event_mean")
    ):
        mean_ok = bool(
            torch.allclose(
                core["event_mean"].float().cpu(),
                reference["event_mean"].float().cpu(),
                atol=atol,
                rtol=rtol,
            )
        )
        max_ok = bool(
            torch.allclose(
                core["event_max"].float().cpu(),
                reference["event_max"].float().cpu(),
                atol=atol,
                rtol=rtol,
            )
        )
    passed = id_eq and win_eq and tgt_eq and ctx_eq and mean_ok and max_ok
    return {
        "core_reference_valid_event_id_set_equal": id_eq,
        "core_reference_window_id_order_equal": win_eq,
        "core_reference_event_to_window_membership_equal": tgt_eq and ctx_eq,
        "event_mean_allclose": mean_ok,
        "event_max_allclose": max_ok,
        "cold_rebuild_equivalence_test_pass": passed,
        "reference_calls_core_cold_build_method": False,
    }
