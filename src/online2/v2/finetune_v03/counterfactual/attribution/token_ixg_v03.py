"""True token IxG for v0.3 CF (search-fold only, binary_abnormal_logit objective)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03
from src.online2.v2.finetune_v03.risk_saliency import (
    event_ixg_abnormal_margin,
    risk_margin_from_logits,
)

from ..attribution.aggregation import infer_token_role
from ..attribution.fold_consensus import (
    consensus_mean,
    normalize_fold_scores,
    robust_scale,
    sign_agreement_mask,
)
from ..stage_a_loader import load_stage_a_module


OBJECTIVE_BINARY = "binary_abnormal_logit"
OBJECTIVE_MARGIN = "abnormal_margin"


def _pool_target_span(H: torch.Tensor, s: int, e: int) -> Tuple[torch.Tensor, torch.Tensor]:
    span = H[0, s:e, :]
    return span.mean(dim=0), span.max(dim=0).values


def _objective(model, h_case: torch.Tensor, *, normal_class_id: int, objective_id: str) -> torch.Tensor:
    if objective_id == OBJECTIVE_BINARY and hasattr(model, "binary_head"):
        return model.binary_head(h_case)[0]
    logits = model.fine_head(h_case) if hasattr(model, "fine_head") else model.head(h_case)
    return risk_margin_from_logits(logits, normal_class_id=normal_class_id)[0]


@torch.enable_grad()
def token_ixg_for_event_v03(
    *,
    encoder,
    diagnosis_models: Sequence[Any],
    stage_a_mod: Any,
    events: list,
    target_idx: int,
    case_t0: pd.Timestamp,
    abspos_reference: pd.Timestamp,
    vocab: Any,
    batch: Dict[str, torch.Tensor],
    event_index: int,
    normal_class_id: int,
    device: torch.device,
    max_length: int = 1024,
    objective_id: str = OBJECTIVE_BINARY,
    min_abs_score_per_fold: float = 1.0e-5,
    require_sign_agreement: bool = True,
) -> Dict[str, Any]:
    """Re-encode one event via Stage A, stitch into case, IxG on target tokens."""
    window = stage_a_mod.construct_target_window(events, target_idx, max_length=max_length)
    x, pad_mask, _L = stage_a_mod.window_to_tensors(
        events,
        window,
        vocab=vocab,
        abspos_reference=abspos_reference,
        case_t0=case_t0,
        max_length=max_length,
    )
    x = x.unsqueeze(0).to(device)
    pad = pad_mask.unsqueeze(0).to(device).long()
    s, e = int(window.target_token_start), int(window.target_token_end)
    target_tokens = list(events[target_idx].sentence_tokens)
    span_len = e - s
    tokens_out = target_tokens[:span_len]

    with torch.no_grad():
        H0 = encoder.transformer.forward_finetuning(x=x, padding_mask=pad)
        mean0, _max0 = _pool_target_span(H0, s, e)
        cache_mean = batch["event_mean"][0, event_index].detach()
        pool_l2 = float(torch.norm(mean0 - cache_mean).item())

    fold_score_list: List[torch.Tensor] = []
    used_objective = objective_id
    for model in diagnosis_models:
        model.eval()
        emb, _ = encoder.transformer.get_sequence_embedding(x.long())
        emb = emb.detach().requires_grad_(True)
        H = encoder.transformer.forward_finetuning_with_embeddings(emb, pad)
        mean_v, max_v = _pool_target_span(H, s, e)

        event_mean_t = batch["event_mean"].detach().clone()
        event_max_t = batch["event_max"].detach().clone()
        event_mean_t[0, event_index] = mean_v
        event_max_t[0, event_index] = max_v

        if hasattr(model, "fuse_image_into_events"):
            event_mean_t, event_max_t = model.fuse_image_into_events(
                event_mean_t,
                event_max_t,
                dino_vec=batch.get("dino_vec"),
                dino_mask=batch.get("dino_mask"),
            )
        z = model.meta(
            event_mean_t,
            event_max_t,
            batch["case_age_hours"],
            batch["view_id"],
            batch["zone_id"],
            batch["local_hour"],
            batch["padding_mask"],
        )
        h_case, _ = model.pad(z, batch["padding_mask"])
        obj_id = objective_id
        if obj_id == OBJECTIVE_BINARY and not hasattr(model, "binary_head"):
            obj_id = OBJECTIVE_MARGIN
            used_objective = OBJECTIVE_MARGIN
        objective = _objective(model, h_case, normal_class_id=normal_class_id, objective_id=obj_id)
        model.zero_grad(set_to_none=True)
        if emb.grad is not None:
            emb.grad = None
        objective.backward()
        assert emb.grad is not None
        tok_score = (emb.grad * emb).sum(dim=-1)[0, s:e].detach()
        fold_score_list.append(tok_score)

    stack = torch.stack(fold_score_list, dim=0)
    fold_np = stack.detach().cpu().numpy().astype(np.float64)
    # L1-normalize each fold, then mean consensus (v3 §3.9)
    norm_folds = normalize_fold_scores(fold_np, method="l1")
    consensus_scores = consensus_mean(norm_folds).astype(np.float32)
    sign_ok = sign_agreement_mask(
        fold_np,
        min_abs_score_per_fold=float(min_abs_score_per_fold),
        require_all_folds=True,
    )
    grad_abs_sum = float(np.abs(consensus_scores).sum())
    return {
        "event_id": str(events[target_idx].event_id),
        "event_index": int(event_index),
        "tokens": tokens_out,
        "scores": consensus_scores,
        "fold_scores": fold_np.astype(np.float32),
        "fold_scores_l1": norm_folds.astype(np.float32),
        "sign_agreement_mask": sign_ok.astype(np.bool_),
        "fold_normalization": "l1",
        "pool_l2_err": pool_l2,
        "token_grad_abs_sum": grad_abs_sum,
        "objective_id": used_objective,
        "span_start": s,
        "span_end": e,
    }


def preselect_events_search_folds(
    *,
    case_id: str,
    search_ckpts: Sequence[Path],
    embeddings_dir: Path,
    labels_path: Path,
    label_map_path: Path,
    device: str = "cuda",
    gpu_fraction: float = 0.4,
    top_k: int = 8,
    use_binary: bool = True,
) -> pd.DataFrame:
    """Coarse event IxG using search folds only."""
    from ..attribution.token_attribution import compute_event_attribution_for_case

    ev = compute_event_attribution_for_case(
        case_id=case_id,
        ckpt_paths=search_ckpts,
        embeddings_dir=embeddings_dir,
        labels_path=labels_path,
        label_map_path=label_map_path,
        device=device,
        gpu_fraction=gpu_fraction,
        use_binary=use_binary,
    )
    # median across search folds
    g = (
        ev.groupby("event_id", as_index=False)
        .agg(
            signed_attribution=("signed_attribution", "median"),
            absolute_attribution=("absolute_attribution", "median"),
            event_index=("event_index", "first"),
            view=("view", "first"),
            zone=("zone", "first"),
            timestamp=("timestamp", "first"),
        )
    )
    g = g.sort_values("absolute_attribution", ascending=False).head(int(top_k))
    g["event_preselector"] = "search_fold_event_ixg"
    return g.reset_index(drop=True)


def compute_token_ixg_for_case(
    *,
    case_id: str,
    events: list,
    batch: Dict[str, torch.Tensor],
    side: pd.DataFrame,
    search_ckpts: Sequence[Path],
    encoder,
    stage_a_mod,
    vocab,
    abspos_reference: pd.Timestamp,
    case_t0: pd.Timestamp,
    normal_class_id: int,
    preselected: pd.DataFrame,
    device: torch.device,
    max_length: int = 1024,
    objective_id: str = OBJECTIVE_BINARY,
    measurement_group_from_tokens: bool = True,
    min_abs_score_per_fold: float = 1.0e-5,
    require_sign_agreement: bool = True,
    fold_normalization: str = "l1",
) -> pd.DataFrame:
    """Run token IxG on preselected events with search-fold diagnosis models only."""
    models = []
    for ckpt in search_ckpts:
        m, _meta = load_fold_model_v03(Path(ckpt), device=device)
        models.append(m)

    eid_to_case_idx = {str(r.event_id): int(r.event_index) for r in side.itertuples()}
    eid_to_events_idx = {str(ev.event_id): i for i, ev in enumerate(events)}

    rows = []
    for _, prow in preselected.iterrows():
        eid = str(prow["event_id"])
        if eid not in eid_to_case_idx or eid not in eid_to_events_idx:
            continue
        event_index = eid_to_case_idx[eid]
        target_idx = eid_to_events_idx[eid]
        # diagnosis batch truncation
        if event_index >= int(batch["padding_mask"].shape[1]):
            continue
        if not bool(batch["padding_mask"][0, event_index].item()):
            continue
        result = token_ixg_for_event_v03(
            encoder=encoder,
            diagnosis_models=models,
            stage_a_mod=stage_a_mod,
            events=events,
            target_idx=target_idx,
            case_t0=case_t0,
            abspos_reference=abspos_reference,
            vocab=vocab,
            batch={k: v for k, v in batch.items() if torch.is_tensor(v)},
            event_index=event_index,
            normal_class_id=normal_class_id,
            device=device,
            max_length=max_length,
            objective_id=objective_id,
            min_abs_score_per_fold=float(min_abs_score_per_fold),
            require_sign_agreement=bool(require_sign_agreement),
        )
        scores = result["scores"]
        if float(np.abs(scores).sum()) <= 0:
            raise RuntimeError(f"token IxG collapsed to zero for event={eid}")
        norm = robust_scale(scores)
        sign_mask = result.get("sign_agreement_mask")
        # crude MG: group consecutive value tokens with preceding FEATURE
        mg_id = "mg0"
        feature = "unknown"
        for j, tok in enumerate(result["tokens"]):
            role = infer_token_role(tok)
            if tok.startswith("FEATURE|"):
                feature = tok.split("|", 1)[-1]
                mg_id = f"mg:{feature}"
            sign_ok = bool(sign_mask[j]) if sign_mask is not None and j < len(sign_mask) else True
            rows.append(
                {
                    "case_id": case_id,
                    "fold_id": -1,  # L1-mean consensus; per-fold in fold_scores
                    "event_id": eid,
                    "event_index": event_index,
                    "measurement_group_id": mg_id,
                    "feature": feature,
                    "token_index": j,
                    "token_string": tok,
                    "token_role": role,
                    "signed_attribution": float(scores[j]),
                    "absolute_attribution": float(abs(scores[j])),
                    "normalized_signed_attribution": float(norm[j]),
                    "normalized_absolute_attribution": float(abs(norm[j])),
                    "sign_agreement": sign_ok,
                    "fold_normalization": result.get("fold_normalization", fold_normalization),
                    "attribution_method": "token_ixg_v03",
                    "objective_id": result["objective_id"],
                    "token_grad_abs_sum": result["token_grad_abs_sum"],
                    "pool_l2_err": result["pool_l2_err"],
                    "event_preselector": str(prow.get("event_preselector") or "search_fold_event_ixg"),
                }
            )
    for m in models:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("token_ixg_v03 produced zero token rows")
    return df
