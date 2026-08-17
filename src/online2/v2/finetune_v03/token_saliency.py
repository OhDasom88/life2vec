"""Token-level IxG for v0.3 — re-encodes ONE target event's Stage A window with
gradients enabled, splices its differentiable pooled vector into the (mostly
cached) case event tensor, and backprops the diagnosis objective through the
frozen pretrain encoder down to the input token embeddings.

Ports `src.online2.v2.v01_layer_saliency.token_ixg_for_event` (v0.1, frozen —
not modified here) to: (1) the v0.3 model interface
(`EventPoolingDiagnosisModelV03.forward`, task-query pooling, binary/fine
heads) instead of v0.1's plain model call, and (2) the 2026-07-26 single-
token-per-cell tokenizer (`FEATURE_NAME|value` combined tokens replacing the
old FEATURE|/OBSERVED_VALUE|/QUALITY|/STATE_SEMANTICS| split), so
`classify_token_type` correctly labels the new token shapes instead of
falling through to "other" for every real measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .model import EventPoolingDiagnosisModelV03
from .risk_saliency import risk_from_binary_logit, risk_margin_from_logits


def classify_token_type(token: str) -> str:
    """2026-07-26 셀당 토큰 1개(FEATURE_NAME|value) 스킴 기준 토큰 타입 분류.

    옛 v0.1 스킴(FEATURE|name + OBSERVED_VALUE|* 분리)은 더 이상 없다 -- 이제
    측정값은 전부 `{FEATURE_NAME}|{bin_or_raw}` 하나로 결합돼 있어 분류가
    오히려 단순해졌다."""
    t = str(token)
    if t in {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "[EVENT_SEP]", "[IMAGE_SLOT]", "[TEXT_SLOT]"}:
        return "special"
    if t.startswith("VIEW|"):
        return "view"
    if t.startswith("EVENT_KIND|"):
        return "meta"
    if t.startswith(("ZONE_LOCAL|", "FARM_LOCAL|", "NARRATIVE|")):
        return "context"
    if "|" in t:
        return "measurement"
    return "other"


@dataclass
class TokenIxGResultV03:
    event_id: str
    tokens: List[str]
    scores: np.ndarray  # [L_target], signed IxG (>0 pushes toward abnormal)
    token_types: List[str]
    pool_l2_err: float
    span_start: int
    span_end: int


def token_ixg_for_event_v03(
    *,
    encoder: nn.Module,
    model: EventPoolingDiagnosisModelV03,
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
    use_binary: bool = False,
    max_length: int = 3072,
) -> TokenIxGResultV03:
    """Re-encode one event via Stage A (grad-enabled), stitch its differentiable
    pooled vector into the case batch, IxG on the target event's input token
    embeddings w.r.t. the diagnosis objective. batch must be batch_size=1
    (same convention as risk_saliency.event_ixg_abnormal_margin)."""
    if batch["event_mean"].size(0) != 1:
        raise ValueError("token_ixg_for_event_v03 expects batch_size=1")

    window = stage_a_mod.construct_target_window(events, target_idx, max_length=max_length)
    x, pad_mask, _L = stage_a_mod.window_to_tensors(
        events, window, vocab=vocab, abspos_reference=abspos_reference,
        case_t0=case_t0, max_length=max_length,
    )
    x = x.unsqueeze(0).to(device)  # [1,4,L]
    pad = pad_mask.unsqueeze(0).to(device).long()
    s, e = int(window.target_token_start), int(window.target_token_end)
    target_ev = events[target_idx]
    # Must mirror window_to_tensors' own prefixing exactly (ZONE_LOCAL|i added
    # per-event when it has a real zone) or tokens_out silently misaligns by
    # one position against the real span content -- the display would show
    # e.g. "[EVENT_SEP]" at index 0 when the actual embedded token there is
    # "ZONE_LOCAL|i", shifting every score one token off.
    selected = [events[i] for i in window.event_indices]
    zones_in_window = sorted({ev.zone for ev in selected if stage_a_mod.zone_present(ev.zone)})
    zone_local = {z: f"ZONE_LOCAL|{i}" for i, z in enumerate(zones_in_window)}
    prefix_tok = [zone_local[target_ev.zone]] if target_ev.zone in zone_local else []
    tokens_out = (prefix_tok + target_ev.sentence_tokens)[: e - s]

    model.eval()
    encoder.eval()

    # Consistency check only (no grad): confirms the re-encoded window's pooled
    # vector matches what Stage A actually cached for this event (catches any
    # window-construction drift between caching time and this re-encode).
    with torch.no_grad():
        H0 = encoder.transformer.forward_finetuning(x=x, padding_mask=pad)
        span0 = H0[0, s:e, :]
        mean0 = span0.mean(dim=0) if span0.numel() else H0.new_zeros(H0.size(-1))
        cache_mean = batch["event_mean"][0, event_index].detach()
        pool_l2 = float(torch.norm(mean0 - cache_mean).item())

    emb, _ = encoder.transformer.get_sequence_embedding(x.long())
    emb = emb.detach().requires_grad_(True)
    H = encoder.transformer.forward_finetuning_with_embeddings(emb, pad)
    span = H[0, s:e, :]
    if span.numel() == 0:
        mean_v = H.new_zeros(H.size(-1))
        max_v = H.new_zeros(H.size(-1))
    else:
        mean_v = span.mean(dim=0)
        max_v = span.max(dim=0).values

    event_mean_t = batch["event_mean"].detach().clone()
    event_max_t = batch["event_max"].detach().clone()
    event_mean_t[0, event_index] = mean_v
    event_max_t[0, event_index] = max_v

    out = model(
        event_mean=event_mean_t,
        event_max=event_max_t,
        case_age_hours=batch["case_age_hours"],
        view_id=batch["view_id"],
        zone_id=batch["zone_id"],
        local_hour=batch["local_hour"],
        padding_mask=batch["padding_mask"],
        dino_vec=batch.get("dino_vec"),
        dino_mask=batch.get("dino_mask"),
    )
    if use_binary:
        objective = risk_from_binary_logit(out["abnormal_logit"][0])
    else:
        objective = risk_margin_from_logits(out["logits"], normal_class_id=normal_class_id)[0]

    model.zero_grad(set_to_none=True)
    if emb.grad is not None:
        emb.grad = None
    objective.backward()
    assert emb.grad is not None
    tok_score = (emb.grad * emb).sum(dim=-1)[0, s:e].detach().cpu().numpy().astype(np.float32)

    types = [classify_token_type(t) for t in tokens_out]
    if len(types) < len(tok_score):
        pad_n = len(tok_score) - len(types)
        types += ["other"] * pad_n
        tokens_out = list(tokens_out) + ["<?>"] * pad_n
    elif len(types) > len(tok_score):
        types = types[: len(tok_score)]
        tokens_out = tokens_out[: len(tok_score)]

    return TokenIxGResultV03(
        event_id=str(events[target_idx].event_id),
        tokens=list(tokens_out),
        scores=tok_score,
        token_types=types,
        pool_l2_err=pool_l2,
        span_start=s,
        span_end=e,
    )


def token_results_to_frame(results: Sequence[TokenIxGResultV03]) -> pd.DataFrame:
    rows = []
    for r in results:
        for tok, score, ttype, pos in zip(r.tokens, r.scores, r.token_types, range(len(r.tokens))):
            rows.append(
                {
                    "event_id": r.event_id,
                    "token": tok,
                    "score": float(score),
                    "token_type": ttype,
                    "position_in_span": pos,
                    "pool_l2_err": r.pool_l2_err,
                }
            )
    return pd.DataFrame(rows)
