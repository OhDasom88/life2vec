"""Constrained MLM bundle candidates from observed token bundles.

Uses deterministic masks + joint log-prob over observed bundles only.
GroupedMLMMasker is never used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .inference_masker import InferenceMaskResult, mask_measurement_group


@dataclass
class BundleCandidate:
    source: str
    tokens: List[str]
    token_ids: List[int]
    score: float
    is_original: bool
    feature: str
    measurement_group_id: str


def role_signature(roles: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(r) for r in roles)


def build_observed_bundle_bank(
    rows: Iterable[Mapping[str, Any]],
    *,
    exclude_event_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build feature -> list of observed value-token bundles.

    Each row needs: feature, event_id, tokens (value tokens), token_ids, roles.
    """
    bank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        eid = str(row.get("event_id") or "")
        if exclude_event_id and eid == str(exclude_event_id):
            continue
        feat = str(row["feature"])
        toks = tuple(str(t) for t in row["tokens"])
        key = (feat, toks)
        if key in seen:
            continue
        seen.add(key)
        bank[feat].append(
            {
                "feature": feat,
                "tokens": list(toks),
                "token_ids": [int(x) for x in row["token_ids"]],
                "roles": [str(r) for r in row.get("roles") or []],
                "event_id": eid,
                "signature": role_signature(row.get("roles") or []),
            }
        )
    return dict(bank)


def filter_bundles(
    bank: Sequence[Mapping[str, Any]],
    *,
    feature: str,
    expected_signature: Sequence[str],
    original_tokens: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    sig = role_signature(expected_signature)
    out = []
    for b in bank:
        if str(b.get("feature")) != str(feature):
            continue
        if tuple(b.get("signature") or ()) != sig and sig:
            # if bank has signature, enforce; empty expected allows length match
            if b.get("signature"):
                continue
        if expected_signature and len(b.get("tokens") or []) != len(expected_signature):
            continue
        if original_tokens is not None and list(b["tokens"]) == list(original_tokens):
            b = dict(b)
            b["is_original"] = True
        else:
            b = dict(b)
            b["is_original"] = False
        out.append(b)
    return out


@torch.no_grad()
def score_bundles_joint(
    model,
    *,
    input_ids_4ch: torch.Tensor,
    padding_mask: torch.Tensor,
    mask_result: InferenceMaskResult,
    bundles: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> List[BundleCandidate]:
    """Score each observed bundle with Σ log P(token at masked positions).

    input_ids_4ch: [1, 4, L] Stage-A style channels (ids in channel 0).
    """
    x = input_ids_4ch.to(device)
    pad = padding_mask.to(device)
    # apply mask on channel 0
    x = x.clone()
    for i, pos in enumerate(mask_result.masked_indices):
        # mask id already in masked_ids; sync channel 0
        pass
    # rebuild ids from mask_result
    L = x.size(-1)
    ids = torch.as_tensor(mask_result.masked_ids, device=device, dtype=torch.long)
    if ids.numel() < L:
        pad_ids = torch.zeros(L, device=device, dtype=torch.long)
        pad_ids[: ids.numel()] = ids
        ids = pad_ids
    x[0, 0, : ids.numel()] = ids[: x.size(-1)]

    hidden = model.transformer.forward_finetuning(x=x, padding_mask=pad.long())
    pos = torch.as_tensor(mask_result.target_pos, device=device, dtype=torch.long).view(1, -1)
    # pad target_pos to decoder expectation if needed
    logits = model.mlm_decoder(hidden, {"target_pos": pos})  # [1, n_pos, V]
    logp = F.log_softmax(logits[0], dim=-1)  # [n_pos, V]

    scored: List[BundleCandidate] = []
    n_pos = int(pos.numel())
    for b in bundles:
        tids = [int(t) for t in b["token_ids"]]
        if len(tids) != n_pos:
            # align by taking value tokens only matching mask length
            if len(tids) < n_pos:
                continue
            tids = tids[:n_pos]
        s = 0.0
        ok = True
        for j, tid in enumerate(tids):
            if tid < 0 or tid >= logp.size(-1):
                ok = False
                break
            s += float(logp[j, tid].item())
        if not ok:
            continue
        scored.append(
            BundleCandidate(
                source="noop" if b.get("is_original") else "constrained_mlm",
                tokens=list(b["tokens"])[:n_pos],
                token_ids=tids,
                score=s,
                is_original=bool(b.get("is_original")),
                feature=str(b.get("feature") or ""),
                measurement_group_id=str(mask_result.measurement_group_id),
            )
        )
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def select_mlm_candidates(
    scored: Sequence[BundleCandidate],
    *,
    top_k: int = 3,
    exclude_original_from_edits: bool = True,
) -> List[BundleCandidate]:
    edits = []
    noops = []
    for c in scored:
        if c.is_original or c.source.startswith("noop"):
            noops.append(
                BundleCandidate(
                    source="noop",
                    tokens=c.tokens,
                    token_ids=c.token_ids,
                    score=c.score,
                    is_original=True,
                    feature=c.feature,
                    measurement_group_id=c.measurement_group_id,
                )
            )
        else:
            edits.append(c)
    out: List[BundleCandidate] = []
    if noops:
        best_noop = max(noops, key=lambda c: c.score)
        out.append(best_noop)
    if exclude_original_from_edits:
        out.extend(edits[:top_k])
    else:
        out.extend(scored[: top_k + (1 if noops else 0)])
    return out


def mlm_reconstruction_metrics(
    scored: Sequence[BundleCandidate],
    *,
    original_tokens: Sequence[str],
    random_baseline_recall_at_3: Optional[float] = None,
) -> Dict[str, Any]:
    orig = list(original_tokens)
    ranks = [i for i, c in enumerate(scored) if list(c.tokens) == orig]
    rank = ranks[0] if ranks else None
    recall1 = 1.0 if rank == 0 else 0.0
    recall3 = 1.0 if rank is not None and rank < 3 else 0.0
    mrr = (1.0 / (rank + 1)) if rank is not None else 0.0
    n = max(len(scored), 1)
    rand3 = random_baseline_recall_at_3
    if rand3 is None:
        rand3 = min(3, n) / float(n)
    return {
        "eligible_group_count": 1,
        "recall@1": recall1,
        "recall@3": recall3,
        "MRR": mrr,
        "random_baseline_recall@3": float(rand3),
        "lift_over_random": float(recall3 - rand3),
        "original_rank": rank,
        "n_scored": len(scored),
    }
