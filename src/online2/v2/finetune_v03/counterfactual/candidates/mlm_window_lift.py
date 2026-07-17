"""Lift event-local MG masks to Stage-A window absolute offsets (token-ID channel only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..attribution.aggregation import infer_token_role
from .inference_masker import InferenceMaskResult, mask_measurement_group


@dataclass
class AnnotatedTokens:
    tokens: List[str]
    token_ids: List[int]
    group_ids: List[str]
    roles: List[str]
    features: List[str]


def annotate_sentence_tokens(
    tokens: Sequence[str],
    *,
    vocab_token2index: Optional[dict] = None,
    unk_id: int = 0,
) -> AnnotatedTokens:
    """Assign measurement_group_id / role / feature per token (FEATURE-span heuristic)."""
    toks = [str(t) for t in tokens]
    group_ids: List[str] = []
    roles: List[str] = []
    features: List[str] = []
    mg_id = "mg:none"
    feat = "unknown"
    for t in toks:
        role = infer_token_role(t)
        if t.startswith("FEATURE|"):
            feat = t.split("|", 1)[-1]
            mg_id = f"mg:{feat}"
        group_ids.append(mg_id)
        roles.append(role)
        features.append(feat)
    if vocab_token2index is not None:
        ids = [int(vocab_token2index.get(t, unk_id)) for t in toks]
    else:
        ids = list(range(len(toks)))
    return AnnotatedTokens(
        tokens=toks,
        token_ids=ids,
        group_ids=group_ids,
        roles=roles,
        features=features,
    )


def extract_value_bundle(
    ann: AnnotatedTokens,
    *,
    measurement_group_id: str,
) -> Tuple[List[str], List[int], List[str]]:
    """Value-role tokens inside MG ( foredit / bank rows)."""
    toks, ids, roles = [], [], []
    for i, g in enumerate(ann.group_ids):
        if str(g) != str(measurement_group_id):
            continue
        if ann.roles[i] in {
            "abs_value",
            "global_value",
            "farm_relative_value",
            "value",
        } or (
            ann.tokens[i].startswith("VALUE_")
            or ann.tokens[i].startswith("OBSERVED_VALUE|")
        ):
            toks.append(ann.tokens[i])
            ids.append(ann.token_ids[i])
            roles.append(ann.roles[i])
    return toks, ids, roles


@dataclass
class WindowLiftResult:
    mask_result: InferenceMaskResult
    input_ids_4ch: torch.Tensor  # [1,4,L] with channel-0 masked
    padding_mask: torch.Tensor  # [1,L]
    local_masked_indices: List[int]
    window_masked_indices: List[int]
    modified_channels: List[str]
    unchanged_auxiliary_channels: bool
    unchanged_non_target_positions: bool
    unchanged_attention_mask: bool


def lift_local_mask_to_window(
    *,
    local_token_ids: Sequence[int],
    local_group_ids: Sequence[str],
    local_roles: Sequence[str],
    local_tokens: Sequence[str],
    target_group_id: str,
    mask_id: int,
    window_input_ids_4ch: torch.Tensor,
    window_padding_mask: torch.Tensor,
    target_token_start: int,
) -> WindowLiftResult:
    """Mask MG value tokens locally, then apply only to token-id channel at absolute offsets."""
    local = mask_measurement_group(
        local_token_ids,
        local_group_ids,
        local_roles,
        target_group_id=target_group_id,
        mask_id=mask_id,
        tokens=local_tokens,
    )
    start = int(target_token_start)
    window_pos = [start + int(i) for i in local.masked_indices]

    x = window_input_ids_4ch.clone()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    pad = window_padding_mask.clone()
    if pad.dim() == 1:
        pad = pad.unsqueeze(0)
    pad_before = pad.clone()
    aux_before = x[:, 1:, :].clone()
    L = int(x.size(-1))

    # Build full-window masked id channel
    ids0 = x[0, 0, :].clone()
    for lp, wp in zip(local.masked_indices, window_pos):
        if wp < 0 or wp >= L:
            raise ValueError(f"window position {wp} out of range L={L}")
        ids0[wp] = int(mask_id)
        # verify local→window identity of original id
        if int(local.masked_ids[lp]) != int(mask_id):
            raise AssertionError("local mask inconsistent")

    # Non-target positions in channel 0 must match original window
    orig0 = window_input_ids_4ch[0, 0] if window_input_ids_4ch.dim() == 3 else window_input_ids_4ch[0]
    unchanged_non_target = True
    for i in range(min(L, int(orig0.numel()))):
        if i in window_pos:
            continue
        if int(ids0[i].item()) != int(orig0[i].item()):
            unchanged_non_target = False
            break

    x = x.clone()
    x[0, 0, :] = ids0
    aux_after = x[:, 1:, :]
    unchanged_aux = bool(torch.equal(aux_before, aux_after))
    unchanged_attn = bool(torch.equal(pad_before, pad))
    if not unchanged_aux:
        raise AssertionError("auxiliary channels mutated during MLM mask lift")
    if not unchanged_attn:
        raise AssertionError("attention/padding mask mutated during MLM mask lift")

    # Window-length InferenceMaskResult for score_bundles_joint
    full_masked = ids0.detach().cpu().numpy().astype(np.int64)
    lifted = InferenceMaskResult(
        masked_ids=full_masked,
        target_pos=np.asarray(window_pos, dtype=np.int64),
        target_tok=local.target_tok.copy(),
        masked_indices=list(window_pos),
        measurement_group_id=str(target_group_id),
    )
    return WindowLiftResult(
        mask_result=lifted,
        input_ids_4ch=x,
        padding_mask=pad,
        local_masked_indices=list(local.masked_indices),
        window_masked_indices=list(window_pos),
        modified_channels=["token_id"],
        unchanged_auxiliary_channels=unchanged_aux,
        unchanged_non_target_positions=unchanged_non_target,
        unchanged_attention_mask=unchanged_attn,
    )
