"""Stage A encode helpers with IMAGE_SLOT ← DINO residual injection."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.online2.v2.finetune_v02.image_adapter import SharedImageAdapter, inject_slot_residuals
from src.transformer.models import TransformerEncoder


def find_image_slot_positions(tokens: Sequence[str]) -> List[int]:
    return [i for i, t in enumerate(tokens) if t == "[IMAGE_SLOT]"]


@torch.no_grad()
def encode_batch_pool_with_dino_slots(
    model: TransformerEncoder,
    xs: List[torch.Tensor],
    masks: List[torch.Tensor],
    spans: List[Tuple[int, int]],
    *,
    flat_token_lists: List[List[str]],
    slot_dino: List[Optional[np.ndarray]],
    adapter: Optional[nn.Module],
    device: torch.device,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Embed tokens, add gated DINO residual at [IMAGE_SLOT], run encoder, pool.

    slot_dino[b] is the DINO vector for the *target* IMAGE event when the window
    target is IMAGE; used to inject at every IMAGE_SLOT that belongs to that
    target span. For non-IMAGE targets, injection still applies to IMAGE_SLOT
    tokens in context if we pass a context map — here we inject only when
    slot_dino[b] is not None (typically target IMAGE events).
    """
    x = torch.stack(xs, dim=0).to(device)
    mask = torch.stack(masks, dim=0).to(device).long()

    # token embeddings
    emb, _ = model.transformer.get_sequence_embedding(x)  # (B, L, H) expected
    if isinstance(emb, tuple):
        emb = emb[0]

    if adapter is not None:
        for b, dino in enumerate(slot_dino):
            if dino is None:
                continue
            slots = find_image_slot_positions(flat_token_lists[b])
            if not slots:
                continue
            # Restrict to target span slots when possible
            s, e = spans[b]
            slots = [i for i in slots if s <= i < e] or slots
            vec = torch.as_tensor(dino, device=device, dtype=emb.dtype).view(1, -1)
            residual = adapter(vec).view(-1)  # [H]
            for i in slots:
                if i < emb.size(1):
                    emb[b, i] = inject_slot_residuals(emb[b, i], residual)

    hidden = model.transformer.forward_finetuning_with_embeddings(emb, mask)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for b, (s, e) in enumerate(spans):
        span = hidden[b, s:e, :]
        if span.numel() == 0:
            h = hidden.shape[-1]
            z = np.zeros(h, dtype=np.float32)
            out.append((z, z))
            continue
        mean_v = span.mean(dim=0).float().cpu().numpy().astype(np.float32)
        max_v = span.max(dim=0).values.float().cpu().numpy().astype(np.float32)
        out.append((mean_v, max_v))
    return out


def make_stage_a_image_adapter(hidden_dim: int, dino_dim: int = 1024) -> SharedImageAdapter:
    from src.online2.v2.finetune_v02.config import ImageAdapterConfig

    return SharedImageAdapter(
        ImageAdapterConfig(
            dino_dim=dino_dim,
            model_dim=hidden_dim,
            gate_init=0.01,
            normalize_l2=True,
        )
    )
