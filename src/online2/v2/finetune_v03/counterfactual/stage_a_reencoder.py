"""Stage A affected-window re-encode + cache patch for CF M2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from .stage_a_loader import load_stage_a_module


@dataclass
class ParityReport:
    passed: bool
    cosine_mean: float
    mean_abs_error: float
    max_abs_error: float
    n_events: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReencodeResult:
    stage_a_reencode_mode: str
    edited_event_ids: List[str]
    affected_target_event_ids: List[str]
    patched_event_indices: List[int]
    cache_parity_passed: Optional[bool]
    batch: Dict[str, torch.Tensor]
    parity: Optional[ParityReport] = None


def build_reverse_dependency_index(
    events: list,
    stage_a_mod,
    *,
    max_length: int = 1024,
) -> Dict[int, Set[int]]:
    """Map context_event_idx -> set of target indices whose window includes it."""
    dep: Dict[int, Set[int]] = {i: set() for i in range(len(events))}
    for t in range(len(events)):
        window = stage_a_mod.construct_target_window(events, t, max_length=max_length)
        for ctx in window.event_indices:
            dep[int(ctx)].add(int(t))
    return dep


def affected_targets_for_edits(
    dep: Mapping[int, Set[int]],
    edited_indices: Sequence[int],
) -> List[int]:
    out: Set[int] = set()
    for i in edited_indices:
        out |= set(dep.get(int(i), set()))
        out.add(int(i))
    return sorted(out)


class StageAReencoder:
    def __init__(
        self,
        *,
        encoder,
        stage_a_mod,
        vocab,
        abspos_reference: pd.Timestamp,
        case_t0: pd.Timestamp,
        max_length: int = 1024,
        device: torch.device,
        parity_cosine_min: float = 0.9999,
        parity_mean_max_abs_error: float = 1e-4,
        parity_max_max_abs_error: float = 5e-4,
    ) -> None:
        self.encoder = encoder
        self.stage_a = stage_a_mod
        self.vocab = vocab
        self.abspos_reference = abspos_reference
        self.case_t0 = case_t0
        self.max_length = int(max_length)
        self.device = device
        self.parity_cosine_min = float(parity_cosine_min)
        self.parity_mean_max_abs_error = float(parity_mean_max_abs_error)
        self.parity_max_max_abs_error = float(parity_max_max_abs_error)

    def _window_batch(self, events: list, target_indices: Sequence[int]):
        xs, masks, spans = [], [], []
        for t in target_indices:
            window = self.stage_a.construct_target_window(events, int(t), max_length=self.max_length)
            x, pad_mask, _L = self.stage_a.window_to_tensors(
                events,
                window,
                vocab=self.vocab,
                abspos_reference=self.abspos_reference,
                case_t0=self.case_t0,
                max_length=self.max_length,
            )
            xs.append(x)
            masks.append(pad_mask)
            spans.append((int(window.target_token_start), int(window.target_token_end)))
        return xs, masks, spans

    @torch.no_grad()
    def validate_cache_parity(
        self,
        events: list,
        batch: Mapping[str, torch.Tensor],
        *,
        event_indices: Optional[Sequence[int]] = None,
    ) -> ParityReport:
        mask = batch["padding_mask"][0].detach().cpu().numpy().astype(bool)
        indices = (
            list(event_indices)
            if event_indices is not None
            else [i for i in range(len(events)) if i < len(mask) and mask[i]]
        )
        cosines = []
        abs_errs = []
        for i in indices:
            if i >= len(events) or i >= len(mask) or not mask[i]:
                continue
            mean_v, max_v = self._pool_one(events, i)
            cache_mean = batch["event_mean"][0, i].detach().float().cpu()
            cache_max = batch["event_max"][0, i].detach().float().cpu()
            m = mean_v.detach().float().cpu()
            cos = float(
                torch.nn.functional.cosine_similarity(m.unsqueeze(0), cache_mean.unsqueeze(0)).item()
            )
            err = float((m - cache_mean).abs().mean().item())
            err_max = float((max_v.detach().float().cpu() - cache_max).abs().max().item())
            cosines.append(cos)
            abs_errs.append(err)
            abs_errs.append(err_max)
        cosine_mean = float(np.mean(cosines)) if cosines else 0.0
        mean_abs = float(np.mean(abs_errs)) if abs_errs else 0.0
        max_abs = float(np.max(abs_errs)) if abs_errs else 0.0
        passed = (
            cosine_mean >= self.parity_cosine_min
            and mean_abs <= self.parity_mean_max_abs_error
            and max_abs <= self.parity_max_max_abs_error
        )
        return ParityReport(
            passed=passed,
            cosine_mean=cosine_mean,
            mean_abs_error=mean_abs,
            max_abs_error=max_abs,
            n_events=len(cosines),
        )

    @torch.no_grad()
    def _pool_one(self, events: list, target_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, masks, spans = self._window_batch(events, [target_idx])
        pooled = self.stage_a.encode_batch_pool(
            self.encoder, xs, masks, spans, device=self.device
        )
        mean_np, max_np = pooled[0]
        return (
            torch.as_tensor(mean_np, device=self.device),
            torch.as_tensor(max_np, device=self.device),
        )

    @torch.no_grad()
    def apply_edits_and_reencode(
        self,
        events: list,
        edits: Sequence[Mapping[str, Any]],
        batch: Mapping[str, torch.Tensor],
        *,
        event_id_to_index: Mapping[str, int],
    ) -> ReencodeResult:
        events_edit = deepcopy(events)
        edited_ids = []
        edited_idx = []
        for ed in edits:
            eid = str(ed["event_id"])
            if eid not in event_id_to_index:
                continue
            i = int(event_id_to_index[eid])
            if i >= len(events_edit):
                continue
            new_tokens = list(ed.get("to_tokens") or ed.get("sentence_tokens") or [])
            if not new_tokens:
                continue
            # replace full sentence or splice by from/to if provided
            if ed.get("replace_sentence", True):
                events_edit[i].sentence_tokens = list(new_tokens)
                events_edit[i].token_count = len(new_tokens)
            edited_ids.append(eid)
            edited_idx.append(i)

        dep = build_reverse_dependency_index(events_edit, self.stage_a, max_length=self.max_length)
        affected = affected_targets_for_edits(dep, edited_idx)
        mask = batch["padding_mask"][0].detach().cpu().numpy().astype(bool)
        affected = [i for i in affected if i < len(mask) and mask[i]]

        out_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        patched = []
        for i in affected:
            mean_v, max_v = self._pool_one(events_edit, i)
            out_batch["event_mean"][0, i] = mean_v.to(out_batch["event_mean"].dtype)
            out_batch["event_max"][0, i] = max_v.to(out_batch["event_max"].dtype)
            patched.append(int(i))

        # untouched equality check vs original
        for i in range(int(mask.sum())):
            if i in patched:
                continue
            if not torch.equal(out_batch["event_mean"][0, i], batch["event_mean"][0, i]):
                raise AssertionError(f"untouched event_mean changed at {i}")
            if not torch.equal(out_batch["event_max"][0, i], batch["event_max"][0, i]):
                raise AssertionError(f"untouched event_max changed at {i}")

        return ReencodeResult(
            stage_a_reencode_mode="full_affected_window_reencode",
            edited_event_ids=edited_ids,
            affected_target_event_ids=[str(events_edit[i].event_id) for i in affected],
            patched_event_indices=patched,
            cache_parity_passed=None,
            batch=out_batch,
        )

    @torch.no_grad()
    def cold_rebuild_full_sequence(
        self,
        events: list,
        *,
        batch_size: int = 8,
        special_token_budget: int = 6,
        valid_event_cap: int = 4096,
    ) -> Dict[str, Any]:
        """FULL_SEQUENCE_COLD_REBUILD: encode every valid target window from scratch.

        Does not read or reuse any prior event embedding cache.
        Each valid event maps to exactly one target-centered window; event embedding
        is mean/max over that window's target span only (no cross-window reaggregation).
        """
        max_target = int(self.max_length) - int(special_token_budget)
        pre_cap = list(events)
        pre_cap_count = len(pre_cap)
        # EARLIEST_EVENT_ORDER: keep first valid_event_cap events in given order.
        capped = pre_cap[: int(valid_event_cap)]
        valid_indices: List[int] = []
        valid_event_ids: List[str] = []
        window_ids: List[str] = []
        target_membership: Dict[str, List[str]] = {}
        context_membership: Dict[str, List[str]] = {}

        for i, ev in enumerate(capped):
            tokens = list(getattr(ev, "sentence_tokens", None) or [])
            tc = int(getattr(ev, "token_count", len(tokens)) or len(tokens))
            eid = str(getattr(ev, "event_id", f"idx_{i}"))
            if tc < 1 or tc > max_target:
                continue
            if not tokens:
                continue
            window = self.stage_a.construct_target_window(
                capped, i, max_length=self.max_length
            )
            wid = f"window_target_{eid}"
            valid_indices.append(i)
            valid_event_ids.append(eid)
            window_ids.append(wid)
            target_membership[eid] = [wid]
            for ctx_i in window.event_indices:
                ctx_id = str(getattr(capped[int(ctx_i)], "event_id", f"idx_{ctx_i}"))
                context_membership.setdefault(ctx_id, []).append(wid)

        if not valid_indices:
            raise AssertionError("cold_rebuild_full_sequence: no valid target events")

        # Batched encode_batch_pool — never reuse cache tensors.
        means: List[torch.Tensor] = []
        maxes: List[torch.Tensor] = []
        bs = max(1, int(batch_size))
        for start in range(0, len(valid_indices), bs):
            chunk = valid_indices[start : start + bs]
            xs, masks, spans = self._window_batch(capped, chunk)
            pooled = self.stage_a.encode_batch_pool(
                self.encoder, xs, masks, spans, device=self.device
            )
            assert len(pooled) == len(chunk), "encode_batch_pool sidecar length mismatch"
            for j, (mean_np, max_np) in enumerate(pooled):
                mean_t = torch.as_tensor(mean_np, dtype=torch.float32, device=self.device)
                max_t = torch.as_tensor(max_np, dtype=torch.float32, device=self.device)
                assert mean_t.dtype == torch.float32
                means.append(mean_t)
                maxes.append(max_t)
                # Order must match chunk event ids
                assert str(getattr(capped[chunk[j]], "event_id")) == valid_event_ids[start + j]

        event_mean = torch.stack(means, dim=0)
        event_max = torch.stack(maxes, dim=0)
        return {
            "reencode_mode": "FULL_SEQUENCE_COLD_REBUILD",
            "affected_window_only_mode": False,
            "untouched_embedding_cache_reuse_count": 0,
            "stale_stage_a_cache_read_count": 0,
            "all_valid_target_windows_reencoded": True,
            "all_valid_event_embeddings_reconstructed": True,
            "incremental_reencode_used": False,
            "parity_status": "NOT_APPLICABLE",
            "valid_event_ids": valid_event_ids,
            "window_ids": window_ids,
            "target_membership": target_membership,
            "context_membership": context_membership,
            "event_mean": event_mean,
            "event_max": event_max,
            "pre_cap_valid_event_count": pre_cap_count,
            "post_cap_valid_event_count": len(capped),
            "valid_event_cap_applied": pre_cap_count > int(valid_event_cap),
            "excluded_tail_event_count": max(0, pre_cap_count - len(capped)),
            "max_sequence_length": int(self.max_length),
            "special_token_budget": int(special_token_budget),
            "max_target_token_count": int(max_target),
            "special_token_budget_source": "FROZEN_TOKENIZER_CONTRACT",
            "stage_a_batch_size": bs,
            "valid_event_cap": int(valid_event_cap),
            "valid_event_cap_policy": "EARLIEST_EVENT_ORDER",
        }


def apply_token_string_edit(events: list, event_index: int, new_sentence_tokens: Sequence[str]) -> list:
    events = deepcopy(events)
    events[event_index].sentence_tokens = list(new_sentence_tokens)
    events[event_index].token_count = len(new_sentence_tokens)
    return events
