"""Grouped measurement masking with family-aware random replacement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .vocab import TOKEN_USAGE, VocabV2, _family


@dataclass
class MaskReport:
    masked_measurement_group_ratio: float
    masked_token_ratio: float
    masked_group_count: int
    masked_token_count: int
    selected_group_ids: List[str]


class GroupedMLMMasker:
    def __init__(
        self,
        vocab: VocabV2,
        mask_ratio: float = 0.30,
        mask_feature_identity: bool = False,
        mask_value_tokens_together: bool = True,
        mask_directly_revealing_derived_tokens: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.vocab = vocab
        self.mask_ratio = mask_ratio
        self.mask_feature_identity = mask_feature_identity
        self.mask_value_tokens_together = mask_value_tokens_together
        self.mask_directly_revealing_derived_tokens = mask_directly_revealing_derived_tokens
        self.rng = np.random.RandomState(seed if seed is not None else None)
        self.mask_id = vocab.get("[MASK]")
        self.pad_id = vocab.get("[PAD]")

    def _replacement_pool(self, family: str) -> np.ndarray:
        usage = TOKEN_USAGE.get(family, {})
        if not usage.get("random_replacement_allowed", False):
            return np.asarray([], dtype=np.int64)
        ids = [i for i in self.vocab.family_to_ids.get(family, []) if i != self.mask_id and i != self.pad_id]
        return np.asarray(ids, dtype=np.int64)

    def mask(
        self,
        token_ids: Sequence[int],
        group_ids: Sequence[str],
        roles: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MaskReport]:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        n = len(token_ids)
        assert n == len(group_ids) == len(roles)

        # Eligible groups: those containing at least one maskable value role
        groups: Dict[str, List[int]] = {}
        for idx, gid in enumerate(group_ids):
            if not gid or gid == "NONE":
                continue
            groups.setdefault(gid, []).append(idx)

        eligible = []
        for gid, idxs in groups.items():
            role_set = {roles[i] for i in idxs}
            if role_set & {
                "value_abs",
                "value_global",
                "value_farm",
                "value_abs_combined",
                "value_global_combined",
                "value_farm_combined",
                "literal",
                "circular",
            }:
                eligible.append(gid)

        n_mask_groups = int(np.floor(len(eligible) * self.mask_ratio)) if eligible else 0
        if n_mask_groups:
            chosen = list(self.rng.choice(eligible, size=n_mask_groups, replace=False))
        else:
            chosen = []

        masked = token_ids.copy()
        target_pos = []
        target_tok = []
        value_roles = {
            "value_abs",
            "value_global",
            "value_farm",
            "value_abs_combined",
            "value_global_combined",
            "value_farm_combined",
            "literal",
            "circular",
            "quality",
        }
        if self.mask_directly_revealing_derived_tokens:
            value_roles.add("derived")

        for gid in chosen:
            idxs = groups[gid]
            for i in idxs:
                role = roles[i]
                if role == "feature_identity" and not self.mask_feature_identity:
                    continue
                if role not in value_roles and role != "feature_identity":
                    # keep separators etc.
                    if role not in {"feature_identity"}:
                        continue
                target_pos.append(i)
                target_tok.append(int(token_ids[i]))
                draw = self.rng.rand()
                if draw < 0.8:
                    masked[i] = self.mask_id
                elif draw < 0.9:
                    pass  # keep
                else:
                    fam = _family(self.vocab.id_to_token[int(token_ids[i])])
                    pool = self._replacement_pool(fam)
                    if pool.size:
                        masked[i] = int(self.rng.choice(pool))
                    else:
                        masked[i] = self.mask_id

        report = MaskReport(
            masked_measurement_group_ratio=(len(chosen) / max(1, len(eligible))),
            masked_token_ratio=(len(target_pos) / max(1, n)),
            masked_group_count=len(chosen),
            masked_token_count=len(target_pos),
            selected_group_ids=chosen,
        )
        return (
            masked,
            np.asarray(target_pos, dtype=np.int64),
            np.asarray(target_tok, dtype=np.int64),
            report,
        )
