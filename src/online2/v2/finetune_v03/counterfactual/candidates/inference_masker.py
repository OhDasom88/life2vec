"""Deterministic inference masker for CF (no RNG, no 80/10/10).

Masks only value-role tokens inside a specified measurement_group_id.
Never call GroupedMLMMasker from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

VALUE_ROLES = {
    "abs_value",
    "value_abs",
    "global_value",
    "value_global",
    "farm_relative_value",
    "value_farm",
    "value",
    "observed_value",
    "continuous_literal",
}

PRESERVE_ROLE_PREFIXES = (
    "FEATURE|",
    "UNIT|",
    "SOURCE|",
    "VIEW|",
    "FARM|",
    "ZONE|",
    "TIME|",
    "DATE|",
    "[",
)


@dataclass
class InferenceMaskResult:
    masked_ids: np.ndarray
    target_pos: np.ndarray
    target_tok: np.ndarray
    masked_indices: List[int]
    measurement_group_id: str


def _is_value_role(role: str) -> bool:
    r = str(role).lower()
    if r in VALUE_ROLES:
        return True
    if "value" in r and "feature" not in r:
        return True
    return False


def _should_preserve_token(token: str, role: str) -> bool:
    t = str(token)
    if any(t.startswith(p) for p in PRESERVE_ROLE_PREFIXES):
        return True
    r = str(role).lower()
    if r in {"feature_identity", "unit", "source", "time_context", "farm_context", "view", "special", "meta"}:
        return True
    return False


def mask_measurement_group(
    token_ids: Sequence[int],
    group_ids: Sequence[str],
    roles: Sequence[str],
    *,
    target_group_id: str,
    mask_id: int,
    tokens: Optional[Sequence[str]] = None,
) -> InferenceMaskResult:
    """Replace value-role tokens in target_group_id with mask_id. Deterministic."""
    ids = np.asarray(token_ids, dtype=np.int64)
    n = len(ids)
    if not (n == len(group_ids) == len(roles)):
        raise ValueError("token_ids, group_ids, roles length mismatch")
    if tokens is not None and len(tokens) != n:
        raise ValueError("tokens length mismatch")

    masked = ids.copy()
    positions: List[int] = []
    originals: List[int] = []
    for i in range(n):
        if str(group_ids[i]) != str(target_group_id):
            continue
        role = str(roles[i])
        tok = str(tokens[i]) if tokens is not None else ""
        if tokens is not None and _should_preserve_token(tok, role):
            continue
        if not _is_value_role(role):
            # if roles unavailable granularity, still preserve FEATURE etc via token string
            if tokens is not None and _should_preserve_token(tok, role):
                continue
            if tokens is None:
                continue
            # without explicit value role, skip non-VALUE tokens
            if not (tok.startswith("VALUE_") or tok.startswith("OBSERVED_VALUE|")):
                continue
        positions.append(i)
        originals.append(int(ids[i]))
        masked[i] = int(mask_id)

    if not positions:
        raise ValueError(f"no value tokens to mask in group={target_group_id}")

    return InferenceMaskResult(
        masked_ids=masked,
        target_pos=np.asarray(positions, dtype=np.int64),
        target_tok=np.asarray(originals, dtype=np.int64),
        masked_indices=positions,
        measurement_group_id=str(target_group_id),
    )


def assert_only_group_changed(
    original_ids: Sequence[int],
    masked_ids: Sequence[int],
    group_ids: Sequence[str],
    target_group_id: str,
) -> None:
    for i, (a, b, g) in enumerate(zip(original_ids, masked_ids, group_ids)):
        if str(g) != str(target_group_id) and int(a) != int(b):
            raise AssertionError(f"token outside target group changed at i={i}")
