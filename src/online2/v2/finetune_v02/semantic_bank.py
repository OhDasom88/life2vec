"""Build / load High·Medium·Low 상태진단·원인분석 embedding banks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


QUALITY_MAP = {
    "상": "high",
    "중": "medium",
    "하": "low",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "H": "high",
    "M": "medium",
    "L": "low",
}


_STATE_MARKERS = ("상태진단", "종합 진단", "종합 판단", "[상태진단]")
_CAUSE_MARKERS = ("원인분석", "원인 분석", "[원인분석]", "추정 원인")


def split_state_cause(text: str) -> Tuple[str, str]:
    """Heuristic split of interpretation text into state vs cause sections."""
    raw = text.strip()
    if not raw:
        return "", ""

    # Prefer explicit headings
    lower = raw
    cause_idx = -1
    for m in _CAUSE_MARKERS:
        i = lower.find(m)
        if i >= 0:
            cause_idx = i if cause_idx < 0 else min(cause_idx, i)
    if cause_idx >= 0:
        state = raw[:cause_idx].strip()
        cause = raw[cause_idx:].strip()
        # strip heading lines
        cause = re.sub(r"^\[?원인\s*분석\]?\s*:?\s*", "", cause, count=1)
        return state, cause

    # Fallback: first paragraph = state, rest = cause
    parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "\n\n".join(parts[1:])


def mask_diagnosis_in_cause(cause_text: str, diagnosis_names: Iterable[str]) -> str:
    """Replace diagnosis surface forms so cause head does not latch onto label words."""
    out = cause_text
    for name in sorted(diagnosis_names, key=len, reverse=True):
        if not name:
            continue
        out = out.replace(name, "[DIAGNOSIS_MASK]")
        # also collapsed variants
        collapsed = name.replace(" ", "").replace("_", "")
        if collapsed and collapsed != name:
            out = out.replace(collapsed, "[DIAGNOSIS_MASK]")
    return out


def normalize_quality(raw: str) -> Optional[str]:
    return QUALITY_MAP.get(str(raw).strip())


def empty_bank_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["case_id", "quality", "axis", "text", "embedding"])


def rows_from_texts(
    case_id: str,
    quality: str,
    state_text: str,
    cause_text: str,
    *,
    embed_fn,
) -> List[Dict]:
    q = normalize_quality(quality)
    if q is None:
        raise ValueError(f"unknown quality: {quality}")
    rows = []
    if state_text.strip():
        rows.append(
            {
                "case_id": case_id,
                "quality": q,
                "axis": "state",
                "text": state_text.strip(),
                "embedding": embed_fn(state_text.strip()),
            }
        )
    if cause_text.strip():
        rows.append(
            {
                "case_id": case_id,
                "quality": q,
                "axis": "cause",
                "text": cause_text.strip(),
                "embedding": embed_fn(cause_text.strip()),
            }
        )
    return rows


def hash_embed(text: str, dim: int = 192) -> List[float]:
    """Deterministic placeholder embedding (no external model). For smoke only."""
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    v = rng.normal(size=dim).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.tolist()
