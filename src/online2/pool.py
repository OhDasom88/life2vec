"""Public-pool admission and leakage contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PoolDecision:
    allowed: bool
    role: str
    reason: str
    answer_tier: Optional[str] = None
    quality_score: Optional[int] = None


SCORE_QUALITY = {"score50": 50, "score70": 70, "score90": 90}


def classify_public_path(path: Path, set_name: Optional[str] = None) -> PoolDecision:
    """Classify source provenance without inferring hidden problem answers."""
    normalized = path.as_posix().lower()
    if "reference_answers" in normalized:
        tier = next((tier for tier in SCORE_QUALITY if tier in normalized), None)
        if set_name != "example_set":
            return PoolDecision(False, "forbidden", "problem/unknown interpretation")
        if tier == "score90":
            return PoolDecision(True, "primary_interpretation", "public example answer", tier, 90)
        if tier in ("score50", "score70"):
            return PoolDecision(
                True, "contrast_interpretation", "auxiliary public example answer",
                tier, SCORE_QUALITY[tier],
            )
        return PoolDecision(False, "forbidden", "answer tier is not recognized")
    if "/i_images/" in normalized or normalized.endswith((".jpg", ".jpeg", ".png")):
        return PoolDecision(True, "image_observation", "public image")
    if normalized.endswith(".csv"):
        return PoolDecision(True, "observation", "public observation")
    return PoolDecision(False, "unknown", "source is outside the declared public pool")


def assert_no_future_leakage(
    feature_window_end: datetime,
    event_timestamp: datetime,
    uses_future_data: bool,
) -> None:
    """Reject a derived feature that reaches beyond its event anchor."""
    if uses_future_data or feature_window_end > event_timestamp:
        raise ValueError("future data cannot be attached to a past event")
