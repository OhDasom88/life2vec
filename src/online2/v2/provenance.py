"""Provenance helpers for transductive V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class ProvenanceMeta:
    training_mode: str = "transductive_public_pretraining"
    contains_problem_observations: bool = True
    contains_problem_images: bool = True
    contains_problem_hidden_targets: bool = False
    tokenization_version: str = "v2"
    vocab_version: str = "v2"
    feature_schema_version: str = "v2.0.0"
    materialization_version: str = "v2"
    config_hash: str = ""
    git_commit_hash: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def training_mode_meta(**kwargs: Any) -> dict[str, Any]:
    return ProvenanceMeta(**kwargs).as_dict()
