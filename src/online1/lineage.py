from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .paths import load_yaml, CONFIG_DIR


@dataclass
class FeatureLineage:
    name: str
    source_columns: list[str]
    transform_chain: list[str]
    target_dependency: bool = False
    future_dependency: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_columns": self.source_columns,
            "transform_chain": self.transform_chain,
            "target_dependency": self.target_dependency,
            "future_dependency": self.future_dependency,
            "tags": self.tags,
        }


def load_deny_config(path=None) -> dict[str, Any]:
    return load_yaml(path or (CONFIG_DIR / "lineage_deny.yaml"))


def build_env_lineage(columns: list[str], deny_cfg: dict[str, Any]) -> list[FeatureLineage]:
    deny_cols = set(deny_cfg.get("deny_source_columns", []))
    patterns = [re.compile(p) for p in deny_cfg.get("deny_name_patterns", [])]
    allowed = set(deny_cfg.get("allowed_env_columns", []))
    out: list[FeatureLineage] = []
    for c in columns:
        is_target = c in deny_cols or any(p.search(c) for p in patterns)
        if c == "time":
            continue
        if allowed and c not in allowed and not is_target:
            # unknown column — mark for review but not automatically target
            tags = ["unknown_column"]
        else:
            tags = []
        out.append(
            FeatureLineage(
                name=c,
                source_columns=[c],
                transform_chain=["raw"],
                target_dependency=bool(is_target),
                future_dependency=False,
                tags=tags + (["target_dependency"] if is_target else []),
            )
        )
    return out


def assert_encoder_safe(lineages: list[FeatureLineage]) -> dict[str, Any]:
    bad = [l for l in lineages if l.target_dependency or l.future_dependency]
    return {
        "target_lineage_violations": sum(1 for l in bad if l.target_dependency),
        "future_dependency_violations": sum(1 for l in bad if l.future_dependency),
        "violating_features": [l.name for l in bad],
        "pass": len(bad) == 0,
    }


def encoder_columns(lineages: list[FeatureLineage]) -> list[str]:
    return [
        l.name
        for l in lineages
        if (not l.target_dependency) and (not l.future_dependency) and l.name != "time"
    ]


def scan_dataframe_for_denied(df: pd.DataFrame, deny_cfg: dict[str, Any]) -> list[str]:
    deny_cols = set(deny_cfg.get("deny_source_columns", []))
    patterns = [re.compile(p) for p in deny_cfg.get("deny_name_patterns", [])]
    hits = []
    for c in df.columns:
        if c in deny_cols or any(p.search(str(c)) for p in patterns):
            hits.append(c)
    return hits
