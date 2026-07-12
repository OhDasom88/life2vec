"""Versioned numeric binning and categorical token policies."""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .canonical import SCHEMA_VERSION, canonical_json, stable_id


def _unique_edges(values: np.ndarray, bins: int, quantile: bool) -> list[float]:
    if values.size == 0:
        return []
    if np.all(values == values[0]):
        return [float(values[0])]
    if quantile:
        edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    else:
        edges = np.linspace(float(values.min()), float(values.max()), bins + 1)
    return [float(value) for value in np.unique(edges)]


@dataclass(frozen=True)
class BinRule:
    feature: str
    kind: str
    edges: tuple[float, ...]
    scope_id: Optional[str] = None
    closed_side: str = "right"
    null_token: str = "NULL"

    @property
    def constant(self) -> bool:
        return len(self.edges) == 1

    def token(self, value: Optional[float]) -> str:
        prefix = self.feature.upper()
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f"{prefix}|{self.kind}_NULL"
        if self.constant:
            return f"{prefix}|{self.kind}_CONSTANT"
        position = bisect.bisect_left(self.edges[1:-1], float(value))
        return f"{prefix}|{self.kind}_B{position:02d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "scope_id": self.scope_id,
            "edges": list(self.edges),
            "closed_side": self.closed_side,
            "duplicate_edge_policy": "drop",
            "constant_policy": "CONSTANT",
            "null_policy": self.null_token,
        }


class TokenRegistry:
    """Fit and apply ABS/GLOBAL/FARM_REL policies."""

    def __init__(self, version: str = "1", bins: int = 10) -> None:
        self.version = version
        self.bins = bins
        self.rules: dict[tuple[str, str, Optional[str]], BinRule] = {}
        self.fit_population_hash = ""

    def fit(
        self,
        rows: Iterable[tuple[str, Optional[str], Optional[float]]],
        source_checksums: Iterable[str] = (),
    ) -> None:
        grouped: dict[str, list[float]] = {}
        farms: dict[tuple[str, str], list[float]] = {}
        canonical_rows = []
        for feature, farm, value in rows:
            if value is None or not np.isfinite(value):
                continue
            number = float(value)
            grouped.setdefault(feature, []).append(number)
            if farm:
                farms.setdefault((feature, farm), []).append(number)
            canonical_rows.append((feature, farm, number))
        for feature, values in sorted(grouped.items()):
            array = np.asarray(values, dtype=np.float64)
            self.rules[(feature, "ABS", None)] = BinRule(
                feature, "ABS", tuple(_unique_edges(array, self.bins, False))
            )
            self.rules[(feature, "GLOBAL", None)] = BinRule(
                feature, "GLOBAL", tuple(_unique_edges(array, self.bins, True))
            )
        for (feature, farm), values in sorted(farms.items()):
            array = np.asarray(values, dtype=np.float64)
            self.rules[(feature, "FARM_REL", farm)] = BinRule(
                feature, "FARM_REL", tuple(_unique_edges(array, self.bins, True)), farm
            )
        fit_key = {
            "rows": sorted(canonical_rows),
            "source_checksums": sorted(source_checksums),
            "bins": self.bins,
        }
        self.fit_population_hash = hashlib.sha256(
            canonical_json(fit_key).encode("utf-8")
        ).hexdigest()

    def encode(
        self, feature: str, value: Optional[float], kind: str = "GLOBAL",
        farm_id: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        scope = farm_id if kind == "FARM_REL" else None
        rule = self.rules[(feature, kind, scope)]
        token = rule.token(value)
        metadata = {
            "token_id": stable_id("token", {"token": token, "version": self.version}),
            "token_string": token,
            "token_category": feature,
            "feature_name": feature,
            "tokenization_kind": kind,
            "registry_version": self.version,
            "rule_hash": hashlib.sha256(
                canonical_json(rule.as_dict()).encode("utf-8")
            ).hexdigest(),
            "is_special": False,
            "is_background": False,
            "is_maskable": True,
        }
        return token, metadata

    def payload(self) -> dict[str, Any]:
        rules = [
            self.rules[key].as_dict()
            for key in sorted(self.rules, key=lambda item: (item[0], item[1], item[2] or ""))
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "registry_version": self.version,
            "bin_count": self.bins,
            "fit_population_hash": self.fit_population_hash,
            "rules": rules,
        }
        payload["registry_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload


def categorical_token(feature: str, raw: str, mapping: Mapping[str, str] | None = None) -> str:
    mapped = mapping.get(raw) if mapping else None
    value = mapped if mapped is not None else f"UNKNOWN_CODE|{raw}"
    return f"{feature.upper()}|{value}"
