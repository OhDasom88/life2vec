"""Transductive V2 binning registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .feature_schema import FeatureSchema
from ..canonical import canonical_json, stable_id


def _unique_edges(values: np.ndarray, bins: int, quantile: bool) -> list[float]:
    if values.size == 0:
        return []
    if np.all(values == values[0]):
        return [float(values[0])]
    if quantile:
        edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    else:
        lo, hi = np.percentile(values, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(values.min()), float(values.max())
        edges = np.linspace(float(lo), float(hi), bins + 1)
    return [float(v) for v in np.unique(edges)]


@dataclass(frozen=True)
class BinRuleV2:
    feature: str
    kind: str  # ABS | GLOBAL_REL | FARM_REL
    edges: tuple[float, ...]
    scope_id: Optional[str] = None
    closed_side: str = "right"
    fit_scope: str = "transductive_public_pretrain_pool"
    causal: bool = False
    uses_problem_inputs: bool = True
    absolute_policy: str = "winsorized_range"
    relative_scope: str = "full_public_farm_distribution"

    @property
    def constant(self) -> bool:
        return len(self.edges) == 1

    def token_suffix(self, value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return f"{self.kind}_NULL"
        if self.constant:
            return f"{self.kind}_CONSTANT"
        import bisect

        position = bisect.bisect_left(self.edges[1:-1], float(value))
        return f"{self.kind}_B{position:02d}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BinningRegistryV2:
    def __init__(self, version: str = "v2_transductive", bins: int = 10) -> None:
        self.version = version
        self.bins = bins
        self.rules: dict[tuple[str, str, Optional[str]], BinRuleV2] = {}
        self.meta: dict[str, Any] = {}

    def fit(
        self,
        schema: FeatureSchema,
        rows: Iterable[tuple[str, Optional[str], Optional[float], str]],
        source_file_hashes: Mapping[str, str] | None = None,
        example_count: int = 0,
        problem_count: int = 0,
        code_commit_hash: str = "",
    ) -> None:
        grouped: dict[str, list[float]] = {}
        farms: dict[tuple[str, str], list[float]] = {}
        set_counts = {"example_set": 0, "problem_set": 0, "unknown": 0}
        hasher = hashlib.sha256()
        n_fit_values = 0
        for feature, farm, value, source_set in rows:
            set_counts[source_set if source_set in set_counts else "unknown"] += 1
            if value is None or not np.isfinite(float(value)):
                continue
            number = float(value)
            spec = schema.features.get(feature)
            if spec is None:
                continue
            if not (
                spec.absolute_encoding
                or spec.global_relative_encoding
                or spec.farm_relative_encoding
            ):
                continue
            grouped.setdefault(feature, []).append(number)
            if farm:
                farms.setdefault((feature, farm), []).append(number)
            hasher.update(f"{feature}|{farm}|{number:.12g}|{source_set}\n".encode())
            n_fit_values += 1

        for feature, values in sorted(grouped.items()):
            array = np.asarray(values, dtype=np.float64)
            spec = schema.get(feature)
            if spec.absolute_encoding:
                self.rules[(feature, "ABS", None)] = BinRuleV2(
                    feature,
                    "ABS",
                    tuple(_unique_edges(array, self.bins, False)),
                    absolute_policy="winsorized_p01_p99",
                )
            if spec.global_relative_encoding:
                self.rules[(feature, "GLOBAL_REL", None)] = BinRuleV2(
                    feature,
                    "GLOBAL_REL",
                    tuple(_unique_edges(array, self.bins, True)),
                )
        for (feature, farm), values in sorted(farms.items()):
            spec = schema.features.get(feature)
            if not spec or not spec.farm_relative_encoding:
                continue
            array = np.asarray(values, dtype=np.float64)
            self.rules[(feature, "FARM_REL", farm)] = BinRuleV2(
                feature,
                "FARM_REL",
                tuple(_unique_edges(array, self.bins, True)),
                scope_id=farm,
                causal=False,
                uses_problem_inputs=True,
                relative_scope="full_public_farm_distribution",
            )

        hasher.update(
            canonical_json(
                {
                    "bins": self.bins,
                    "source_file_hashes": dict(sorted((source_file_hashes or {}).items())),
                    "n_fit_values": n_fit_values,
                }
            ).encode("utf-8")
        )
        self.meta = {
            "registry_version": self.version,
            "fit_scope": "transductive_public_pretrain_pool",
            "example_observation_count": example_count,
            "problem_observation_count": problem_count,
            "source_set_counts": set_counts,
            "source_file_hashes": dict(sorted((source_file_hashes or {}).items())),
            "bin_count": self.bins,
            "closed_side": "right",
            "outlier_policy": "winsorize_abs_p01_p99",
            "constant_policy": "CONSTANT",
            "missing_policy": "NULL",
            "farm_relative": {
                "enabled": True,
                "fit_scope": "full_public_farm_distribution",
                "causal": False,
            },
            "training_mode": "transductive_public_pretraining",
            "contains_problem_observations": True,
            "contains_problem_images": True,
            "contains_problem_hidden_targets": False,
            "fit_population_hash": hasher.hexdigest(),
            "n_fit_values": n_fit_values,
            "code_commit_hash": code_commit_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "schema_config_hash": schema.config_hash(),
        }
        self.meta["registry_hash"] = hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def encode_suffix(self, feature: str, value: Optional[float], kind: str, farm_id: Optional[str] = None) -> str:
        scope = farm_id if kind == "FARM_REL" else None
        rule = self.rules[(feature, kind, scope)]
        return rule.token_suffix(value)

    def payload(self) -> dict[str, Any]:
        rules = [
            self.rules[key].as_dict()
            for key in sorted(self.rules, key=lambda item: (item[0], item[1], item[2] or ""))
        ]
        return {"meta": self.meta, "rules": rules}

    def save(self, path) -> None:
        path.write_text(json.dumps(self.payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BinningRegistryV2":
        payload = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(version=payload["meta"].get("registry_version", "v2_transductive"))
        obj.meta = payload["meta"]
        for rule in payload["rules"]:
            edges = tuple(rule["edges"])
            key = (rule["feature"], rule["kind"], rule.get("scope_id"))
            obj.rules[key] = BinRuleV2(
                feature=rule["feature"],
                kind=rule["kind"],
                edges=edges,
                scope_id=rule.get("scope_id"),
                closed_side=rule.get("closed_side", "right"),
                fit_scope=rule.get("fit_scope", "transductive_public_pretrain_pool"),
                causal=bool(rule.get("causal", False)),
                uses_problem_inputs=bool(rule.get("uses_problem_inputs", True)),
                absolute_policy=rule.get("absolute_policy", "winsorized_range"),
                relative_scope=rule.get("relative_scope", "full_public_farm_distribution"),
            )
        obj.bins = int(payload["meta"].get("bin_count", 10))
        return obj


def fit_binning_v2(*args, **kwargs) -> BinningRegistryV2:
    registry = BinningRegistryV2()
    registry.fit(*args, **kwargs)
    return registry
