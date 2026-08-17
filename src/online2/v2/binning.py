"""Transductive V2 binning registry (adaptive equal-frequency)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml

from .feature_schema import FeatureSchema
from ..canonical import canonical_json


DEFAULT_BINS = 10
DEFAULT_ZERO_MASS_THRESHOLD = 0.15
DEFAULT_MIN_POSITIVE_FOR_ZERO_SPLIT = 20
DEFAULT_MIN_SAMPLES_PER_BIN = 40
DEFAULT_MIN_SAMPLES_PER_BIN_FARM = 8
DEFAULT_MAX_BINS = 100
DEFAULT_MIN_BINS = 2


@dataclass(frozen=True)
class ChannelBinPlan:
    n_bins: int
    strategy: str
    anchors: tuple[float, ...] = ()
    zero_mass_threshold: float = DEFAULT_ZERO_MASS_THRESHOLD
    rationale: str = ""


@dataclass
class BinningPolicy:
    """Declarative per-feature / tier binning policy."""

    meta: dict[str, Any] = field(default_factory=dict)
    tiers: dict[str, dict[str, Any]] = field(default_factory=dict)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "BinningPolicy":
        if path is None or not Path(path).exists():
            return cls()
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            meta=dict(payload.get("meta") or {}),
            tiers={str(k): dict(v) for k, v in (payload.get("tiers") or {}).items()},
            features={str(k): dict(v) for k, v in (payload.get("features") or {}).items()},
        )

    def default_bins(self) -> int:
        return int(self.meta.get("default_bins", DEFAULT_BINS))

    def plan_for(
        self,
        feature: str,
        *,
        channel: str,
        feature_type: str = "continuous",
        n_values: int = 0,
        n_unique: int = 0,
        schema_n_bins: Optional[int] = None,
        default_bins_override: Optional[int] = None,
    ) -> ChannelBinPlan:
        feat = self.features.get(feature, {})
        tier_name = str(feat.get("tier") or _default_tier(feature_type))
        tier = self.tiers.get(tier_name, {})

        strategy = str(
            feat.get("strategy")
            or tier.get("strategy")
            or "equal_frequency"
        )
        anchors = tuple(float(x) for x in (feat.get("anchors") or []))
        zero_thr = float(
            feat.get("zero_mass_threshold")
            or self.meta.get("zero_mass_threshold", DEFAULT_ZERO_MASS_THRESHOLD)
        )

        if channel == "ABS":
            key = "abs_bins"
        elif channel == "GLOBAL_REL":
            key = "global_bins"
        else:
            key = "farm_bins"

        requested = None
        if schema_n_bins is not None:
            requested = int(schema_n_bins)
        elif feat.get(key) is not None:
            requested = int(feat[key])
        elif tier.get(key) is not None:
            requested = int(tier[key])
        else:
            # feature/tier YAML overrides (checked above) always win; only fall through to
            # the registry-level bins=N override (BinningRegistryV2(bins=...)) when neither
            # specifies a count, and only then fall back to the policy file's own default.
            base_default = default_bins_override if default_bins_override is not None else self.default_bins()
            requested = base_default
            if channel == "FARM_REL":
                scale = float(self.meta.get("farm_bins_scale", 0.5))
                abs_bins = int(feat.get("abs_bins") or tier.get("abs_bins") or base_default)
                requested = max(DEFAULT_MIN_BINS, int(round(abs_bins * scale)))

        min_bins = int(self.meta.get("min_bins", DEFAULT_MIN_BINS))
        max_bins = int(self.meta.get("max_bins", DEFAULT_MAX_BINS))
        if channel == "FARM_REL":
            min_per = int(self.meta.get("min_samples_per_bin_farm", DEFAULT_MIN_SAMPLES_PER_BIN_FARM))
        elif channel == "GLOBAL_REL":
            min_per = int(self.meta.get("min_samples_per_bin_global", DEFAULT_MIN_SAMPLES_PER_BIN))
        else:
            min_per = int(self.meta.get("min_samples_per_bin_abs", DEFAULT_MIN_SAMPLES_PER_BIN))

        n_bins = _clip_bins(requested, n_values=n_values, n_unique=n_unique, min_bins=min_bins, max_bins=max_bins, min_per_bin=min_per)
        return ChannelBinPlan(
            n_bins=n_bins,
            strategy=strategy,
            anchors=anchors if channel != "FARM_REL" else (),
            zero_mass_threshold=zero_thr,
            rationale=str(feat.get("rationale_ko") or tier.get("note") or ""),
        )


def _default_tier(feature_type: str) -> str:
    if feature_type == "counter":
        return "counter"
    if feature_type in {"flow", "ordinal_actuator", "boolean"}:
        return "flow_or_actuator"
    if feature_type == "circular":
        return "env_continuous"
    return "env_continuous"


def _clip_bins(
    requested: int,
    *,
    n_values: int,
    n_unique: int,
    min_bins: int,
    max_bins: int,
    min_per_bin: int,
) -> int:
    n = max(min_bins, int(requested))
    n = min(n, max_bins)
    if n_unique > 0:
        n = min(n, max(min_bins, int(n_unique)))
    if n_values > 0 and min_per_bin > 0:
        by_samples = max(min_bins, n_values // min_per_bin)
        n = min(n, by_samples)
    return int(max(min_bins, n))


def _occupancy(values: np.ndarray, edges: Sequence[float]) -> list[int]:
    if len(edges) < 2 or values.size == 0:
        return [int(values.size)] if values.size else []
    # digitize with right-closed interior bins matching bisect_left encode
    counts = [0] * (len(edges) - 1)
    for value in values:
        position = int(np.searchsorted(edges[1:-1], float(value), side="left"))
        counts[position] += 1
    return counts


def _occupancy_stats(counts: Sequence[int]) -> dict[str, Any]:
    arr = np.asarray(counts, dtype=np.float64)
    total = float(arr.sum()) if arr.size else 0.0
    shares = (arr / total).tolist() if total > 0 else []
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 0.0
    cv = float(std / mean) if mean > 0 else 0.0
    return {
        "occupancy": [int(x) for x in counts],
        "min_share": float(min(shares)) if shares else None,
        "max_share": float(max(shares)) if shares else None,
        "occupancy_cv": cv,
    }


def compute_edges(
    values: np.ndarray,
    *,
    bins: int,
    strategy: str = "equal_frequency",
    anchors: Sequence[float] = (),
    zero_mass_threshold: float = DEFAULT_ZERO_MASS_THRESHOLD,
    min_positive_for_zero_split: int = DEFAULT_MIN_POSITIVE_FOR_ZERO_SPLIT,
    eps: float = 1e-9,
) -> tuple[list[float], str, dict[str, Any]]:
    """Build bin edges with near-equal occupancy.

    Returns (edges, strategy_used, diagnostics).
    """
    finite = values[np.isfinite(values)]
    diagnostics: dict[str, Any] = {
        "n_values": int(finite.size),
        "n_unique": int(np.unique(finite).size) if finite.size else 0,
    }
    if finite.size == 0:
        return [], "empty", diagnostics
    unique = np.unique(finite)
    if unique.size == 1:
        return [float(unique[0])], "constant", {**diagnostics, "zero_mass_frac": float(np.mean(np.abs(finite) <= eps))}

    zero_mask = np.abs(finite) <= eps
    zero_mass = float(np.mean(zero_mask))
    diagnostics["zero_mass_frac"] = zero_mass
    want_zero = (
        strategy in {"zero_inflated_equal_frequency", "equal_frequency_with_anchors"}
        or strategy == "equal_frequency"
    ) and zero_mass >= zero_mass_threshold and int((~zero_mask).sum()) >= min_positive_for_zero_split

    # Low-cardinality discrete: unique midpoints / values as edges.
    if unique.size <= max(2, bins):
        if unique.size == 2:
            mid = float((unique[0] + unique[1]) / 2.0)
            edges = [float(unique[0]), mid, float(unique[1])]
        else:
            mids = [float(unique[0])]
            for left, right in zip(unique[:-1], unique[1:]):
                mids.append(float((left + right) / 2.0))
            mids.append(float(unique[-1]))
            edges = mids
        edges = _finalize_edges(edges, anchors=() if want_zero else anchors)
        used = "low_cardinality"
        counts = _occupancy(finite, edges)
        diagnostics.update(_occupancy_stats(counts))
        return edges, used, diagnostics

    if want_zero:
        positive = finite[~zero_mask]
        pos_bins = max(1, bins - 1)
        pos_edges = _quantile_edges(positive, pos_bins)
        # Reserve B00 for near-zero: [min_or_0, eps] then positive quantiles.
        lo = float(min(0.0, float(finite.min())))
        edges = [lo, float(eps)] + pos_edges[1:]
        used = "zero_inflated_equal_frequency"
        if strategy == "equal_frequency_with_anchors" and anchors:
            edges = _finalize_edges(edges, anchors=anchors)
            used = "zero_inflated_equal_frequency_with_anchors"
        else:
            edges = _finalize_edges(edges, anchors=())
    else:
        edges = _quantile_edges(finite, bins)
        if strategy == "equal_frequency_with_anchors" and anchors:
            edges = _finalize_edges(edges, anchors=anchors)
            used = "equal_frequency_with_anchors"
        else:
            edges = _finalize_edges(edges, anchors=())
            used = "equal_frequency"

    counts = _occupancy(finite, edges)
    diagnostics.update(_occupancy_stats(counts))
    return edges, used, diagnostics


def _quantile_edges(values: np.ndarray, bins: int) -> list[float]:
    bins = max(1, int(bins))
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, qs)
    return [float(x) for x in edges]


def _finalize_edges(edges: Sequence[float], anchors: Sequence[float] = ()) -> list[float]:
    vals = [float(x) for x in edges if np.isfinite(x)]
    if anchors:
        lo, hi = min(vals), max(vals)
        for a in anchors:
            if lo < float(a) < hi:
                vals.append(float(a))
    uniq = sorted(set(np.round(vals, 12).tolist()))
    # Restore original precision for endpoints where possible
    if not uniq:
        return []
    # Use sorted unique floats
    cleaned: list[float] = []
    for x in sorted(vals):
        if not cleaned or abs(cleaned[-1] - x) > 1e-12:
            cleaned.append(float(x))
    return cleaned


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
    absolute_policy: str = "equal_frequency"
    relative_scope: str = "full_public_farm_distribution"
    strategy: str = "equal_frequency"
    zero_mass_frac: float = 0.0
    n_bins_requested: int = DEFAULT_BINS
    n_bins_effective: int = 1
    occupancy_cv: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def constant(self) -> bool:
        return len(self.edges) <= 1

    def token_suffix(self, value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return f"{self.kind}_NULL"
        if self.constant:
            return f"{self.kind}_CONSTANT"
        import bisect

        position = bisect.bisect_left(self.edges[1:-1], float(value))
        return f"{self.kind}_B{position:02d}"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


class BinningRegistryV2:
    def __init__(
        self,
        version: str = "v2_transductive",
        bins: int = DEFAULT_BINS,
        policy: BinningPolicy | None = None,
    ) -> None:
        self.version = version
        self.bins = bins
        self.policy = policy or BinningPolicy()
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

        feature_summaries: dict[str, Any] = {}
        for feature, values in sorted(grouped.items()):
            array = np.asarray(values, dtype=np.float64)
            spec = schema.get(feature)
            n_unique = int(np.unique(array).size)
            summary: dict[str, Any] = {"n_values": int(array.size), "n_unique": n_unique}

            if spec.absolute_encoding:
                plan = self.policy.plan_for(
                    feature,
                    channel="ABS",
                    feature_type=spec.type,
                    n_values=int(array.size),
                    n_unique=n_unique,
                    schema_n_bins=getattr(spec, "n_bins_abs", None),
                    default_bins_override=self.bins,
                )
                rule = self._make_rule(feature, "ABS", array, plan, scope_id=None)
                self.rules[(feature, "ABS", None)] = rule
                summary["abs"] = {
                    "strategy": rule.strategy,
                    "n_bins_requested": rule.n_bins_requested,
                    "n_bins_effective": rule.n_bins_effective,
                    "occupancy_cv": rule.occupancy_cv,
                    "zero_mass_frac": rule.zero_mass_frac,
                }

            if spec.global_relative_encoding:
                plan = self.policy.plan_for(
                    feature,
                    channel="GLOBAL_REL",
                    feature_type=spec.type,
                    n_values=int(array.size),
                    n_unique=n_unique,
                    schema_n_bins=getattr(spec, "n_bins_global_rel", None),
                    default_bins_override=self.bins,
                )
                rule = self._make_rule(feature, "GLOBAL_REL", array, plan, scope_id=None)
                self.rules[(feature, "GLOBAL_REL", None)] = rule
                summary["global_rel"] = {
                    "strategy": rule.strategy,
                    "n_bins_requested": rule.n_bins_requested,
                    "n_bins_effective": rule.n_bins_effective,
                    "occupancy_cv": rule.occupancy_cv,
                    "zero_mass_frac": rule.zero_mass_frac,
                }
            feature_summaries[feature] = summary

        farm_strategy_hist: dict[str, int] = {}
        for (feature, farm), values in sorted(farms.items()):
            spec = schema.features.get(feature)
            if not spec or not spec.farm_relative_encoding:
                continue
            array = np.asarray(values, dtype=np.float64)
            plan = self.policy.plan_for(
                feature,
                channel="FARM_REL",
                feature_type=spec.type,
                n_values=int(array.size),
                n_unique=int(np.unique(array).size),
                schema_n_bins=getattr(spec, "n_bins_farm_rel", None),
                default_bins_override=self.bins,
            )
            rule = self._make_rule(
                feature,
                "FARM_REL",
                array,
                plan,
                scope_id=farm,
                relative_scope="full_public_farm_distribution",
            )
            self.rules[(feature, "FARM_REL", farm)] = rule
            farm_strategy_hist[rule.strategy] = farm_strategy_hist.get(rule.strategy, 0) + 1

        policy_meta = {
            "edge_policy": self.policy.meta.get("edge_policy", "equal_frequency_v2"),
            "abs_policy": self.policy.meta.get("abs_policy", "equal_frequency_with_zero_inflation_and_optional_anchors"),
            "global_rel_policy": self.policy.meta.get("global_rel_policy", "equal_frequency_with_zero_inflation_and_optional_anchors"),
            "farm_rel_policy": self.policy.meta.get("farm_rel_policy", "per_farm_equal_frequency_with_zero_inflation"),
            "zero_mass_threshold": float(self.policy.meta.get("zero_mass_threshold", DEFAULT_ZERO_MASS_THRESHOLD)),
            "policy_version": self.policy.meta.get("policy_version", "inline_default"),
            "default_bins": self.policy.default_bins() if self.policy.meta else self.bins,
        }
        hasher.update(
            canonical_json(
                {
                    "bins": self.bins,
                    "policy": policy_meta,
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
            **policy_meta,
            "outlier_policy": "none_equal_frequency_uses_full_support",
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
            "feature_summaries": feature_summaries,
            "farm_rule_count": sum(1 for k in self.rules if k[1] == "FARM_REL"),
            "farm_strategy_histogram": farm_strategy_hist,
        }
        self.meta["registry_hash"] = hashlib.sha256(
            json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _make_rule(
        self,
        feature: str,
        kind: str,
        array: np.ndarray,
        plan: ChannelBinPlan,
        *,
        scope_id: Optional[str],
        relative_scope: str = "n/a",
    ) -> BinRuleV2:
        edges, used, diag = compute_edges(
            array,
            bins=plan.n_bins,
            strategy=plan.strategy,
            anchors=plan.anchors,
            zero_mass_threshold=plan.zero_mass_threshold,
            min_positive_for_zero_split=int(
                self.policy.meta.get("min_positive_for_zero_split", DEFAULT_MIN_POSITIVE_FOR_ZERO_SPLIT)
            ),
        )
        n_eff = max(1, len(edges) - 1) if len(edges) >= 2 else 1
        abs_policy = used if kind == "ABS" else "n/a"
        return BinRuleV2(
            feature=feature,
            kind=kind,
            edges=tuple(edges),
            scope_id=scope_id,
            absolute_policy=abs_policy if kind == "ABS" else "n/a",
            relative_scope=relative_scope if kind != "ABS" else "n/a",
            causal=False,
            uses_problem_inputs=True,
            strategy=used,
            zero_mass_frac=float(diag.get("zero_mass_frac") or 0.0),
            n_bins_requested=int(plan.n_bins),
            n_bins_effective=int(n_eff if len(edges) != 1 else 1),
            occupancy_cv=float(diag.get("occupancy_cv") or 0.0),
            diagnostics=diag,
        )

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
        Path(path).write_text(json.dumps(self.payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BinningRegistryV2":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
                absolute_policy=rule.get("absolute_policy", "equal_frequency"),
                relative_scope=rule.get("relative_scope", "full_public_farm_distribution"),
                strategy=rule.get("strategy", "equal_frequency"),
                zero_mass_frac=float(rule.get("zero_mass_frac") or 0.0),
                n_bins_requested=int(rule.get("n_bins_requested") or max(1, len(edges) - 1)),
                n_bins_effective=int(rule.get("n_bins_effective") or max(1, len(edges) - 1 if len(edges) > 1 else 1)),
                occupancy_cv=float(rule.get("occupancy_cv") or 0.0),
                diagnostics=dict(rule.get("diagnostics") or {}),
            )
        obj.bins = int(payload["meta"].get("bin_count", DEFAULT_BINS))
        return obj


def fit_binning_v2(
    schema: FeatureSchema,
    rows: Iterable[tuple[str, Optional[str], Optional[float], str]],
    *,
    source_file_hashes: Mapping[str, str] | None = None,
    example_count: int = 0,
    problem_count: int = 0,
    code_commit_hash: str = "",
    bins: int = DEFAULT_BINS,
    policy: BinningPolicy | None = None,
    policy_path: Path | None = None,
) -> BinningRegistryV2:
    if policy is None and policy_path is not None:
        policy = BinningPolicy.load(policy_path)
    registry = BinningRegistryV2(bins=bins, policy=policy)
    registry.fit(
        schema,
        rows,
        source_file_hashes=source_file_hashes,
        example_count=example_count,
        problem_count=problem_count,
        code_commit_hash=code_commit_hash,
    )
    return registry
