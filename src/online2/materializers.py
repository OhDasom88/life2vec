"""Catalog-driven narrative instance matchers.

Matchers only inspect the current observation and its past context. Population
quantiles are fit once over the public pool and are never calculated from a
future window attached to an individual event.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Protocol
from zoneinfo import ZoneInfo

import numpy as np

VIEW_NAMES = {
    "E_environment",
    "A_actuator",
    "G_growth",
    "R_rootzone",
    "I_images",
    "interpretation",
}
NON_ORDERED = {
    "RELATION_ORDERED",
    "SET_COMPARISON",
    "RETRIEVAL_PAIR",
    "PERIOD_LOGICAL",
    "PERIOD_LOGICAL_ALIGNMENT",
}
LOCAL_TZ = ZoneInfo("Asia/Seoul")


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


@dataclass(frozen=True)
class NarrativeTemplate:
    narrative_id: str
    materialization_key: str
    event_views: tuple[str, ...]
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    min_events: int
    max_events: int
    timestamp_precision: str
    order_semantics: str
    requested_op_eligible: bool
    category: str
    time_resolution: str
    alignment_policy_id: str
    entity_scope: str
    window_definition: str
    threshold_type: str
    threshold_or_rule: str

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "NarrativeTemplate":
        explicit_views = _split(row.get("event_views", ""))
        source_views = tuple(
            view for view in _split(row.get("data_sources", "")) if view in VIEW_NAMES
        )
        key = row.get("materialization_key", "").strip()
        if not explicit_views and not source_views:
            if key in {"growth_any", "crosszone_growth"}:
                source_views = ("G_growth",)
            elif key.startswith("image_"):
                source_views = ("I_images", "E_environment", "R_rootzone", "G_growth")
            elif key == "score90_terminal_relation":
                source_views = ("interpretation", "E_environment")
            else:
                source_views = ("E_environment", "A_actuator", "R_rootzone")
        if key == "score90_terminal_relation" and "interpretation" not in source_views:
            source_views = source_views + ("interpretation",)
        max_events_raw = row.get("max_events", "").strip()
        min_events_raw = row.get("min_events", "").strip()
        return cls(
            narrative_id=row["narrative_id"].strip(),
            materialization_key=key,
            event_views=explicit_views or source_views,
            required_columns=_split(row.get("required_columns", "")),
            optional_columns=_split(row.get("optional_columns", "")),
            min_events=max(1, int(float(min_events_raw))) if min_events_raw else 1,
            max_events=max(1, int(float(max_events_raw))) if max_events_raw else 24,
            timestamp_precision=row.get("timestamp_precision", "1h").strip() or "1h",
            order_semantics=(
                row.get("order_semantics", "STRICT_CHRONOLOGICAL").strip()
                or "STRICT_CHRONOLOGICAL"
            ),
            requested_op_eligible=row.get("op_eligible", "false").lower() == "true",
            category=row.get("category", "").strip(),
            time_resolution=row.get("time_resolution", "").strip(),
            alignment_policy_id=row.get("alignment_policy_id", "").strip(),
            entity_scope=row.get("entity_scope", "").strip(),
            window_definition=row.get("window_definition", "").strip(),
            threshold_type=row.get("threshold_type", "").strip(),
            threshold_or_rule=row.get("threshold_or_rule", "").strip(),
        )


class SequenceEmitter(Protocol):
    event_index: Mapping[tuple[str, str], list[dict[str, Any]]]
    points: Mapping[tuple[str, str, str], dict[str, Any]]
    case_sets: Mapping[str, str]

    def emit_template_sequence(
        self,
        template: NarrativeTemplate,
        events: list[dict[str, Any]],
        flags: list[str],
    ) -> tuple[bool, str]:
        ...


@dataclass
class MatchReport:
    narrative_id: str
    materialization_key: str
    matcher: str
    status: str = "PASS"
    reason: str = "MATCHER_EXECUTED"
    trigger_count: int = 0
    unique_count: int = 0
    duplicate_count: int = 0
    dropped_short_count: int = 0
    dropped_long_count: int = 0

    def record(self, emitted: bool, reason: str) -> None:
        if emitted:
            self.unique_count += 1
        elif reason == "EXACT_DUPLICATE":
            self.duplicate_count += 1
        elif reason == "TOO_SHORT":
            self.dropped_short_count += 1
        elif reason == "TOO_LONG":
            self.dropped_long_count += 1

    def as_dict(self) -> dict[str, Any]:
        if self.trigger_count == 0:
            self.reason = "NO_TRIGGER_IN_PUBLIC_POOL"
        elif self.unique_count == 0:
            self.reason = "TRIGGERS_FOUND_BUT_NO_VALID_SEQUENCE"
        return vars(self)


def permutation_invariant_summary(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return order-independent aggregate provenance for a set of local events."""
    identifiers = sorted(str(event["id"]) for event in events)
    return {
        "parent_segment_ids": identifiers,
        "aggregation_policy": "SORTED_SET_MEMBERSHIP",
        "member_count": len(identifiers),
        "permutation_invariant": True,
    }


def _number(point: Mapping[str, Any], feature: str) -> Optional[float]:
    value = point["values"].get(feature)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and isfinite(float(value)):
        return float(value)
    return None


def _local_datetime(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(LOCAL_TZ)


def _events(point: Mapping[str, Any], template: NarrativeTemplate) -> list[dict[str, Any]]:
    views = set(template.event_views)
    if not views:
        return list(point["events"].values())
    return [
        event for view, event in sorted(point["events"].items())
        if view in views
    ]


def _has_required(point: Mapping[str, Any], template: NarrativeTemplate) -> bool:
    identity = {"farm_id", "zone_id", "timestamp", "observation_date"}
    required = [column for column in template.required_columns if column not in identity]
    return all(
        column in point["values"] and point["values"][column] is not None
        for column in required
    )


def _flatten(
    points: Iterable[Mapping[str, Any]], template: NarrativeTemplate
) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for point in points:
        for event in _events(point, template):
            events[event["id"]] = event
    return sorted(events.values(), key=lambda event: (event["timestamp"], event["id"]))


def _histories(builder: SequenceEmitter) -> dict[tuple[str, str], list[dict[str, Any]]]:
    histories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in builder.points.values():
        if any(view != "G_growth" for view in point["events"]):
            histories[(point["farm"], point["zone"])].append(point)
    for points in histories.values():
        points.sort(key=lambda point: point["timestamp"])
    return histories


def _growth_histories(builder: SequenceEmitter) -> dict[tuple[str, str], list[dict[str, Any]]]:
    histories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in builder.points.values():
        if "G_growth" in point["events"]:
            histories[(point["farm"], point["zone"])].append(point)
    for points in histories.values():
        points.sort(key=lambda point: point["timestamp"])
    return histories


def _quantiles(
    histories: Mapping[tuple[str, str], list[dict[str, Any]]]
) -> dict[tuple[str, str, str], tuple[float, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    farm_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    entity_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (farm, zone), points in histories.items():
        for point in points:
            for feature, value in point["values"].items():
                if isinstance(value, (int, float)) and isfinite(float(value)):
                    values[feature].append(float(value))
                    farm_values[(farm, "*", feature)].append(float(value))
                    entity_values[(farm, zone, feature)].append(float(value))
            sensor = _number(point, "ec_sensor")
            root = _number(point, "substrate_ec_ds_m")
            if sensor is not None and root is not None:
                gap = abs(sensor - root)
                values["__ec_gap__"].append(gap)
                farm_values[(farm, "*", "__ec_gap__")].append(gap)
                entity_values[(farm, zone, "__ec_gap__")].append(gap)
    result = {
        ("*", "*", feature): (
            float(np.quantile(items, 0.1)), float(np.quantile(items, 0.9))
        )
        for feature, items in values.items() if items
    }
    result.update({
        key: (float(np.quantile(items, 0.1)), float(np.quantile(items, 0.9)))
        for key, items in farm_values.items() if items
    })
    result.update({
        key: (float(np.quantile(items, 0.1)), float(np.quantile(items, 0.9)))
        for key, items in entity_values.items() if items
    })
    if "__ec_gap__" in values:
        items = values["__ec_gap__"]
        result[("*", "*", "__ec_gap__")] = (
            float(np.quantile(items, 0.05)), float(np.quantile(items, 0.95))
        )
    for key, items in entity_values.items():
        if key[2] == "__ec_gap__":
            result[key] = (
                float(np.quantile(items, 0.05)), float(np.quantile(items, 0.95))
            )
    for key, items in farm_values.items():
        if key[2] == "__ec_gap__":
            result[key] = (
                float(np.quantile(items, 0.05)), float(np.quantile(items, 0.95))
            )
    if "co2_ppm" in values:
        items = values["co2_ppm"]
        result[("*", "*", "__co2_p99__")] = (
            float(np.quantile(items, 0.01)), float(np.quantile(items, 0.99))
        )
    for key, items in entity_values.items():
        if key[2] == "co2_ppm":
            result[(key[0], key[1], "__co2_p99__")] = (
                float(np.quantile(items, 0.01)), float(np.quantile(items, 0.99))
            )
    return result


def _jump_thresholds(
    histories: Mapping[tuple[str, str], list[dict[str, Any]]]
) -> dict[tuple[str, str, str], float]:
    deltas: dict[str, list[float]] = defaultdict(list)
    entity_deltas: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (farm, zone), points in histories.items():
        for previous, current in zip(points, points[1:]):
            for feature in current["values"]:
                before, after = _number(previous, feature), _number(current, feature)
                if before is not None and after is not None:
                    delta = abs(after - before)
                    deltas[feature].append(delta)
                    entity_deltas[(farm, zone, feature)].append(delta)
    result = {
        ("*", "*", feature): float(np.quantile(values, 0.95))
        for feature, values in deltas.items() if values
    }
    result.update({
        key: float(np.quantile(values, 0.95))
        for key, values in entity_deltas.items() if values
    })
    return result


def _bounds(
    quantiles: Mapping[tuple[str, str, str], tuple[float, float]],
    point: Mapping[str, Any],
    feature: str,
) -> Optional[tuple[float, float]]:
    return quantiles.get(
        (point["farm"], "*", feature),
        quantiles.get(
            (point["farm"], point["zone"], feature),
            quantiles.get(("*", "*", feature)),
        ),
    )


def _entity_bounds(
    quantiles: Mapping[tuple[str, str, str], tuple[float, float]],
    point: Mapping[str, Any],
    feature: str,
) -> Optional[tuple[float, float]]:
    return quantiles.get(
        (point["farm"], point["zone"], feature),
        _bounds(quantiles, point, feature),
    )


def _jump_limit(
    jumps: Mapping[tuple[str, str, str], float],
    point: Mapping[str, Any],
    feature: str,
) -> Optional[float]:
    return jumps.get(
        (point["farm"], point["zone"], feature),
        jumps.get(("*", "*", feature)),
    )


def _emit_window(
    builder: SequenceEmitter,
    template: NarrativeTemplate,
    report: MatchReport,
    points: list[dict[str, Any]],
    end: int,
    flags: Optional[list[str]] = None,
) -> None:
    report.trigger_count += 1
    candidate = _flatten(points[: end + 1], template)
    if len(candidate) > template.max_events:
        candidate = candidate[-template.max_events :]
        flags = list(flags or []) + ["CATALOG_MAX_EVENTS_APPLIED"]
    emitted, reason = builder.emit_template_sequence(
        template, candidate, list(flags or []) + ["FEATURE_WINDOW_PAST_ONLY"]
    )
    report.record(emitted, reason)


def _parse_numeric(value: str) -> float:
    return float(value)


def _predicate(
    key: str,
    point: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
    quantiles: Mapping[tuple[str, str, str], tuple[float, float]],
    jumps: Mapping[tuple[str, str, str], float],
    run_length: int,
) -> bool:
    parts = key.split(":")
    kind = parts[0]
    if kind == "positive":
        value = _number(point, parts[1])
        return value is not None and value > 0
    if kind == "change":
        return previous is not None and point["values"].get(parts[1]) != previous["values"].get(parts[1])
    if kind in {"lt", "le", "gt"}:
        value = _number(point, parts[1])
        target = _parse_numeric(parts[2])
        return value is not None and (
            value < target if kind == "lt" else value <= target if kind == "le" else value > target
        )
    if kind == "quantile_high":
        value = _number(point, parts[1])
        bounds = _bounds(quantiles, point, parts[1])
        return value is not None and bounds is not None and value >= bounds[1]
    if kind == "quantile_low":
        value = _number(point, parts[1])
        bounds = _bounds(quantiles, point, parts[1])
        return value is not None and bounds is not None and value <= bounds[0]
    if kind == "jump":
        if previous is None:
            return False
        before, after = _number(previous, parts[1]), _number(point, parts[1])
        limit = _jump_limit(jumps, point, parts[1])
        return (
            before is not None and after is not None and limit is not None
            and abs(after - before) >= limit and limit > 0
        )
    if kind == "reset":
        if previous is None:
            return False
        before, after = _number(previous, parts[1]), _number(point, parts[1])
        return before is not None and after is not None and before > 0 and after <= 0
    if kind == "fixed_zero":
        value = _number(point, parts[1])
        return value == 0 and run_length >= 2
    if kind == "run_ge":
        value = _number(point, parts[1])
        return value is not None and value >= float(parts[2]) and run_length >= int(parts[3])
    if kind == "and":
        first, second = _number(point, parts[1]), _number(point, parts[3])
        return (
            first is not None and second is not None
            and first >= float(parts[2]) and second <= float(parts[4])
        )
    if kind == "diff_positive":
        left, right = _number(point, parts[1]), _number(point, parts[2])
        return left is not None and right is not None and left > right
    if kind == "joint_quantile":
        first, second = _number(point, parts[1]), _number(point, parts[2])
        first_bounds = _bounds(quantiles, point, parts[1])
        second_bounds = _bounds(quantiles, point, parts[2])
        if first is None or second is None or first_bounds is None or second_bounds is None:
            return False
        return first >= first_bounds[1] and second >= second_bounds[1]
    return False


class MaterializerRegistry:
    SIMPLE_PREFIXES = {
        "positive", "change", "lt", "le", "gt", "quantile_high", "quantile_low",
        "jump", "reset", "fixed_zero", "run_ge", "and", "diff_positive",
        "joint_quantile",
    }
    NAMED = {
        "all_hourly", "solar_up", "solar_down", "growth_any", "crosszone_growth",
        "mixed_growth", "crosszone_hourly", "score90_terminal_relation",
        "fan_pump_mismatch", "co2_high_off", "ec_sensor_root_divergence",
        "cross_zone_actuation", "rootcold12", "root_simultaneous_jump",
        "noon_solar_hourly", "image_preceding_env", "image_root_context",
        "image_growth_alignment", "public_observation_similarity",
    }

    @classmethod
    def supports(cls, key: str) -> bool:
        return key in cls.NAMED or key.split(":", 1)[0] in cls.SIMPLE_PREFIXES

    @classmethod
    def coverage(cls, templates: Iterable[NarrativeTemplate]) -> dict[str, bool]:
        return {template.narrative_id: cls.supports(template.materialization_key) for template in templates}

    def materialize(
        self, builder: SequenceEmitter, templates: Iterable[NarrativeTemplate]
    ) -> list[dict[str, Any]]:
        hourly = _histories(builder)
        growth = _growth_histories(builder)
        quantiles = _quantiles(hourly)
        jumps = _jump_thresholds(hourly)
        reports = []
        for template in templates:
            report = MatchReport(
                template.narrative_id, template.materialization_key,
                template.materialization_key.split(":", 1)[0],
            )
            if not self.supports(template.materialization_key):
                report.status = "FAIL_UNSUPPORTED_SEMANTICS"
                report.reason = "NO_REGISTERED_MATCHER"
                reports.append(report.as_dict())
                continue
            self._dispatch(builder, template, report, hourly, growth, quantiles, jumps)
            reports.append(report.as_dict())
        return reports

    def _dispatch(
        self,
        builder: SequenceEmitter,
        template: NarrativeTemplate,
        report: MatchReport,
        hourly: Mapping[tuple[str, str], list[dict[str, Any]]],
        growth: Mapping[tuple[str, str], list[dict[str, Any]]],
        quantiles: Mapping[tuple[str, str, str], tuple[float, float]],
        jumps: Mapping[tuple[str, str, str], float],
    ) -> None:
        key = template.materialization_key
        if key == "all_hourly":
            self._all_hourly(builder, template, report, hourly)
        elif key in {"growth_any", "mixed_growth"}:
            self._growth(builder, template, report, hourly, growth, key == "mixed_growth")
        elif key in {"crosszone_growth", "crosszone_hourly", "cross_zone_actuation"}:
            source = growth if key == "crosszone_growth" else hourly
            self._crosszone(builder, template, report, source)
        elif key.startswith("image_"):
            self._external(builder, template, report, key)
        elif key == "score90_terminal_relation":
            self._interpretation(builder, template, report)
        elif key == "public_observation_similarity":
            self._similarity(builder, template, report, hourly)
        else:
            self._triggered(builder, template, report, hourly, quantiles, jumps)

    def _all_hourly(
        self, builder: SequenceEmitter, template: NarrativeTemplate, report: MatchReport,
        histories: Mapping[tuple[str, str], list[dict[str, Any]]],
    ) -> None:
        for points in histories.values():
            by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for point in points:
                if _has_required(point, template):
                    by_day[_local_datetime(point["timestamp"]).date().isoformat()].append(point)
            for day_points in by_day.values():
                report.trigger_count += len(day_points)
                events = _flatten(day_points, template)[-template.max_events :]
                emitted, reason = builder.emit_template_sequence(template, events, [])
                report.record(emitted, reason)

    def _growth(
        self, builder: SequenceEmitter, template: NarrativeTemplate, report: MatchReport,
        hourly: Mapping[tuple[str, str], list[dict[str, Any]]],
        growth: Mapping[tuple[str, str], list[dict[str, Any]]],
        mixed: bool,
    ) -> None:
        for entity, points in growth.items():
            for index, point in enumerate(points):
                if not _has_required(point, template):
                    continue
                report.trigger_count += 1
                context = list(points[: index + 1])
                if mixed:
                    past = [
                        candidate for candidate in hourly.get(entity, [])
                        if candidate["timestamp"] <= point["timestamp"]
                    ]
                    context = past[-template.max_events :] + context
                events = _flatten(context, template)[-template.max_events :]
                emitted, reason = builder.emit_template_sequence(template, events, [])
                report.record(emitted, reason)

    def _crosszone(
        self, builder: SequenceEmitter, template: NarrativeTemplate, report: MatchReport,
        histories: Mapping[tuple[str, str], list[dict[str, Any]]],
    ) -> None:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for (farm, _zone), points in histories.items():
            for point in points:
                grouped[(farm, point["timestamp"])].append(point)
        candidates = []
        for (farm, timestamp), points in grouped.items():
            valid = [point for point in points if _has_required(point, template)]
            if len({point["zone"] for point in valid}) < 4:
                continue
            ranges = []
            for feature in template.required_columns:
                values = [
                    value for point in valid
                    if (value := _number(point, feature)) is not None
                ]
                if len(values) >= 2:
                    ranges.append(max(values) - min(values))
            candidates.append((farm, timestamp, valid, max(ranges, default=0.0)))
        thresholds = {}
        if template.materialization_key == "crosszone_hourly":
            by_farm: dict[str, list[float]] = defaultdict(list)
            for farm, _timestamp, _points, metric in candidates:
                by_farm[farm].append(metric)
            thresholds = {
                farm: float(np.quantile(metrics, 0.99))
                for farm, metrics in by_farm.items()
            }
        for farm, _timestamp, points, metric in candidates:
            if thresholds and metric < thresholds[farm]:
                continue
            report.trigger_count += len(points)
            events = _flatten(points, template)[-template.max_events :]
            emitted, reason = builder.emit_template_sequence(
                template, events, ["PERMUTATION_INVARIANT_SET"]
            )
            report.record(emitted, reason)

    def _triggered(
        self, builder: SequenceEmitter, template: NarrativeTemplate, report: MatchReport,
        histories: Mapping[tuple[str, str], list[dict[str, Any]]],
        quantiles: Mapping[tuple[str, str, str], tuple[float, float]],
        jumps: Mapping[tuple[str, str, str], float],
    ) -> None:
        key = template.materialization_key
        feature = key.split(":")[1] if ":" in key else ""
        for points in histories.values():
            run_length = 0
            for index, point in enumerate(points):
                if not _has_required(point, template):
                    run_length = 0
                    continue
                previous = points[index - 1] if index else None
                value = _number(point, feature)
                if key.startswith("run_ge:"):
                    threshold = float(key.split(":")[2])
                    run_length = run_length + 1 if value is not None and value >= threshold else 0
                elif key.startswith("fixed_zero:"):
                    run_length = run_length + 1 if value == 0 else 0
                elif key == "rootcold12":
                    root_temp = _number(point, "substrate_temp_c")
                    run_length = run_length + 1 if root_temp is not None and root_temp < 12 else 0
                else:
                    run_length = 0
                matched = self._named_predicate(key, point, previous, quantiles, jumps)
                if matched is None:
                    matched = _predicate(
                        key, point, previous, quantiles, jumps, run_length
                    )
                if key == "rootcold12":
                    matched = matched and run_length >= 4
                if matched:
                    _emit_window(builder, template, report, points, index)

    def _named_predicate(
        self, key: str, point: Mapping[str, Any], previous: Optional[Mapping[str, Any]],
        quantiles: Mapping[tuple[str, str, str], tuple[float, float]],
        jumps: Mapping[tuple[str, str, str], float],
    ) -> Optional[bool]:
        if key in {"solar_up", "solar_down"}:
            before = _number(previous, "outside_solar_radiation") if previous else None
            after = _number(point, "outside_solar_radiation")
            if before is None or after is None:
                return False
            return after > before if key == "solar_up" else after < before
        if key == "fan_pump_mismatch":
            fan, pump = _number(point, "fcu_fan"), _number(point, "fcu_pump")
            return fan is not None and pump is not None and fan != pump
        if key == "co2_high_off":
            co2, supply = _number(point, "co2_ppm"), _number(point, "co2_supply")
            bounds = _entity_bounds(quantiles, point, "__co2_p99__")
            return (
                co2 is not None and supply is not None and bounds is not None
                and co2 >= bounds[1] and supply <= 0
            )
        if key == "ec_sensor_root_divergence":
            sensor, root = _number(point, "ec_sensor"), _number(point, "substrate_ec_ds_m")
            bounds = _entity_bounds(quantiles, point, "__ec_gap__")
            return (
                sensor is not None and root is not None and bounds is not None
                and abs(sensor - root) >= bounds[1]
            )
        if key == "rootcold12":
            value = _number(point, "substrate_temp_c")
            return value is not None and value <= 12
        if key == "root_simultaneous_jump":
            if previous is None:
                return False
            features = (
                "substrate_temp_c",
                "substrate_water_content_pct",
                "substrate_ec_ds_m",
            )
            for feature in features:
                before = _number(previous, feature)
                after = _number(point, feature)
                limit = _jump_limit(jumps, point, feature)
                if (
                    before is None or after is None or limit is None
                    or limit <= 0 or abs(after - before) < limit
                ):
                    return False
            return True
        if key == "noon_solar_hourly":
            solar = _number(point, "outside_solar_radiation")
            return _local_datetime(point["timestamp"]).hour == 12 and solar is not None and solar > 0
        if key.startswith(tuple(self.SIMPLE_PREFIXES)):
            return None
        return False

    def _external(
        self, builder: SequenceEmitter, template: NarrativeTemplate, report: MatchReport,
        key: str,
    ) -> None:
        context_view = {
            "image_preceding_env": "E_environment",
            "image_root_context": "R_rootzone",
            "image_growth_alignment": "G_growth",
        }[key]
        for (farm, modality), external in builder.event_index.items():
            if modality != "I_images":
                continue
            context = builder.event_index.get((farm, context_view), [])
            for event in external:
                report.trigger_count += 1
                past = [item for item in context if item["timestamp"] <= event["timestamp"]]
                events = past[-(template.max_events - 1) :] + [event]
                emitted, reason = builder.emit_template_sequence(
                    template, events, ["PERIOD_ALIGNED", "EMBEDDING_PENDING"]
                )
                report.record(emitted, reason)

    def _interpretation(
        self, builder: SequenceEmitter, template: NarrativeTemplate, report: MatchReport
    ) -> None:
        for (farm, modality), interpretations in builder.event_index.items():
            if modality != "interpretation":
                continue
            context = []
            for view in template.event_views:
                if view != "interpretation":
                    context.extend(builder.event_index.get((farm, view), []))
            context.sort(key=lambda event: (event["timestamp"], event["id"]))
            for event in interpretations:
                report.trigger_count += 1
                past = [item for item in context if item["timestamp"] <= event["timestamp"]]
                events = past[-(template.max_events - 1) :] + [event]
                emitted, reason = builder.emit_template_sequence(
                    template, events, ["PERIOD_ALIGNED", "RELATION_TERMINAL", "EMBEDDING_PENDING"]
                )
                report.record(emitted, reason)

    def _similarity(
        self, builder: SequenceEmitter, template: NarrativeTemplate, report: MatchReport,
        histories: Mapping[tuple[str, str], list[dict[str, Any]]],
    ) -> None:
        by_farm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (farm, _zone), points in histories.items():
            by_farm[farm].extend(points)
        vectors = {}
        features = (
            "inside_temp_c",
            "inside_humidity_pct",
            "substrate_temp_c",
            "substrate_water_content_pct",
            "substrate_ec_ds_m",
            "plant_height_cm",
            "leaf_count",
        )
        for farm, points in by_farm.items():
            vector = []
            complete = True
            for feature in features:
                values = [
                    value for point in points
                    if (value := _number(point, feature)) is not None
                ]
                if feature in template.required_columns and not values:
                    complete = False
                    break
                vector.append(mean(values) if values else 0.0)
            if complete:
                vectors[farm] = tuple(vector)
        if vectors:
            farm_order = sorted(vectors)
            matrix = np.asarray([vectors[farm] for farm in farm_order], dtype=np.float64)
            center = matrix.mean(axis=0)
            scale = matrix.std(axis=0)
            scale[scale == 0] = 1.0
            vectors = {
                farm: tuple((matrix[index] - center) / scale)
                for index, farm in enumerate(farm_order)
            }
        examples = [farm for farm in vectors if builder.case_sets.get(farm) == "example_set"]
        problems = [farm for farm in vectors if builder.case_sets.get(farm) == "problem_set"]
        for problem in sorted(problems):
            if not examples:
                continue
            example = min(
                examples,
                key=lambda farm: (
                    sum((left - right) ** 2 for left, right in zip(vectors[problem], vectors[farm])),
                    farm,
                ),
            )
            left = sorted(by_farm[example], key=lambda point: point["timestamp"])[-1:]
            right = sorted(by_farm[problem], key=lambda point: point["timestamp"])[-1:]
            report.trigger_count += 1
            events = _flatten(left + right, template)[-template.max_events :]
            emitted, reason = builder.emit_template_sequence(
                template, events, ["SET_COMPARISON", "PUBLIC_OBSERVATIONS_ONLY"]
            )
            report.record(emitted, reason)
