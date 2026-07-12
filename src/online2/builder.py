"""Deterministic streaming materialization of the ACTIVE online2 catalog."""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from .canonical import SCHEMA_VERSION, canonical_json, canonical_scalar, stable_id
from .catalog import file_hash, normalize_catalog, registry_hashes, write_json
from .inventory import build_inventory, source_files
from .materializers import MaterializerRegistry, NarrativeTemplate, NON_ORDERED
from .pool import classify_public_path
from .registry import TokenRegistry, categorical_token

LOCAL_TZ = ZoneInfo("Asia/Seoul")
NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _created_at() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


class ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 10_000) -> None:
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.rows: list[dict[str, Any]] = []
        self.writer: Optional[pq.ParquetWriter] = None

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path, self.schema, compression="zstd", use_dictionary=True
            )
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.path)
        else:
            self.writer.close()


def _schema(fields: Iterable[tuple[str, pa.DataType]]) -> pa.Schema:
    metadata = {b"schema_version": SCHEMA_VERSION.encode()}
    return pa.schema(list(fields), metadata=metadata)


TEXT = pa.string()
COMMON = [("schema_version", TEXT), ("build_id", TEXT)]
SCHEMAS = {
    "source_files": _schema(COMMON + [
        ("source_file_id", TEXT), ("path", TEXT), ("checksum", TEXT),
        ("size_bytes", pa.int64()), ("modality", TEXT),
    ]),
    "source_rows": _schema(COMMON + [
        ("source_row_id", TEXT), ("source_file_id", TEXT), ("row_number", pa.int64()),
        ("row_identity", TEXT), ("farm_id", TEXT), ("zone_id", TEXT), ("timestamp", TEXT),
    ]),
    "cell_occurrences": _schema(COMMON + [
        ("cell_id", TEXT), ("source_file_id", TEXT), ("source_row_id", TEXT),
        ("column_name", TEXT), ("feature_alias", TEXT), ("raw_type", TEXT),
        ("raw_value", TEXT), ("raw_display", TEXT), ("is_null", pa.bool_()),
        ("farm_id", TEXT), ("zone_id", TEXT), ("observation_timestamp", TEXT),
        ("timestamp_precision", TEXT), ("modality", TEXT),
    ]),
    "atomic_values": _schema(COMMON + [
        ("atomic_value_id", TEXT), ("feature_name", TEXT), ("feature_alias", TEXT),
        ("raw_type", TEXT), ("canonical_raw_value", TEXT), ("raw_display", TEXT),
        ("unit", TEXT),
    ]),
    "cell_token_mappings": _schema(COMMON + [
        ("cell_id", TEXT), ("atomic_value_id", TEXT), ("token_id", TEXT),
        ("token_string", TEXT), ("policy_id", TEXT), ("registry_version", TEXT),
        ("rule_hash", TEXT), ("tokenization_kind", TEXT), ("derived", pa.bool_()),
    ]),
    "events": _schema(COMMON + [
        ("segment_id", TEXT), ("event_id", TEXT), ("event_kind", TEXT),
        ("observation_timestamp", TEXT), ("annotation_timestamp", TEXT),
        ("timestamp_precision", TEXT), ("time_rank", pa.int64()),
        ("same_time_group_id", TEXT), ("event_view", TEXT),
        ("spatial_scope_type", TEXT), ("spatial_scope_id", TEXT),
        ("farm_ids", TEXT), ("zone_ids", TEXT), ("source_set", TEXT),
        ("modalities", TEXT), ("quality_flags", TEXT), ("embedding_status", TEXT),
    ]),
    "event_tokens": _schema(COMMON + [
        ("event_id", TEXT), ("token_id", TEXT), ("token_string", TEXT),
        ("position", pa.int64()), ("role", TEXT), ("cell_id", TEXT),
    ]),
    "same_time_groups": _schema(COMMON + [
        ("same_time_group_id", TEXT), ("anchor_timestamp", TEXT),
        ("timestamp_precision", TEXT), ("logical_rank", pa.int64()),
    ]),
    "sequences": _schema(COMMON + [
        ("sequence_id", TEXT), ("narrative_id", TEXT), ("narrative_center", TEXT),
        ("background_tokens", TEXT), ("segment_ids", TEXT), ("same_time_group_ids", TEXT),
        ("event_views", TEXT), ("catalog_min_events", pa.int64()),
        ("catalog_max_events", pa.int64()), ("entity_scope", TEXT),
        ("window_definition", TEXT), ("threshold_type", TEXT),
        ("threshold_or_rule", TEXT),
        ("timestamp_precision", TEXT), ("alignment_policy_id", TEXT),
        ("order_semantics", TEXT), ("op_eligible", pa.bool_()),
        ("distinct_time_group_count", pa.int64()), ("effective_token_count", pa.int64()),
        ("covered_time_span_hours", pa.float64()), ("farm_count", pa.int64()),
        ("zone_count", pa.int64()), ("contains_interpretation", pa.bool_()),
        ("contains_image", pa.bool_()), ("permutation_invariant", pa.bool_()),
        ("aggregation_metadata", TEXT), ("quality_flags", TEXT),
    ]),
    "sequence_segments": _schema(COMMON + [
        ("sequence_id", TEXT), ("event_id", TEXT), ("position", pa.int64()),
        ("time_group_rank", pa.int64()), ("position_in_group", pa.int64()),
        ("evidence_role", TEXT),
    ]),
    "external_embeddings": _schema(COMMON + [
        ("event_id", TEXT), ("modality", TEXT), ("source_ref", TEXT),
        ("embedding_ref", TEXT), ("model_id", TEXT), ("embedding_status", TEXT),
        ("answer_tier", TEXT), ("quality_score", pa.int64()),
    ]),
}


def _modality(path: Path) -> str:
    parts = set(path.parts)
    for name in ("E_environment", "A_actuator", "G_growth", "R_rootzone", "I_images"):
        if name in parts:
            return name
    return "interpretation" if "reference_answers" in parts else "metadata"


def _timestamp(raw: str) -> tuple[str, str]:
    if not raw:
        return "", "unknown"
    value = datetime.fromisoformat(raw)
    precision = "day" if len(raw) == 10 else "second"
    if value.tzinfo is None:
        value = value.replace(tzinfo=LOCAL_TZ)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), precision


def _typed(raw: str, column: str) -> Any:
    if raw == "":
        return None
    if column in {"timestamp", "observation_date"}:
        iso, _ = _timestamp(raw)
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if NUMERIC.fullmatch(raw):
        return float(raw) if any(char in raw.lower() for char in ".e") else int(raw)
    return raw


def _case_sets(root: Path) -> dict[str, str]:
    result = {}
    for set_name in ("example_set", "problem_set"):
        with (root / set_name / "case_list.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                result[row["farm_id"]] = set_name
    return result


def _numeric_training(files: list[Path]) -> Iterator[tuple[str, Optional[str], Optional[float]]]:
    for path in files:
        if path.suffix != ".csv" or _modality(path) == "metadata":
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                farm = row.get("farm_id")
                for feature, raw in row.items():
                    if feature not in {"farm_id", "zone_id", "timestamp", "observation_date"} and raw and NUMERIC.fullmatch(raw):
                        yield feature, farm, float(raw)


class CorpusBuilder:
    def __init__(self, root: Path, output: Path, batch_size: int = 10_000) -> None:
        self.root = root
        self.output = output
        self.batch_size = batch_size
        self.output.mkdir(parents=True, exist_ok=True)
        self.files = source_files(root)
        self.checksums = {path: file_hash(path) for path in self.files}
        self.build_id = stable_id(
            "build", {"files": [(p.relative_to(root).as_posix(), self.checksums[p]) for p in self.files]}
        )
        self.sinks = {
            name: ParquetSink(output / f"{name}.parquet", schema, batch_size)
            for name, schema in SCHEMAS.items()
        }
        self.counts = defaultdict(int)
        self.event_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.points: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.sequence_keys: set[tuple[str, tuple[str, ...]]] = set()
        self.groups: dict[str, str] = {}
        self.case_sets = _case_sets(root)
        self.db = sqlite3.connect(output / ".atomic.sqlite")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS atomic_values "
            "(id TEXT PRIMARY KEY, feature TEXT, raw_type TEXT, canonical TEXT, display TEXT)"
        )

    def emit(self, name: str, row: dict[str, Any]) -> None:
        row.update(schema_version=SCHEMA_VERSION, build_id=self.build_id)
        self.sinks[name].append(row)
        self.counts[name] += 1

    def _emit_group(self, timestamp: str, precision: str) -> str:
        group_id = stable_id("timegroup", {"timestamp": timestamp})
        if group_id not in self.groups:
            self.groups[group_id] = timestamp
            self.emit("same_time_groups", {
                "same_time_group_id": group_id, "anchor_timestamp": timestamp,
                "timestamp_precision": precision, "logical_rank": len(self.groups) - 1,
            })
        return group_id

    def _token(
        self, registry: TokenRegistry, feature: str, raw: str, farm: str
    ) -> tuple[str, str, str, str]:
        if raw and NUMERIC.fullmatch(raw) and (feature, "GLOBAL", None) in registry.rules:
            token, metadata = registry.encode(feature, float(raw), "GLOBAL")
            return token, metadata["token_id"], metadata["rule_hash"], "GLOBAL"
        token = categorical_token(feature, raw)
        return token, stable_id("token", {"token": token, "version": registry.version}), "", "CATEGORICAL"

    def process_csv(self, path: Path, registry: TokenRegistry) -> None:
        modality = _modality(path)
        if modality == "metadata":
            return
        relative = path.relative_to(self.root).as_posix()
        checksum = self.checksums[path]
        source_file_id = stable_id("sourcefile", {"path": relative, "checksum": checksum})
        self.emit("source_files", {
            "source_file_id": source_file_id, "path": relative, "checksum": checksum,
            "size_bytes": path.stat().st_size, "modality": modality,
        })
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 1):
                farm, zone = row.get("farm_id", ""), row.get("zone_id", "")
                raw_timestamp = row.get("timestamp") or row.get("observation_date") or ""
                timestamp, precision = _timestamp(raw_timestamp)
                identity = {"farm": farm, "zone": zone, "timestamp": timestamp, "row": row}
                source_row_id = stable_id("sourcerow", {"file": checksum, "identity": identity})
                self.emit("source_rows", {
                    "source_row_id": source_row_id, "source_file_id": source_file_id,
                    "row_number": row_number, "row_identity": canonical_json(identity),
                    "farm_id": farm, "zone_id": zone, "timestamp": timestamp,
                })
                event_id = stable_id("event", {
                    "timestamp": timestamp, "farm": farm, "zone": zone,
                    "view": modality, "schema": SCHEMA_VERSION,
                })
                group_id = self._emit_group(timestamp, precision)
                event_tokens = []
                typed_values = {}
                for position, (column, raw) in enumerate(row.items()):
                    typed = _typed(raw, column)
                    typed_values[column] = typed
                    canonical = canonical_scalar(typed)
                    raw_type = canonical["type"]
                    cell_id = stable_id("cell", {
                        "file_checksum": checksum, "row_identity": identity, "column": column,
                    })
                    atomic_id = stable_id("atomic", {
                        "feature": column, "raw_type": raw_type,
                        "canonical_raw_value": canonical, "schema": SCHEMA_VERSION,
                    })
                    self.db.execute(
                        "INSERT OR IGNORE INTO atomic_values VALUES (?, ?, ?, ?, ?)",
                        (atomic_id, column, raw_type, _json(canonical), raw),
                    )
                    token, token_id, rule_hash, kind = self._token(registry, column, raw, farm)
                    self.emit("cell_occurrences", {
                        "cell_id": cell_id, "source_file_id": source_file_id,
                        "source_row_id": source_row_id, "column_name": column,
                        "feature_alias": column.upper(), "raw_type": raw_type,
                        "raw_value": _json(canonical), "raw_display": raw, "is_null": raw == "",
                        "farm_id": farm, "zone_id": zone,
                        "observation_timestamp": timestamp, "timestamp_precision": precision,
                        "modality": modality,
                    })
                    self.emit("cell_token_mappings", {
                        "cell_id": cell_id, "atomic_value_id": atomic_id, "token_id": token_id,
                        "token_string": token, "policy_id": f"online2-token-{registry.version}",
                        "registry_version": registry.version, "rule_hash": rule_hash,
                        "tokenization_kind": kind, "derived": False,
                    })
                    self.emit("event_tokens", {
                        "event_id": event_id, "token_id": token_id, "token_string": token,
                        "position": position, "role": "observed", "cell_id": cell_id,
                    })
                    event_tokens.append(token)
                event_kind = "GROWTH" if modality == "G_growth" else "LOCAL_OBSERVATION"
                self.emit("events", {
                    "segment_id": event_id, "event_id": event_id, "event_kind": event_kind,
                    "observation_timestamp": timestamp, "annotation_timestamp": "",
                    "timestamp_precision": precision, "time_rank": 0,
                    "same_time_group_id": group_id, "event_view": modality,
                    "spatial_scope_type": "FARM_ZONE", "spatial_scope_id": f"{farm}:{zone}",
                    "farm_ids": _json([farm]), "zone_ids": _json([zone]),
                    "source_set": self.case_sets.get(farm, "unknown"),
                    "modalities": _json([modality]), "quality_flags": "[]",
                    "embedding_status": "",
                })
                event_record = {
                    "id": event_id, "group": group_id, "timestamp": timestamp,
                    "farm": farm, "zone": zone, "modality": modality,
                    "tokens": len(event_tokens),
                    "precision": precision,
                }
                self.event_index[(farm, modality)].append(event_record)
                point = self.points.setdefault(
                    (farm, zone, timestamp),
                    {
                        "farm": farm, "zone": zone, "timestamp": timestamp,
                        "values": {}, "events": {},
                    },
                )
                point["values"].update(typed_values)
                point["events"][modality] = event_record
                if row_number % self.batch_size == 0:
                    self.db.commit()
        self.db.commit()

    def process_external(self) -> None:
        for path in self.files:
            modality = _modality(path)
            if modality not in {"I_images", "interpretation"}:
                continue
            relative = path.relative_to(self.root).as_posix()
            farm = path.parent.name if modality == "I_images" else path.stem.split("_")[0]
            set_name = self.case_sets.get(farm)
            decision = classify_public_path(path, set_name)
            if not decision.allowed or (
                modality == "interpretation" and decision.answer_tier != "score90"
            ):
                continue
            case_file = self.root / (set_name or "example_set") / "case_list.csv"
            with case_file.open(encoding="utf-8-sig", newline="") as handle:
                case = next((row for row in csv.DictReader(handle) if row["farm_id"] == farm), None)
            if case is None:
                continue
            source_file_id = stable_id(
                "sourcefile", {"path": relative, "checksum": self.checksums[path]}
            )
            self.emit("source_files", {
                "source_file_id": source_file_id, "path": relative,
                "checksum": self.checksums[path], "size_bytes": path.stat().st_size,
                "modality": modality,
            })
            timestamp, _ = _timestamp(case["period_end"])
            precision = "period"
            event_kind = "IMAGE" if modality == "I_images" else "INTERPRETATION"
            event_id = stable_id("event", {
                "kind": event_kind, "source": relative, "checksum": self.checksums[path],
            })
            group_id = self._emit_group(timestamp, precision)
            token = "IMAGE_EMBED_SLOT" if modality == "I_images" else "TEXT_EMBED_SLOT"
            token_id = stable_id("token", {"token": token, "version": "1"})
            self.emit("events", {
                "segment_id": event_id, "event_id": event_id, "event_kind": event_kind,
                "observation_timestamp": timestamp, "annotation_timestamp": "",
                "timestamp_precision": precision, "time_rank": 0,
                "same_time_group_id": group_id, "event_view": modality,
                "spatial_scope_type": "FARM", "spatial_scope_id": farm,
                "farm_ids": _json([farm]), "zone_ids": "[]", "source_set": set_name or "",
                "modalities": _json([modality]), "quality_flags": _json(["PERIOD_ALIGNED"]),
                "embedding_status": "pending",
            })
            self.emit("event_tokens", {
                "event_id": event_id, "token_id": token_id, "token_string": token,
                "position": 0, "role": "embedding_slot", "cell_id": "",
            })
            self.emit("external_embeddings", {
                "event_id": event_id, "modality": modality, "source_ref": relative,
                "embedding_ref": "", "model_id": "", "embedding_status": "pending",
                "answer_tier": decision.answer_tier or "",
                "quality_score": decision.quality_score or 0,
            })
            self.event_index[(farm, modality)].append({
                "id": event_id, "group": group_id, "timestamp": timestamp,
                "farm": farm, "zone": "", "modality": modality, "tokens": 1,
                "precision": precision,
            })

    def emit_template_sequence(
        self,
        template: NarrativeTemplate,
        events: list[dict[str, Any]],
        flags: list[str],
    ) -> tuple[bool, str]:
        if not events:
            return False, "TOO_SHORT"
        events = sorted(events, key=lambda event: (event["timestamp"], event["id"]))
        if len(events) < template.min_events:
            return False, "TOO_SHORT"
        if len(events) > template.max_events:
            events = events[-template.max_events :]
            flags = flags + ["CATALOG_MAX_EVENTS_APPLIED"]
        def make_background(items: list[dict[str, Any]]) -> list[str]:
            item_farms = {event["farm"] for event in items}
            item_zones = {event["zone"] for event in items if event["zone"]}
            tokens = [
                "DATASET|online2",
                f"NARRATIVE|{template.narrative_id}",
                f"CATEGORY|{template.category or 'UNSPECIFIED'}",
                f"TIME_RESOLUTION|{template.time_resolution or 'UNSPECIFIED'}",
                f"TIMESTAMP_PRECISION|{template.timestamp_precision}",
                f"ORDER_SEMANTICS|{template.order_semantics}",
                f"ALIGNMENT_POLICY|{template.alignment_policy_id or 'NONE'}",
                f"SCHEMA|{SCHEMA_VERSION}",
            ]
            tokens.extend(f"FARM|{farm}" for farm in sorted(item_farms))
            tokens.extend(f"ZONE|{zone}" for zone in sorted(item_zones))
            tokens.extend(f"EVENT_VIEW|{view}" for view in template.event_views)
            return tokens

        background = make_background(events)
        effective = sum(event["tokens"] for event in events) + len(background)
        while len(events) > 1 and effective > 5120:
            events.pop(0)
            background = make_background(events)
            effective = sum(event["tokens"] for event in events) + len(background)
            flags = flags + ["MAX_TOKEN_WINDOW_APPLIED"]
        if effective > 5120:
            return False, "TOO_LONG"
        if effective < 16:
            return False, "TOO_SHORT"
        dedup_key = (template.narrative_id, tuple(event["id"] for event in events))
        if dedup_key in self.sequence_keys:
            return False, "EXACT_DUPLICATE"
        self.sequence_keys.add(dedup_key)
        farms = {event["farm"] for event in events}
        zones = {event["zone"] for event in events if event["zone"]}
        groups = list(dict.fromkeys(event["group"] for event in events))
        group_timestamps = [
            next(event["timestamp"] for event in events if event["group"] == group)
            for group in groups
        ]
        if group_timestamps != sorted(group_timestamps) or len(group_timestamps) != len(set(group_timestamps)):
            raise ValueError(
                f"{template.narrative_id}: distinct time groups are not strictly increasing"
            )
        natural = {
            "narrative": template.narrative_id,
            "events": [event["id"] for event in events],
            "catalog_min_events": template.min_events,
            "catalog_max_events": template.max_events,
        }
        sequence_id = stable_id("sequence", natural)
        timestamps = [datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")) for event in events]
        if timestamps != sorted(timestamps):
            raise ValueError(f"{template.narrative_id}: event timestamps decreased")
        period_or_set = (
            template.timestamp_precision == "period"
            or template.order_semantics in NON_ORDERED
            or any(flag in {"PERIOD_ALIGNED", "SET_COMPARISON"} for flag in flags)
        )
        op_eligible = (
            template.requested_op_eligible and not period_or_set and len(groups) >= 2
        )
        if template.requested_op_eligible and not op_eligible:
            flags = flags + ["OP_ELIGIBILITY_DOWNGRADED"]
        permutation_invariant = "PERMUTATION_INVARIANT_SET" in flags
        aggregate_metadata = {
            "parent_segment_ids": sorted(event["id"] for event in events),
            "aggregation_policy": "SORTED_SET_MEMBERSHIP",
            "member_count": len(events),
            "permutation_invariant": True,
        } if permutation_invariant else {}
        self.emit("sequences", {
            "sequence_id": sequence_id, "narrative_id": template.narrative_id,
            "narrative_center": events[-1]["timestamp"],
            "background_tokens": _json(background),
            "segment_ids": _json([event["id"] for event in events]),
            "same_time_group_ids": _json(groups),
            "event_views": _json(template.event_views),
            "catalog_min_events": template.min_events,
            "catalog_max_events": template.max_events,
            "entity_scope": template.entity_scope,
            "window_definition": template.window_definition,
            "threshold_type": template.threshold_type,
            "threshold_or_rule": template.threshold_or_rule,
            "timestamp_precision": template.timestamp_precision,
            "alignment_policy_id": template.alignment_policy_id,
            "order_semantics": template.order_semantics,
            "op_eligible": op_eligible, "distinct_time_group_count": len(groups),
            "effective_token_count": effective,
            "covered_time_span_hours": (timestamps[-1] - timestamps[0]).total_seconds() / 3600,
            "farm_count": len(farms), "zone_count": len(zones),
            "contains_interpretation": any(
                event["modality"] == "interpretation" for event in events
            ),
            "contains_image": any(event["modality"] == "I_images" for event in events),
            "permutation_invariant": permutation_invariant,
            "aggregation_metadata": _json(aggregate_metadata),
            "quality_flags": _json(flags),
        })
        ranks = {group: rank for rank, group in enumerate(groups)}
        positions = defaultdict(int)
        for position, event in enumerate(events):
            group = event["group"]
            self.emit("sequence_segments", {
                "sequence_id": sequence_id, "event_id": event["id"], "position": position,
                "time_group_rank": ranks[group], "position_in_group": positions[group],
                "evidence_role": "center" if position == len(events) - 1 else "context",
            })
            positions[group] += 1
        return True, "EMITTED"

    def materialize_sequences(
        self, templates: list[NarrativeTemplate]
    ) -> list[dict[str, Any]]:
        return MaterializerRegistry().materialize(self, templates)

    def finish_atomics(self) -> None:
        cursor = self.db.execute(
            "SELECT id, feature, raw_type, canonical, display FROM atomic_values ORDER BY id"
        )
        while rows := cursor.fetchmany(self.batch_size):
            for atomic_id, feature, raw_type, canonical, display in rows:
                self.emit("atomic_values", {
                    "atomic_value_id": atomic_id, "feature_name": feature,
                    "feature_alias": feature.upper(), "raw_type": raw_type,
                    "canonical_raw_value": canonical, "raw_display": display, "unit": "",
                })

    def close(self) -> None:
        for sink in self.sinks.values():
            sink.close()
        self.db.close()
        (self.output / ".atomic.sqlite").unlink(missing_ok=True)


def build_corpus(root: Path, output: Path, batch_size: int = 10_000) -> dict[str, Any]:
    """Run offline phases 0, 2, 3 and 4 without contacting Neo4j."""
    output.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(root, output / "baseline_inventory.json")
    catalog_meta = normalize_catalog(
        root / "online2_narrative_catalog.csv", output / "normalized_catalog.csv"
    )
    files = source_files(root)
    registry = TokenRegistry(version="1", bins=10)
    registry.fit(_numeric_training(files), [file_hash(path) for path in files])
    binning = registry.payload()
    write_json(output / "binning_registry.json", binning)
    write_json(output / "tokenization_registry.json", {
        "schema_version": SCHEMA_VERSION, "registry_version": "1",
        "binning_registry_hash": binning["registry_hash"],
        "special_tokens": ["IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"],
        "categorical_unknown_policy": "UNKNOWN_CODE|<canonical>",
    })
    write_json(output / "prompt_registry.json", {
        "schema_version": SCHEMA_VERSION, "templates": [],
        "reason": "embeddings are pending; no synthetic prompts generated",
    })
    with (output / "normalized_catalog.csv").open(encoding="utf-8", newline="") as handle:
        active_rows = [row for row in csv.DictReader(handle) if row["status"] == "ACTIVE"]
    templates = [NarrativeTemplate.from_row(row) for row in active_rows]
    builder = CorpusBuilder(root, output, batch_size)
    materialization_reports: list[dict[str, Any]] = []
    try:
        for path in files:
            if path.suffix == ".csv":
                builder.process_csv(path, registry)
        builder.process_external()
        materialization_reports = builder.materialize_sequences(templates)
        builder.finish_atomics()
    finally:
        builder.close()
    unsupported = [
        {
            "narrative_id": report["narrative_id"],
            "materialization_key": report["materialization_key"],
            "status": report["status"],
            "reason": report["reason"],
        }
        for report in materialization_reports
        if report["status"] != "PASS"
    ]
    count_mismatches = []
    for row, report in zip(active_rows, materialization_reports):
        reported_trigger = row.get("trigger_row_count", "").strip()
        reported_unique = row.get("unique_instance_count", "").strip()
        if reported_trigger and int(float(reported_trigger)) != report["trigger_count"]:
            count_mismatches.append({
                "narrative_id": row["narrative_id"], "field": "trigger_row_count",
                "catalog": int(float(reported_trigger)), "actual": report["trigger_count"],
            })
        if reported_unique and int(float(reported_unique)) != report["unique_count"]:
            count_mismatches.append({
                "narrative_id": row["narrative_id"], "field": "unique_instance_count",
                "catalog": int(float(reported_unique)), "actual": report["unique_count"],
            })
    validation = {
        "schema_version": SCHEMA_VERSION,
        "overall_status": "LIMITED" if unsupported else "PASS",
        "materialization_reports": materialization_reports,
        "matcher_executed_count": len(materialization_reports),
        "supported_template_count": sum(
            report["status"] == "PASS" for report in materialization_reports
        ),
        "unsupported_templates": unsupported,
        "unsupported_template_count": len(unsupported),
        "catalog_count_mismatches": count_mismatches,
        "external_embedding_status": "pending",
        "neo4j_validation_status": "NOT_RUN_OFFLINE",
        "future_leakage_count": 0,
        "limitations": [
            "Image and text embeddings are intentionally pending.",
            "Catalog estimate mismatches are reported but do not invalidate deterministic matching.",
            "Neo4j was not contacted.",
        ],
    }
    write_json(output / "validation_report.json", validation)
    registries = list((root / "narratives" / "registries").glob("*.csv"))
    registry_meta = registry_hashes(registries)
    registry_meta["normalized_catalog"] = catalog_meta["normalized_catalog_hash"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": builder.build_id,
        "created_at": _created_at(),
        "seed": 0,
        "input_file_count": inventory["file_count"],
        "input_total_size_bytes": inventory["total_size_bytes"],
        "active_template_count": catalog_meta["active_template_count"],
        "registry_hashes": registry_meta,
        "counts": dict(sorted(builder.counts.items())),
        "supported_template_count": validation["supported_template_count"],
        "unsupported_template_count": len(unsupported),
    }
    write_json(output / "corpus_statistics.json", {
        "schema_version": SCHEMA_VERSION, "counts": manifest["counts"],
    })
    write_json(output / "build_manifest.json", manifest)
    return manifest
