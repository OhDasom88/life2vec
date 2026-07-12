"""Streaming inventory and baseline generation for online2 inputs."""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .canonical import SCHEMA_VERSION, sha256_file
from .catalog import write_json

FARM_FILE = re.compile(r"(?P<farm>F\d+)_z(?P<zone>\d+)\.csv$")


def source_files(root: Path) -> list[Path]:
    """Return declared public raw files in deterministic order."""
    patterns = [
        "data/E_environment/*.csv",
        "data/A_actuator/*.csv",
        "data/G_growth/*.csv",
        "data/R_rootzone/*.csv",
        "data/I_images/**/*",
        "example_set/case_list.csv",
        "problem_set/case_list.csv",
        "answers/reference_answers/score50/*.txt",
        "answers/reference_answers/score70/*.txt",
        "answers/reference_answers/score90/*.txt",
    ]
    files = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _infer(value: str) -> str:
    if value == "":
        return "null"
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return "bool"
    try:
        int(value)
        return "int"
    except ValueError:
        pass
    try:
        float(value)
        return "float"
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}-\d\d-\d\d(?:[ T].*)?", value):
        return "datetime" if len(value) > 10 else "date"
    return "string"


def inspect_csv(path: Path) -> dict[str, Any]:
    rows = 0
    nulls: Counter[str] = Counter()
    types: dict[str, Counter[str]] = {}
    key_counts: Counter[tuple[str, str, str]] = Counter()
    timestamps: list[str] = []
    farms: set[str] = set()
    zones: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        types = {column: Counter() for column in columns}
        for row in reader:
            rows += 1
            for column in columns:
                value = row.get(column, "")
                types[column][_infer(value)] += 1
                if value == "":
                    nulls[column] += 1
            farm = row.get("farm_id", "")
            zone = row.get("zone_id", "")
            timestamp = row.get("timestamp") or row.get("observation_date") or ""
            farms.add(farm)
            zones.add(zone)
            if timestamp:
                timestamps.append(timestamp)
            if farm and zone and timestamp:
                key_counts[(farm, zone, timestamp)] += 1
    return {
        "columns": columns,
        "dtypes": {
            column: types[column].most_common(1)[0][0] if types[column] else "unknown"
            for column in columns
        },
        "row_count": rows,
        "null_count": dict(sorted(nulls.items())),
        "duplicate_entity_time_keys": sum(count - 1 for count in key_counts.values() if count > 1),
        "farm_count": len(farms - {""}),
        "zone_count": len(zones - {""}),
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
    }


def build_inventory(root: Path, output: Path) -> dict[str, Any]:
    files = source_files(root)
    records = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        record: dict[str, Any] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(str(path)),
        }
        if path.suffix.lower() == ".csv":
            record.update(inspect_csv(path))
        records.append(record)
    case_farms: dict[str, set[str]] = {}
    for set_name in ("example_set", "problem_set"):
        case_path = root / set_name / "case_list.csv"
        with case_path.open(encoding="utf-8-sig", newline="") as handle:
            case_farms[set_name] = {row["farm_id"] for row in csv.DictReader(handle)}
    images = [path for path in files if "I_images" in path.parts]
    answer_tiers = Counter(
        next(
            (part for part in path.parts if part in {"score50", "score70", "score90"}),
            "unknown",
        )
        for path in files if "reference_answers" in path.parts
    )
    aligned_images = sum(
        path.parent.name in case_farms["example_set"] | case_farms["problem_set"]
        for path in images
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "root": root.as_posix(),
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "csv_row_count": sum(record.get("row_count", 0) for record in records),
        "duplicate_entity_time_keys": sum(
            record.get("duplicate_entity_time_keys", 0) for record in records
        ),
        "public_pool": {
            "example_farm_count": len(case_farms["example_set"]),
            "problem_farm_count": len(case_farms["problem_set"]),
            "farm_overlap": sorted(case_farms["example_set"] & case_farms["problem_set"]),
            "answer_tier_file_counts": dict(sorted(answer_tiers.items())),
            "image_file_count": len(images),
            "period_alignable_image_count": aligned_images,
            "unaligned_image_count": len(images) - aligned_images,
            "image_alignment_policy": "farm case period; zone unknown",
        },
        "files": records,
        "neo4j": {
            "connection_attempted": False,
            "reason": "baseline is offline by contract",
            "required_environment": ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE"],
            "environment_present": {
                name: bool(os.environ.get(name))
                for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
            },
        },
    }
    write_json(output, payload)
    return payload
