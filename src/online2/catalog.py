"""Narrative catalog normalization and registry hashing."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import SCHEMA_VERSION


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def normalize_catalog(source: Path, destination: Path) -> dict[str, Any]:
    """Normalize whitespace, status, booleans and numeric policy fields."""
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("catalog has no header")
        fields = list(reader.fieldnames)
        rows = []
        seen = set()
        for raw in reader:
            row = {key: (value or "").strip() for key, value in raw.items()}
            narrative_id = row.get("narrative_id", "")
            if not narrative_id or narrative_id in seen:
                raise ValueError(f"invalid or duplicate narrative_id: {narrative_id!r}")
            seen.add(narrative_id)
            row["status"] = (row.get("status") or "DISABLED").upper()
            if row["status"] not in {"ACTIVE", "DISABLED", "AUXILIARY", "VALIDATION", "RAG"}:
                raise ValueError(f"invalid status for {narrative_id}: {row['status']}")
            row["op_eligible"] = row.get("op_eligible", "false").lower()
            rows.append(row)
    rows.sort(key=lambda item: item["narrative_id"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "active_template_count": sum(row["status"] == "ACTIVE" for row in rows),
        "normalized_catalog_hash": file_hash(destination),
    }


def registry_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.stem: file_hash(path) for path in sorted(paths)}


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return _hash_bytes(encoded.encode("utf-8"))
