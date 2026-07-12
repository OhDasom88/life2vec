"""Canonical scalar encoding and stable identifiers for online2."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "online2-v1"


def _datetime_payload(value: datetime) -> dict[str, Any]:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    utc = value.astimezone(timezone.utc)
    precision = "microsecond" if utc.microsecond else "second"
    timespec = "microseconds" if utc.microsecond else "seconds"
    text = utc.isoformat(timespec=timespec).replace("+00:00", "Z")
    return {
        "type": "datetime",
        "value": text,
        "timezone": "UTC",
        "precision": precision,
    }


def canonical_scalar(value: Any) -> dict[str, Any]:
    """Encode a supported nullable scalar without losing its type or value."""
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": "true" if value else "false"}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "float", "value": "nan"}
        if math.isinf(value):
            return {"type": "float", "value": "inf" if value > 0 else "-inf"}
        return {"type": "float", "value": value.hex()}
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        return {
            "type": "decimal",
            "value": str(value),
            "sign": sign,
            "digits": "".join(str(digit) for digit in digits),
            "exponent": exponent,
        }
    if isinstance(value, datetime):
        return _datetime_payload(value)
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat(), "precision": "day"}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    raise TypeError(f"unsupported canonical scalar: {type(value).__name__}")


def restore_scalar(payload: Mapping[str, Any]) -> Any:
    """Restore a scalar encoded by :func:`canonical_scalar`."""
    kind = payload["type"]
    value = payload.get("value")
    if kind == "null":
        return None
    if kind == "bool":
        return value == "true"
    if kind == "int":
        return int(value)
    if kind == "float":
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
        return float.fromhex(value)
    if kind == "decimal":
        return Decimal((payload["sign"], tuple(map(int, payload["digits"])), payload["exponent"]))
    if kind == "datetime":
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if kind == "date":
        return date.fromisoformat(str(value))
    if kind == "string":
        return str(value)
    if kind == "bytes":
        return base64.b64decode(value)
    raise ValueError(f"unknown canonical type: {kind}")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON for nested natural keys."""
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        if isinstance(item, (str, int, float, bool, Decimal, datetime, date, bytes)) or item is None:
            return canonical_scalar(item)
        raise TypeError(f"unsupported natural-key member: {type(item).__name__}")

    return json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def stable_id(namespace: str, natural_key: Mapping[str, Any] | Sequence[Any]) -> str:
    """Build a namespaced SHA-256 ID from a canonical natural key."""
    body = f"{SCHEMA_VERSION}\n{namespace}\n{canonical_json(natural_key)}"
    return f"{namespace}_{hashlib.sha256(body.encode('utf-8')).hexdigest()}"


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
