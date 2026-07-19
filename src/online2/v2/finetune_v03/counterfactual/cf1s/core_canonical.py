"""CF1S_CANONICAL_EVIDENCE_V2 serialization and SHA helpers."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

import torch

from .core_contract import CoreContractError

CANONICAL_EVIDENCE_VERSION = "CF1S_CANONICAL_EVIDENCE_V2"


def _normalize_str(value: str) -> str:
    return unicodedata.normalize("NFC", str(value))


def _is_pathlike(obj: Any) -> bool:
    return hasattr(obj, "__fspath__")


def canonical_json_value(obj: Any) -> Any:
    """Normalize a Python object for CF1S_CANONICAL_EVIDENCE_V2 JSON hashing."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, Enum):
        return str(obj.value).upper() if isinstance(obj.value, str) else obj.value
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise CoreContractError("NaN/Inf forbidden in canonical JSON")
        if obj == 0.0:
            return 0.0
        return float(obj)
    if isinstance(obj, str):
        return _normalize_str(obj)
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            raise CoreContractError("naive datetime forbidden; require UTC")
        iso = obj.astimezone(timezone.utc).isoformat(timespec="nanoseconds")
        return iso.replace("+00:00", "Z")
    if _is_pathlike(obj):
        return _normalize_str(str(obj)).replace("\\", "/")
    if isinstance(obj, Mapping):
        return {_normalize_str(str(k)): canonical_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical_json_value(v) for v in obj]
    if torch.is_tensor(obj):
        raise CoreContractError("tensors must use canonical_tensor_sha256, not JSON")
    raise CoreContractError(f"unsupported canonical JSON type: {type(obj)!r}")


def canonical_json_dumps(obj: Any) -> str:
    return json.dumps(
        canonical_json_value(obj),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_json_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(obj).encode("utf-8")).hexdigest()


def canonical_tensor_bytes(t: torch.Tensor) -> bytes:
    """C-order little-endian contiguous CPU bytes of the logical tensor."""
    if not torch.is_tensor(t):
        raise CoreContractError("canonical_tensor_bytes requires a torch.Tensor")
    cpu = t.detach().to("cpu")
    if cpu.dtype == torch.float16:
        cpu = cpu.to(torch.float32)
    cont = cpu.contiguous()
    arr = cont.numpy()
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    elif arr.dtype.byteorder == "=" and not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy(order="C")
    return arr.tobytes()


def canonical_tensor_sha256(t: torch.Tensor, *, logical_name: str) -> str:
    payload = {
        "version": CANONICAL_EVIDENCE_VERSION,
        "logical_name": _normalize_str(logical_name),
        "dtype": str(t.dtype).replace("torch.", ""),
        "shape": list(int(x) for x in t.shape),
        "layout": "C_ORDER_LITTLE_ENDIAN_CONTIGUOUS_CPU",
        "bytes_sha256": hashlib.sha256(canonical_tensor_bytes(t)).hexdigest(),
    }
    return canonical_json_sha256(payload)


def semantic_batch_fields(batch: Mapping[str, Any]) -> dict:
    """Model-input fields only — excludes execution provenance metadata."""
    keep = (
        "event_mean",
        "event_max",
        "case_age_hours",
        "view_id",
        "zone_id",
        "local_hour",
        "padding_mask",
        "dino_vec",
        "dino_mask",
        "valid_event_ids",
        "mask_semantics",
        "stage_a_output_sha",
        "source_diagnosis_batch_sha",
    )
    return {k: batch[k] for k in keep if k in batch}


def semantic_critic_input_sha(batch: Mapping[str, Any]) -> str:
    fields = semantic_batch_fields(batch)
    parts: dict = {
        "version": CANONICAL_EVIDENCE_VERSION,
        "kind": "semantic_critic_input",
    }
    for name, value in fields.items():
        if torch.is_tensor(value):
            parts[name] = canonical_tensor_sha256(value, logical_name=name)
        else:
            parts[name] = canonical_json_value(value)
    return canonical_json_sha256(parts)


def stage_a_output_sha(*, event_mean: torch.Tensor, event_max: torch.Tensor, valid_event_ids) -> str:
    return canonical_json_sha256(
        {
            "version": CANONICAL_EVIDENCE_VERSION,
            "kind": "stage_a_output",
            "valid_event_ids": list(valid_event_ids),
            "event_mean": canonical_tensor_sha256(event_mean, logical_name="event_mean"),
            "event_max": canonical_tensor_sha256(event_max, logical_name="event_max"),
        }
    )
