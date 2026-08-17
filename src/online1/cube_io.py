from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

ENVI_DTYPE = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}

CUBE_NAME_RE = re.compile(r"(?P<zone>\d+)_(?P<farm>DAT\d+)_(?P<time>\d{6})")


def parse_envi_header(hdr_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    with hdr_path.open(encoding="utf-8") as f:
        lines = f.readlines()
    if not lines or lines[0].strip().upper() != "ENVI":
        raise ValueError(f"Not an ENVI header: {hdr_path}")
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        key = key.lower()
        if key == "wavelength":
            metadata[key] = [float(x) for x in re.findall(r"[\d.]+", value)]
        elif key in {"samples", "lines", "bands", "data type", "header offset"}:
            metadata[key] = int(value)
        else:
            metadata[key] = value.strip("{}").strip()
    required = {"samples", "lines", "bands", "data type", "interleave"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Missing header fields {sorted(missing)} in {hdr_path}")
    return metadata


def load_envi_cube(hdr_path: Path, raw_path: Path | None = None) -> tuple[np.ndarray, dict]:
    if raw_path is None:
        raw_path = hdr_path.with_suffix(".raw")
    metadata = parse_envi_header(hdr_path)
    dtype = ENVI_DTYPE[metadata["data type"]]
    lines = metadata["lines"]
    samples = metadata["samples"]
    bands = metadata["bands"]
    interleave = str(metadata["interleave"]).lower()
    offset = int(metadata.get("header offset", 0))
    expected = lines * samples * bands * np.dtype(dtype).itemsize
    actual = raw_path.stat().st_size - offset
    if actual != expected:
        raise ValueError(f"Raw size mismatch for {raw_path}: expected {expected}, got {actual}")
    data = np.fromfile(raw_path, dtype=dtype, offset=offset)
    if interleave == "bsq":
        cube = data.reshape(bands, lines, samples).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = data.reshape(lines, bands, samples).transpose(0, 2, 1)
    elif interleave == "bip":
        cube = data.reshape(lines, samples, bands)
    else:
        raise ValueError(f"Unsupported interleave: {interleave}")
    return cube, metadata


def parse_cube_sample(sample_dir: Path) -> dict[str, Any]:
    m = CUBE_NAME_RE.match(sample_dir.name)
    if not m:
        raise ValueError(f"Invalid cube folder name: {sample_dir.name}")
    t = m.group("time")
    hh, mm, ss = int(t[:2]), int(t[2:4]), int(t[4:6])
    # Conservative capture_end: folder HHMMSS treated as capture end timestamp.
    capture_end_minute = hh * 60 + mm  # seconds ignored for env-minute alignment; kept in metadata
    return {
        "cube_id": sample_dir.name,
        "cube_zone": int(m.group("zone")),
        "farm": m.group("farm"),
        "dat": int(m.group("farm")[3:]),
        "capture_hhmmss": t,
        "capture_time": f"{t[:2]}:{t[2:4]}:{t[4:6]}",
        "capture_end_minute": capture_end_minute,
        "capture_end_second": ss,
        "capture_end_timestamp_policy": "conservative_folder_hhmmss_as_end",
        "path": str(sample_dir),
    }


def validate_cube_array(cube: np.ndarray, contract: dict[str, Any]) -> dict[str, Any]:
    exp = tuple(contract.get("expected_shape", [1024, 1280, 10]))
    ok_shape = tuple(cube.shape) == exp
    ok_dtype = str(cube.dtype) == str(contract.get("source_dtype", "uint16"))
    return {
        "shape": list(cube.shape),
        "dtype": str(cube.dtype),
        "shape_ok": ok_shape,
        "dtype_ok": ok_dtype,
        "pass": ok_shape and ok_dtype,
    }
