from __future__ import annotations

from typing import Any

import numpy as np


def band_raw_stats(band: np.ndarray, dead_value: int = 0) -> dict[str, float]:
    flat = band.reshape(-1).astype(np.float64)
    valid = flat[flat != dead_value]
    if valid.size == 0:
        return {
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "saturation_ratio": 1.0,
            "valid_ratio": 0.0,
        }
    sat = float((flat >= np.iinfo(np.uint16).max).mean()) if band.dtype == np.uint16 else 0.0
    return {
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "saturation_ratio": sat,
        "valid_ratio": float(valid.size / flat.size),
    }


def fit_global_percentiles(
    bands_iter,
    *,
    low: float,
    high: float,
    dead_value: int = 0,
    n_bands: int = 10,
) -> dict[str, list[float]]:
    """Fit percentile_low/high per band over a scope sample of cubes."""
    lows = [[] for _ in range(n_bands)]
    highs = [[] for _ in range(n_bands)]
    for cube in bands_iter:
        for b in range(n_bands):
            flat = cube[:, :, b].reshape(-1)
            valid = flat[flat != dead_value]
            if valid.size == 0:
                continue
            lows[b].append(float(np.percentile(valid, low)))
            highs[b].append(float(np.percentile(valid, high)))
    return {
        "percentile_low_values": [float(np.median(v)) if v else 0.0 for v in lows],
        "percentile_high_values": [float(np.median(v)) if v else 1.0 for v in highs],
    }


def stretch_band_fixed(
    band: np.ndarray,
    low_v: float,
    high_v: float,
    dead_value: int = 0,
) -> np.ndarray:
    """Convert uint16 band → float32 [0,1] with fixed global percentiles (not per-image)."""
    x = band.astype(np.float32)
    mask = band != dead_value
    denom = max(float(high_v - low_v), 1e-6)
    out = np.zeros_like(x, dtype=np.float32)
    out[mask] = np.clip((x[mask] - low_v) / denom, 0.0, 1.0)
    return out


def band_to_rgb01(band01: np.ndarray) -> np.ndarray:
    """1→3 channel replicate."""
    return np.stack([band01, band01, band01], axis=-1)


def resize_rgb(rgb: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Bicubic-like resize via torch if available, else numpy nearest+blur fallback."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(t, size=size, mode="bicubic", align_corners=False, antialias=True)
    return y.squeeze(0).permute(1, 2, 0).cpu().numpy()


def cube_to_band_images(
    cube: np.ndarray,
    percentiles: dict[str, list[float]],
    contract: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, float]], dict[str, Any]]:
    dead = int(contract.get("dead_pixel_value", 0))
    size = tuple(contract.get("input_size", [224, 224]))
    images = []
    stats = []
    invalid_ratios = []
    for b in range(cube.shape[2]):
        st = band_raw_stats(cube[:, :, b], dead_value=dead)
        stats.append(st)
        invalid_ratios.append(1.0 - st["valid_ratio"])
        low_v = percentiles["percentile_low_values"][b]
        high_v = percentiles["percentile_high_values"][b]
        band01 = stretch_band_fixed(cube[:, :, b], low_v, high_v, dead_value=dead)
        rgb = band_to_rgb01(band01)
        images.append(resize_rgb(rgb, size=size))
    meta = {
        "mean_invalid_ratio": float(np.mean(invalid_ratios)),
        "max_invalid_ratio": float(np.max(invalid_ratios)),
        "n_bands": int(cube.shape[2]),
    }
    return images, stats, meta
