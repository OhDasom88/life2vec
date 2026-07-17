"""Frozen NO_OP round-trip numerical-noise calibration (outside eval cohort).

Material threshold numerical-noise term MUST NOT be fit on Problem 20.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..io_utils import utc_now, write_json


DEFAULT_CONFIGURED_EPSILON = 0.01
DEFAULT_NOOP_NOISE_MULTIPLIER = 10.0


def compute_effective_epsilon(
    *,
    configured_epsilon: float = DEFAULT_CONFIGURED_EPSILON,
    noop_delta_p99: float = 0.0,
    noop_noise_multiplier: float = DEFAULT_NOOP_NOISE_MULTIPLIER,
) -> float:
    noise_term = abs(float(noop_noise_multiplier)) * abs(float(noop_delta_p99))
    return float(max(abs(float(configured_epsilon)), noise_term))


def build_noop_noise_artifact(
    *,
    noop_delta_rs: Sequence[float],
    calibration_cohort: str,
    configured_epsilon: float = DEFAULT_CONFIGURED_EPSILON,
    noop_noise_multiplier: float = DEFAULT_NOOP_NOISE_MULTIPLIER,
    case_ids: Optional[Sequence[str]] = None,
    measurement_method: str = "tokenizer_stage_a_critic_e2e",
) -> Dict[str, Any]:
    """Build frozen calibration payload from NO_OP round-trip |ΔR| samples.

    Preferred measurement_method: tokenizer_stage_a_critic_e2e
    (raw → production tokenizer → Stage A reencode → critic).
    identical_tensor_rescore is rejected in strict mode.
    """
    import numpy as np

    vals = np.asarray([abs(float(x)) for x in noop_delta_rs], dtype=np.float64)
    if vals.size == 0:
        p99 = 0.0
        p50 = 0.0
        vmax = 0.0
    else:
        p99 = float(np.percentile(vals, 99))
        p50 = float(np.percentile(vals, 50))
        vmax = float(vals.max())
    effective = compute_effective_epsilon(
        configured_epsilon=configured_epsilon,
        noop_delta_p99=p99,
        noop_noise_multiplier=noop_noise_multiplier,
    )
    body = {
        "calibration_cohort": str(calibration_cohort),
        "n_samples": int(vals.size),
        "case_ids": [str(x) for x in (case_ids or [])],
        "noop_delta_abs_p50": p50,
        "noop_delta_p99": p99,
        "noop_delta_abs_max": vmax,
        "configured_epsilon": float(configured_epsilon),
        "noop_noise_multiplier": float(noop_noise_multiplier),
        "effective_epsilon": effective,
        "effective_epsilon_frozen": True,
        "definition": "noop_roundtrip_numerical_noise",
        "measurement_method": measurement_method,
        "created_at": utc_now(),
    }
    blob = json.dumps(
        {k: body[k] for k in sorted(body) if k not in {"created_at", "calibration_hash"}},
        sort_keys=True,
        default=str,
    )
    body["calibration_hash"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return body


def write_noop_noise_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(artifact))
    return path


MIN_STRICT_CALIBRATION_SAMPLES = 20
FORBIDDEN_STRICT_COHORTS = {"deterministic_zero_bootstrap"}
FORBIDDEN_STRICT_METHODS = {"identical_tensor_rescore", "clone_batch_rescore"}
REQUIRED_STRICT_METHODS = {"tokenizer_stage_a_critic_e2e"}


def load_frozen_epsilon(
    artifact_path: Path,
    *,
    configured_epsilon: Optional[float] = None,
    require_frozen: bool = True,
    strict: bool = False,
    min_samples: int = MIN_STRICT_CALIBRATION_SAMPLES,
) -> Dict[str, Any]:
    """Load frozen effective epsilon; never recompute from eval data here."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"NO_OP calibration artifact missing: {path}. "
            "Generate from example_oof / dedicated calibration cohort before Problem 20 eval."
        )
    art = json.loads(path.read_text())
    if require_frozen and not art.get("effective_epsilon_frozen", True):
        raise RuntimeError(f"calibration artifact not frozen: {path}")
    cohort = str(art.get("calibration_cohort") or "")
    n_samples = int(art.get("n_samples") or 0)
    case_ids = list(art.get("case_ids") or [])
    method = str(art.get("measurement_method") or "identical_tensor_rescore")
    if strict:
        if cohort in FORBIDDEN_STRICT_COHORTS:
            raise RuntimeError(
                f"strict mode rejects calibration_cohort={cohort}; "
                "require dedicated_calibration_or_example_oof with real round-trip samples"
            )
        if method in FORBIDDEN_STRICT_METHODS or method not in REQUIRED_STRICT_METHODS:
            raise RuntimeError(
                f"strict mode rejects measurement_method={method}; "
                f"require one of {sorted(REQUIRED_STRICT_METHODS)}"
            )
        if n_samples < int(min_samples):
            raise RuntimeError(
                f"strict calibration requires n_samples>={min_samples}, got {n_samples}"
            )
        if not case_ids:
            raise RuntimeError("strict calibration requires non-empty case_ids")
    cfg_eps = float(
        configured_epsilon
        if configured_epsilon is not None
        else art.get("configured_epsilon", DEFAULT_CONFIGURED_EPSILON)
    )
    if "effective_epsilon" in art and configured_epsilon is None:
        effective = float(art["effective_epsilon"])
    else:
        effective = compute_effective_epsilon(
            configured_epsilon=cfg_eps,
            noop_delta_p99=float(art.get("noop_delta_p99") or 0.0),
            noop_noise_multiplier=float(
                art.get("noop_noise_multiplier") or DEFAULT_NOOP_NOISE_MULTIPLIER
            ),
        )
    return {
        "calibration_cohort": cohort,
        "noop_delta_p99": float(art.get("noop_delta_p99") or 0.0),
        "configured_epsilon": cfg_eps,
        "effective_epsilon": effective,
        "calibration_hash": art.get("calibration_hash"),
        "artifact_path": str(path),
        "effective_epsilon_frozen": True,
        "n_samples": n_samples,
        "case_ids": case_ids,
        "measurement_method": method,
        "strict_ok": bool(
            cohort not in FORBIDDEN_STRICT_COHORTS
            and method in REQUIRED_STRICT_METHODS
            and n_samples >= int(min_samples)
            and bool(case_ids)
        ),
    }


def ensure_default_calibration_artifact(
    path: Path,
    *,
    configured_epsilon: float = DEFAULT_CONFIGURED_EPSILON,
    noop_noise_multiplier: float = DEFAULT_NOOP_NOISE_MULTIPLIER,
    allow_zero_bootstrap: bool = False,
) -> Dict[str, Any]:
    """Dev-only zero bootstrap. Strict smoke must set allow_zero_bootstrap=False."""
    path = Path(path)
    if path.exists():
        return load_frozen_epsilon(
            path, configured_epsilon=configured_epsilon, strict=False
        )
    if not allow_zero_bootstrap:
        raise FileNotFoundError(
            f"NO_OP calibration missing at {path} and zero-bootstrap is forbidden in strict mode"
        )
    art = build_noop_noise_artifact(
        noop_delta_rs=[0.0],
        calibration_cohort="deterministic_zero_bootstrap",
        configured_epsilon=configured_epsilon,
        noop_noise_multiplier=noop_noise_multiplier,
        case_ids=[],
        measurement_method="identical_tensor_rescore",
    )
    write_noop_noise_artifact(path, art)
    return load_frozen_epsilon(path, configured_epsilon=configured_epsilon, strict=False)
