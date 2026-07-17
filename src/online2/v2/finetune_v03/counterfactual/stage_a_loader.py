"""Import helpers for Stage A script APIs (load_frozen_encoder, windows, pool)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

_STAGE_A: Optional[ModuleType] = None


def load_stage_a_module(
    script_path: Optional[Path] = None,
) -> ModuleType:
    global _STAGE_A
    if _STAGE_A is not None:
        return _STAGE_A
    root = Path(__file__).resolve().parents[5]  # life2vec/
    path = Path(script_path) if script_path else root / "scripts/online2_v2/cache_stage_a_event_embeddings.py"
    name = "cf_stage_a_cache"
    if name in sys.modules:
        _STAGE_A = sys.modules[name]
        return _STAGE_A
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Stage A module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required before exec for @dataclass on Py3.10
    spec.loader.exec_module(mod)
    _STAGE_A = mod
    return mod
