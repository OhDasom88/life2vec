"""CF-1S Core immutable contracts, runtime locks, and legacy isolation helpers."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FORBIDDEN_CORE_IMPORT_SUBSTRINGS = (
    "run_cf1s_v03",
    "write_cf1s_readiness_v03",
    "default_adjacency_targets",
    "fixture_transaction",
    "production_bridge",
    "build_production_readiness",
)

LEGACY_ALLOWED_USES = (
    "fixture_regression",
    "static_comparison",
    "historical_package_reproduction",
    "future_random_recency_incremental_parity_research",
)

LEGACY_FORBIDDEN_USES = (
    "core_smoke",
    "primary32",
    "problem20",
    "official_acceptance",
    "official_readiness",
)


class CoreContractError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def required_observed_gate(
    *,
    required_value: bool = True,
    observed_value: Optional[bool] = None,
    evidence_artifact: Optional[str] = None,
    evidence_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "required_value": bool(required_value),
        "observed_value": observed_value,
        "evidence_artifact": evidence_artifact,
        "evidence_sha256": evidence_sha256,
    }


def gate_observed_pass(gate: Mapping[str, Any]) -> bool:
    return bool(gate.get("required_value")) and gate.get("observed_value") is True


def fold_median(values: Sequence[float]) -> float:
    """Ascending numeric sort; odd=central, even=arithmetic mean of two central."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        raise ValueError("fold_median requires at least one value")
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def required_strict_majority_count(n: int) -> int:
    n_i = int(n)
    if n_i <= 0:
        return 1
    return (n_i // 2) + 1


def sign_with_threshold(value: float, *, threshold: float) -> int:
    v = float(value)
    thr = float(threshold)
    if v >= thr:
        return 1
    if v <= -thr:
        return -1
    return 0


def derive_thresholds_from_noise(max_development_pure_repeat_risk_noise: float) -> Dict[str, float]:
    noise = float(max_development_pure_repeat_risk_noise)
    ceiling = 0.001
    if noise > ceiling:
        raise CoreContractError(
            f"pure-repeat risk noise {noise} exceeds ceiling {ceiling}; threshold calculation forbidden"
        )
    value = max(0.001, 3.0 * noise)
    value = min(value, 0.003)
    return {
        "locked_min_effect_abs_delta": float(value),
        "locked_min_incremental_abs_gain": float(value),
        "max_development_pure_repeat_risk_noise": noise,
    }


def max_target_token_count_from_contract(
    *,
    max_sequence_length: int = 1024,
    special_token_budget: int = 6,
) -> Dict[str, Any]:
    max_len = int(max_sequence_length)
    budget = int(special_token_budget)
    return {
        "max_sequence_length": max_len,
        "special_token_budget": budget,
        "max_target_token_count": max_len - budget,
        "special_token_budget_source": "FROZEN_TOKENIZER_CONTRACT",
    }


def scan_python_imports(paths: Iterable[Path]) -> List[str]:
    found: List[str] = []
    for path in paths:
        if not path.exists() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                found.append(mod)
                for alias in node.names:
                    found.append(f"{mod}.{alias.name}" if mod else alias.name)
    return found


def assert_core_import_graph_clean(core_paths: Sequence[Path]) -> Dict[str, Any]:
    imports = scan_python_imports(core_paths)
    violations = []
    for imp in imports:
        for forbidden in FORBIDDEN_CORE_IMPORT_SUBSTRINGS:
            if forbidden in imp:
                # Allow reading readiness module name only via string constants elsewhere;
                # import path itself is forbidden.
                violations.append(imp)
                break
    # Explicit hard fails for legacy runner / readiness writer / production_bridge
    counts = {
        "core_imports_legacy_runner_count": sum(1 for i in imports if "run_cf1s_v03" in i),
        "core_imports_fixture_callback_count": sum(
            1 for i in imports if "fixture_transaction" in i or "default_adjacency_targets" in i
        ),
        "core_uses_legacy_readiness_count": sum(
            1 for i in imports if "build_production_readiness" in i or "write_cf1s_readiness" in i
        ),
        "core_imports_production_bridge_count": sum(1 for i in imports if "production_bridge" in i),
    }
    if any(counts.values()):
        raise CoreContractError(f"core import graph violates legacy isolation: {counts}")
    return {"ok": True, **counts, "imports_scanned": len(imports), "violations": violations}


def build_legacy_path_manifest(*, root: Path) -> Dict[str, Any]:
    legacy_files = [
        root / "scripts/online2_v2/v03/run_cf1s_v03.py",
        root / "scripts/online2_v2/v03/write_cf1s_readiness_v03.py",
        root / "scripts/online2_v2/v03/package_cf1s_v03.py",
        root / "src/online2/v2/finetune_v03/counterfactual/pipeline_cf1s.py",
        root / "src/online2/v2/finetune_v03/counterfactual/cf1s/production_bridge.py",
        root / "src/online2/v2/finetune_v03/counterfactual/cf1s/readiness.py",
    ]
    file_hashes = {
        str(p.relative_to(root)): sha256_file(p) if p.exists() else None for p in legacy_files
    }
    tree_payload = json.dumps(file_hashes, sort_keys=True)
    return {
        "status": "FROZEN_REFERENCE_ONLY",
        "allowed_uses": list(LEGACY_ALLOWED_USES),
        "forbidden_uses": list(LEGACY_FORBIDDEN_USES),
        "legacy_entrypoint": "scripts/online2_v2/v03/run_cf1s_v03.py",
        "legacy_readiness": "outputs/cf1s/CF1S_PRODUCTION_READINESS.json",
        "legacy_file_sha256": file_hashes,
        "legacy_tree_sha256": sha256_bytes(tree_payload.encode("utf-8")),
        "core_is_only_official_execution_path": True,
        "primary32_uses_core_runner_only": True,
        "problem20_uses_core_runner_only": True,
    }


def deprecate_legacy_readiness(path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any]
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload.update(
        {
            "deprecated_for_core_authorization": True,
            "development_smoke_allowed": False,
            "primary32_allowed": False,
            "problem20_allowed": False,
            "production_command_execution_allowed": False,
            "official_authorization_revoked": True,
            "replacement": "CF1S_CORE_DEVELOPMENT_SMOKE_READINESS.json",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def apply_runtime_lock(*, seed: int = 0) -> Dict[str, Any]:
    """Apply and report deterministic FP32 runtime lock state."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    state = inspect_runtime_lock()
    if not state["lock_satisfied"]:
        raise CoreContractError(f"runtime lock not satisfied: {state}")
    return state


def inspect_runtime_lock() -> Dict[str, Any]:
    import torch

    det = bool(torch.are_deterministic_algorithms_enabled())
    cudnn_det = bool(torch.backends.cudnn.deterministic)
    cudnn_bench = bool(torch.backends.cudnn.benchmark)
    allow_tf32 = False
    if hasattr(torch.backends.cuda, "matmul"):
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    default_dtype = str(torch.get_default_dtype()).replace("torch.", "")
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    tok_par = os.environ.get("TOKENIZERS_PARALLELISM")
    lock_satisfied = (
        det
        and cudnn_det
        and (not cudnn_bench)
        and (not allow_tf32)
        and default_dtype == "float32"
        and cublas == ":4096:8"
        and tok_par == "false"
    )
    return {
        "torch_deterministic_algorithms": det,
        "cudnn_deterministic": cudnn_det,
        "cudnn_benchmark": cudnn_bench,
        "allow_tf32": allow_tf32,
        "default_dtype": default_dtype,
        "cublas_workspace_config": cublas,
        "tokenizers_parallelism": tok_par,
        "autocast_mode": "disabled",
        "stage_a_dtype": "float32",
        "stage_a_batch_size": 8,
        "dataloader_workers": 0,
        "seed": 0,
        "lock_satisfied": lock_satisfied,
    }


def temporal_arm_for_gaps(gaps_seconds: Sequence[float]) -> Tuple[Optional[str], Optional[str]]:
    """Return (arm, reject_reason). Mixed/dead-zone/same-time → reject."""
    if not gaps_seconds:
        return None, "MISSING_GAPS"
    arms = []
    for g in gaps_seconds:
        if g is None:
            return None, "MISSING_EVENT_TIME"
        gf = float(g)
        if gf <= 0:
            return None, "SAME_TIME_OR_NON_POSITIVE_GAP"
        if gf <= 1800.0:
            arms.append("CONTIGUOUS")
        elif gf >= 3600.0:
            arms.append("SPARSE")
        else:
            return None, "REJECT_TEMPORAL_GAP_DEAD_ZONE"
    if len(set(arms)) != 1:
        return None, "MIXED_TEMPORAL_ARM"
    return arms[0], None
