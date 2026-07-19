"""Runtime locks, Stage-A/critic pairing, and deterministic GPU policy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError, sha256_file


DETERMINISM_POLICY = "EXACT_BYTE_DETERMINISTIC"


@dataclass
class RuntimeLock:
    policy: str = DETERMINISM_POLICY
    cublas_workspace_config: str = ":4096:8"
    deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    tf32_disabled: bool = True
    matmul_precision: str = "highest"
    autocast_disabled: bool = True
    compute_dtype: str = "float32"
    torch_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    gpu_model: str = ""
    python_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "cublas_workspace_config": self.cublas_workspace_config,
            "deterministic_algorithms": self.deterministic_algorithms,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "tf32_disabled": self.tf32_disabled,
            "matmul_precision": self.matmul_precision,
            "autocast_disabled": self.autocast_disabled,
            "compute_dtype": self.compute_dtype,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "gpu_model": self.gpu_model,
            "python_version": self.python_version,
        }

    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())


_APPLIED_RUNTIME_LOCK: Optional[RuntimeLock] = None


def apply_exact_byte_deterministic_runtime(*, before_cuda_init: bool = True) -> RuntimeLock:
    """Must run before CUDA initialization when possible. Idempotent once locked."""
    import sys

    global _APPLIED_RUNTIME_LOCK
    if _APPLIED_RUNTIME_LOCK is not None:
        return _APPLIED_RUNTIME_LOCK

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if before_cuda_init and torch.cuda.is_initialized():
        raise CoreContractError(
            "CUDA already initialized; EXACT_BYTE_DETERMINISTIC lock must precede CUDA init"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception as exc:  # pragma: no cover
        raise CoreContractError(f"matmul precision lock failed: {exc}") from exc

    gpu_model = ""
    cuda_version = ""
    cudnn_version = ""
    if torch.cuda.is_available():
        # Reading device name may initialize CUDA; only after lock flags are set.
        gpu_model = str(torch.cuda.get_device_name(0))
        cuda_version = str(getattr(torch.version, "cuda", "") or "")
        cudnn_version = str(torch.backends.cudnn.version())
    lock = RuntimeLock(
        torch_version=str(torch.__version__),
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        gpu_model=gpu_model,
        python_version=sys.version.split()[0],
    )
    _APPLIED_RUNTIME_LOCK = lock
    return lock


def assert_runtime_lock_unchanged(expected: RuntimeLock, observed: RuntimeLock) -> None:
    if expected.sha256() != observed.sha256():
        raise CoreContractError("runtime lock drift")


@dataclass
class StageACriticPairing:
    critic_fold_id: int
    critic_checkpoint_path: str
    critic_checkpoint_sha256: str
    stage_a_checkpoint_path: str
    stage_a_checkpoint_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critic_fold_id": int(self.critic_fold_id),
            "critic_checkpoint_path": self.critic_checkpoint_path,
            "critic_checkpoint_sha256": self.critic_checkpoint_sha256,
            "stage_a_checkpoint_path": self.stage_a_checkpoint_path,
            "stage_a_checkpoint_sha256": self.stage_a_checkpoint_sha256,
        }


def resolve_stage_a_critic_pairings(
    *,
    fold_checkpoint_paths: Mapping[int, Path],
    expected_stage_a_sha_by_fold: Mapping[int, str],
    stage_a_checkpoint_by_sha: Mapping[str, Path],
    default_stage_a_path: Path,
) -> Dict[str, Any]:
    """Prove common Stage-A or lock per-fold pairing."""
    pairings: List[StageACriticPairing] = []
    shas = []
    for fold_id, ckpt in sorted(fold_checkpoint_paths.items()):
        expected = expected_stage_a_sha_by_fold.get(int(fold_id))
        if expected is None:
            # Fall back to default Stage-A file SHA
            expected = sha256_file(Path(default_stage_a_path))
        stage_path = stage_a_checkpoint_by_sha.get(expected, Path(default_stage_a_path))
        if not Path(stage_path).is_file():
            raise CoreContractError(f"Stage-A checkpoint missing for fold {fold_id}: {stage_path}")
        stage_sha = sha256_file(Path(stage_path))
        if stage_sha != expected and expected in stage_a_checkpoint_by_sha:
            raise CoreContractError(f"Stage-A SHA mismatch for fold {fold_id}")
        pairings.append(
            StageACriticPairing(
                critic_fold_id=int(fold_id),
                critic_checkpoint_path=str(ckpt),
                critic_checkpoint_sha256=sha256_file(Path(ckpt)) if Path(ckpt).is_file() else "",
                stage_a_checkpoint_path=str(stage_path),
                stage_a_checkpoint_sha256=stage_sha,
            )
        )
        shas.append(stage_sha)
    common = len(set(shas)) == 1
    return {
        "common_stage_a": common,
        "pairings": [p.to_dict() for p in pairings],
        "pairing_manifest_sha256": canonical_json_sha256([p.to_dict() for p in pairings]),
    }


def load_fold_routing_manifest(path: Path) -> Dict[str, Any]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_id") != "CF1S_DEVELOPMENT3_FOLD_ROUTING_V1":
        raise CoreContractError("unexpected fold routing manifest_id")
    return data


def parse_case_id(case_id: str):
    import pandas as pd

    farm_id, start_s, end_s = str(case_id).split("_", 2)
    return farm_id, pd.Timestamp(start_s), pd.Timestamp(end_s)


def load_case_events(cfg: Mapping[str, Any], case_id: str):
    """Load CaseEvent sequence via pipeline_m2 (not production_bridge)."""
    from ..pipeline_m2 import _load_case_events

    _stage_a, events, _farm_id, _period_start = _load_case_events(dict(cfg), case_id)
    return events


def build_stage_a_reencoder(
    cfg: Mapping[str, Any],
    *,
    case_id: str,
    device: Optional[torch.device] = None,
    global_trace: Optional[Any] = None,
):
    """Construct StageAReencoder for FULL_EVENT_COVERAGE_COLD_REBUILD."""
    from ..pipeline_m2 import _resolve_vocab_and_encoder
    from ..stage_a_reencoder import StageAReencoder
    from .core_trace import TraceKind

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairing_sha = canonical_json_sha256(
        {
            "run_dir": str(cfg.get("run_dir")),
            "vocabulary_path": str(cfg.get("vocabulary_path")),
            "feature_schema_path": str(cfg.get("feature_schema_path")),
            "binning_registry_path": str(cfg.get("binning_registry_path")),
        }
    )
    invocation_id = f"stage-a-load::{case_id}::{pairing_sha[:12]}"
    if global_trace is not None:
        global_trace.begin_operation(
            kind=TraceKind.STAGE_A_LOAD,
            invocation_id=invocation_id,
            phase="SEARCH_BASELINE_IDENTITY",
            scope="search",
            allow=True,
            case_id=case_id,
            stage_a_pairing_sha=pairing_sha,
        )
        global_trace.start(
            kind=TraceKind.STAGE_A_LOAD,
            invocation_id=invocation_id,
            phase="SEARCH_BASELINE_IDENTITY",
            scope="search",
            case_id=case_id,
            stage_a_pairing_sha=pairing_sha,
        )
    try:
        stage_a_mod, encoder, vocab, abspos_ref, _hp = _resolve_vocab_and_encoder(
            dict(cfg), dev
        )
    except Exception as exc:
        if global_trace is not None:
            global_trace.fail(
                kind=TraceKind.STAGE_A_LOAD,
                invocation_id=invocation_id,
                phase="SEARCH_BASELINE_IDENTITY",
                scope="search",
                failure_code=type(exc).__name__,
                case_id=case_id,
                stage_a_pairing_sha=pairing_sha,
            )
        raise
    if global_trace is not None:
        global_trace.complete(
            kind=TraceKind.STAGE_A_LOAD,
            invocation_id=invocation_id,
            phase="SEARCH_BASELINE_IDENTITY",
            scope="search",
            case_id=case_id,
            stage_a_pairing_sha=pairing_sha,
        )
    _farm, period_start, _end = parse_case_id(case_id)
    return StageAReencoder(
        encoder=encoder,
        stage_a_mod=stage_a_mod,
        vocab=vocab,
        abspos_reference=abspos_ref,
        case_t0=period_start,
        device=dev,
    )


def discover_critic_checkpoints(cfg: Mapping[str, Any]) -> List[Path]:
    from ..manifest import discover_fold_ckpts

    run_dir = Path(cfg["run_dir"])
    repeat = int(cfg.get("repeat") or 0)
    return list(discover_fold_ckpts(run_dir, repeat=repeat))


def case_fold_provenance(case_entry: Mapping[str, Any]) -> Dict[str, Any]:
    prov = {
        "training_seen": bool(case_entry.get("training_seen", True)),
        "is_training_holdout": bool(case_entry.get("is_training_holdout", False)),
        "holdout_role": case_entry.get("holdout_role", "SELECTION_BLIND_REEVALUATION"),
        "holdout_blind_to_selection": bool(case_entry.get("holdout_blind_to_selection", True)),
        "scientifically_independent_holdout": bool(
            case_entry.get("scientifically_independent_holdout", False)
        ),
        "search_checkpoint_ids": list(case_entry.get("search_checkpoint_ids") or [0, 1]),
        "selection_checkpoint_ids": list(case_entry.get("selection_checkpoint_ids") or [0, 1]),
        "reevaluation_checkpoint_ids": list(case_entry.get("reevaluation_checkpoint_ids") or [2]),
        "forbidden_checkpoint_ids": list(case_entry.get("forbidden_checkpoint_ids") or []),
    }
    if prov["training_seen"] and prov["scientifically_independent_holdout"]:
        raise CoreContractError(
            "training_seen=true cannot coexist with scientifically_independent_holdout=true"
        )
    if prov["holdout_role"] == "HOLDOUT" and not case_entry.get("legacy_migration"):
        raise CoreContractError("standalone HOLDOUT role forbidden")
    return prov
