"""Development3 production adapters for CF-1S Core (no legacy production_bridge import)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.online2.v2.diagnosis_dataset import view_to_id, zone_to_id
from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03
from src.online2.v2.finetune_v03.risk_saliency import risk_from_binary_logit

from .core_canonical import (
    canonical_json_sha256,
    semantic_critic_input_sha,
    stage_a_output_sha,
)
from .core_contract import (
    CoreContractError,
    derive_thresholds_from_noise,
    sha256_file,
    sha256_json,
)
from .core_edit_proposal import PIPELINE_ID, ForwardKind, ScopeGateReason
from .core_execution import (
    production_apply_edits_and_score,
)
from .core_manifest import FoldForwardTrace
from .core_raw_transaction import ValidatedRawTransaction, reject_unchecked_token_dict
from .core_trace import GlobalForwardTrace, TraceKind

REENCODE_MODE = "FULL_EVENT_COVERAGE_COLD_REBUILD"
MASK_SEMANTICS_CORE = "TRUE_IS_VALID"
RISK_PROBABILITY_TOLERANCE = 1e-6


@dataclass
class ForwardCounters:
    attribution_forward_count: int = 0
    baseline_forward_count: int = 0
    identity_forward_count: int = 0
    candidate_effect_forward_count: int = 0
    nonrequested_fold_forward_count: int = 0
    holdout_attribution_forward_count: int = 0
    holdout_candidate_forward_before_closure_count: int = 0
    fold2_attribution_load_count: int = 0
    fold2_critic_load_before_closure_count: int = 0

    def record(self, kind: ForwardKind, *, scope: str = "search", n: int = 1) -> None:
        if kind == ForwardKind.ATTRIBUTION_FORWARD:
            if scope in ("holdout", "selection_blind_reevaluation", "fold2"):
                self.holdout_attribution_forward_count += n
                raise CoreContractError("selection-blind reevaluation attribution forward forbidden")
            self.attribution_forward_count += n
        elif kind == ForwardKind.BASELINE_FORWARD:
            self.baseline_forward_count += n
        elif kind == ForwardKind.FORCED_IDENTITY_FORWARD:
            self.identity_forward_count += n
        elif kind == ForwardKind.CANDIDATE_EFFECT_FORWARD:
            self.candidate_effect_forward_count += n

    def to_dict(self) -> Dict[str, int]:
        return {
            "attribution_forward_count": self.attribution_forward_count,
            "baseline_forward_count": self.baseline_forward_count,
            "identity_forward_count": self.identity_forward_count,
            "candidate_effect_forward_count": self.candidate_effect_forward_count,
            "nonrequested_fold_forward_count": self.nonrequested_fold_forward_count,
            "holdout_attribution_forward_count": self.holdout_attribution_forward_count,
            "holdout_candidate_forward_before_closure_count": self.holdout_candidate_forward_before_closure_count,
            "fold2_attribution_load_count": self.fold2_attribution_load_count,
            "fold2_critic_load_before_closure_count": self.fold2_critic_load_before_closure_count,
        }


def _sidecar_scalar(row: Mapping[str, Any], key: str, default: Any = 0) -> Any:
    if key in row and row[key] is not None:
        return row[key]
    # common parquet aliases
    aliases = {
        "view": ("view", "view_name"),
        "zone": ("zone", "zone_id"),
        "case_age_hours": ("case_age_hours",),
        "local_hour": ("local_hour",),
    }
    for alt in aliases.get(key, ()):
        if alt in row and row[alt] is not None:
            return row[alt]
    return default


def build_cold_diagnosis_batch(
    cold: Mapping[str, Any],
    *,
    sidecar_by_event_id: Mapping[str, Mapping[str, Any]],
    mask_semantics: str = MASK_SEMANTICS_CORE,
    mask_reference_builder_sha256: Optional[str] = None,
    dino_policy: str = "OFFICIAL_MISSING_MODALITY_ZERO",
    baseline_batch: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build production diagnosis batch from FULL_EVENT_COVERAGE_COLD_REBUILD output.

    Contract:
    - event_mean/event_max: [1, T, H] float32 from current cold rebuild only
    - case_age_hours/view_id/zone_id/local_hour: [1, T]
    - padding_mask: [1, T] bool with TRUE_IS_VALID
    - padding events are not Stage-A inputs; T equals valid event count here
    - completed external diagnosis_batch injection is forbidden upstream
    """
    if mask_semantics not in ("TRUE_IS_VALID", "TRUE_IS_PADDED"):
        raise CoreContractError(f"unknown mask_semantics: {mask_semantics}")
    if mask_semantics != MASK_SEMANTICS_CORE:
        raise CoreContractError("production batch requires TRUE_IS_VALID mask semantics")

    valid_ids = [str(x) for x in cold.get("valid_event_ids") or []]
    event_mean = cold.get("event_mean")
    event_max = cold.get("event_max")
    if event_mean is None or event_max is None:
        raise CoreContractError("cold rebuild missing event_mean/event_max")
    if not torch.is_tensor(event_mean) or not torch.is_tensor(event_max):
        raise CoreContractError("cold event_mean/event_max must be tensors")

    mean = event_mean.detach().float().cpu().contiguous()
    maxv = event_max.detach().float().cpu().contiguous()
    if mean.ndim != 2 or maxv.ndim != 2:
        raise CoreContractError("cold embeddings must be [T, H]")
    t = int(mean.shape[0])
    h = int(mean.shape[1])
    if t != len(valid_ids):
        raise CoreContractError("valid_event_ids length != embedding T")
    if maxv.shape != mean.shape:
        raise CoreContractError("event_mean/event_max shape mismatch")

    ages: List[float] = []
    views: List[int] = []
    zones: List[int] = []
    hours: List[int] = []
    meta_rows: List[Dict[str, Any]] = []
    for eid in valid_ids:
        if eid not in sidecar_by_event_id:
            raise CoreContractError(f"sidecar missing event_id={eid}")
        row = dict(sidecar_by_event_id[eid])
        meta_rows.append(row)
        ages.append(float(_sidecar_scalar(row, "case_age_hours", 0.0) or 0.0))
        views.append(int(view_to_id(str(_sidecar_scalar(row, "view", "UNKNOWN")))))
        zones.append(int(zone_to_id(_sidecar_scalar(row, "zone", 0))))
        hours.append(int(float(_sidecar_scalar(row, "local_hour", 0) or 0)) % 24)

    padding_mask = torch.ones(1, t, dtype=torch.bool)
    batch: Dict[str, Any] = {
        "pipeline_id": PIPELINE_ID,
        "reencode_mode": REENCODE_MODE,
        "event_mean": mean.unsqueeze(0),  # [1, T, H]
        "event_max": maxv.unsqueeze(0),
        "case_age_hours": torch.tensor([ages], dtype=torch.float32),
        "view_id": torch.tensor([views], dtype=torch.long),
        "zone_id": torch.tensor([zones], dtype=torch.long),
        "local_hour": torch.tensor([hours], dtype=torch.long),
        "padding_mask": padding_mask,
        "valid_event_ids": valid_ids,
        "metadata_rows": meta_rows,
        "mask_semantics": mask_semantics,
        "mask_polarity": {
            "core_batch": "TRUE_IS_VALID",
            "critic_adapter_input": "TRUE_IS_VALID",
            "transformer_key_padding_mask": "TRUE_IS_PADDING",
        },
        "valid_event_count": t,
        "padded_event_count": 0,
        "embedding_dim": h,
        "mask_reference_builder_sha256": mask_reference_builder_sha256,
        "dino_policy": dino_policy,
    }

    # Official missing-modality parity: zero DINO + all-false mask when absent.
    if baseline_batch is not None and torch.is_tensor(baseline_batch.get("dino_vec")):
        # Re-align is not possible without event-level DINO; use official missing modality.
        pass
    dino_dim = 768
    if baseline_batch is not None and torch.is_tensor(baseline_batch.get("dino_vec")):
        dino_dim = int(baseline_batch["dino_vec"].shape[-1])
    batch["dino_vec"] = torch.zeros(1, t, dino_dim, dtype=torch.float32)
    batch["dino_mask"] = torch.zeros(1, t, dtype=torch.bool)

    computed_stage_a = stage_a_output_sha(
        event_mean=mean, event_max=maxv, valid_event_ids=valid_ids
    )
    batch["stage_a_output_sha"] = computed_stage_a
    batch["diagnosis_batch_sha"] = semantic_critic_input_sha(batch)
    batch["source_stage_a_output_sha"] = computed_stage_a
    return batch


def assert_batch_matches_cold(
    batch: Mapping[str, Any],
    cold: Mapping[str, Any],
) -> None:
    """Fail-closed stale-batch / field-parity gate before scoring."""
    valid_ids = [str(x) for x in cold.get("valid_event_ids") or []]
    if list(batch.get("valid_event_ids") or []) != valid_ids:
        raise CoreContractError("STALE_DIAGNOSIS_BATCH: valid_event_ids mismatch")
    mean = batch.get("event_mean")
    maxv = batch.get("event_max")
    if not torch.is_tensor(mean) or not torch.is_tensor(maxv):
        raise CoreContractError("STALE_DIAGNOSIS_BATCH: missing embedding tensors")
    cold_mean = cold["event_mean"].detach().float().cpu()
    cold_max = cold["event_max"].detach().float().cpu()
    b_mean = mean.detach().float().cpu()
    b_max = maxv.detach().float().cpu()
    if b_mean.ndim == 3:
        b_mean = b_mean[0]
        b_max = b_max[0]
    if not torch.equal(b_mean, cold_mean) or not torch.equal(b_max, cold_max):
        raise CoreContractError("STALE_DIAGNOSIS_BATCH: embedding tensor mismatch")
    expected = stage_a_output_sha(
        event_mean=cold_mean, event_max=cold_max, valid_event_ids=valid_ids
    )
    if batch.get("stage_a_output_sha") not in (None, expected):
        if batch.get("stage_a_output_sha") != expected:
            raise CoreContractError("STALE_DIAGNOSIS_BATCH: stage_a_output_sha mismatch")


def risk_logit_from_batch(
    model: Any,
    batch: Mapping[str, Any],
    *,
    normal_class_id: int,
    use_binary: bool = True,
) -> Dict[str, Any]:
    """Return abnormal risk and binary logit for Core fold_forward_fn contract."""
    req = ("event_mean", "event_max", "case_age_hours", "view_id", "zone_id", "local_hour", "padding_mask")
    for k in req:
        if k not in batch or not torch.is_tensor(batch[k]):
            raise CoreContractError(f"diagnosis batch missing tensor field: {k}")
    if batch["event_mean"].ndim != 3:
        raise CoreContractError("event_mean must be [B,T,H]")

    if hasattr(model, "encode_events"):
        enc = model.encode_events(
            batch["event_mean"],
            batch["event_max"],
            batch["case_age_hours"],
            batch["view_id"],
            batch["zone_id"],
            batch["local_hour"],
            batch["padding_mask"],
            dino_vec=batch.get("dino_vec"),
            dino_mask=batch.get("dino_mask"),
        )
        h_bin = enc.get("h_binary", enc["h_case"])
        if use_binary and hasattr(model, "binary_head"):
            logit = model.binary_head(h_bin)
            risk = risk_from_binary_logit(logit)
            logit_v = float(logit[0].item()) if logit.ndim >= 1 else float(logit.item())
            risk_v = float(risk[0].item()) if risk.ndim >= 1 else float(risk.item())
            expected = float(risk_from_binary_logit(torch.tensor([logit_v]))[0].item())
            if abs(expected - risk_v) > RISK_PROBABILITY_TOLERANCE:
                raise CoreContractError("RISK_LOGIT_CONTRACT_VIOLATION")
            return {"risk": risk_v, "logit": [logit_v], "abnormal_logit": logit_v}

    from src.online2.v2.finetune_v03.counterfactual.evaluation.diagnosis_critic import (
        risk_from_batch,
    )

    risk_v = float(
        risk_from_batch(model, dict(batch), normal_class_id=normal_class_id, use_binary=use_binary)
    )
    return {"risk": risk_v, "logit": [float("nan")], "abnormal_logit": float("nan")}


def make_production_fold_forward_fn(
    *,
    device: Optional[torch.device] = None,
    normal_class_id: int = 0,
    gpu_fraction: float = 0.4,
    load_registry: Optional["CheckpointLoadRegistry"] = None,
    global_trace: Optional[GlobalForwardTrace] = None,
) -> Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]:
    """fold_forward_fn(checkpoint_path, diagnosis_batch) -> {risk, logit}."""

    def _fn(
        checkpoint_path: Any,
        batch: Mapping[str, Any],
        *,
        _trace_context: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        if batch is None:
            raise CoreContractError("diagnosis_batch required")
        for k in ("event_mean", "event_max", "padding_mask", "case_age_hours", "view_id", "zone_id", "local_hour"):
            if k not in batch:
                raise CoreContractError(f"incomplete diagnosis_batch: missing {k}")
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(gpu_fraction), device=0)
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(checkpoint_path)
        if load_registry is not None:
            load_registry.record_load(ckpt)
        context = dict(_trace_context or {})
        checkpoint_sha = (
            sha256_file(ckpt)
            if ckpt.is_file()
            else canonical_json_sha256({"unresolved_checkpoint_path": str(ckpt)})
        )
        load_invocation = (
            f"critic-load::{context.get('parent_invocation_id', 'standalone')}"
            f"::{checkpoint_sha[:12]}"
        )
        if global_trace is not None:
            global_trace.begin_operation(
                kind=TraceKind.CRITIC_LOAD,
                invocation_id=load_invocation,
                phase=str(context.get("phase") or "UNKNOWN"),
                scope=str(context.get("scope") or "search"),
                allow=True,
                case_id=context.get("case_id"),
                candidate_id=context.get("candidate_id"),
                transaction_sha=context.get("transaction_sha"),
                fold_id=context.get("fold_id"),
                checkpoint_sha=checkpoint_sha,
                parent_invocation_id=context.get("parent_invocation_id"),
            )
            global_trace.start(
                kind=TraceKind.CRITIC_LOAD,
                invocation_id=load_invocation,
                phase=str(context.get("phase") or "UNKNOWN"),
                scope=str(context.get("scope") or "search"),
                case_id=context.get("case_id"),
                candidate_id=context.get("candidate_id"),
                transaction_sha=context.get("transaction_sha"),
                fold_id=context.get("fold_id"),
                checkpoint_sha=checkpoint_sha,
                parent_invocation_id=context.get("parent_invocation_id"),
            )
        try:
            model, meta = load_fold_model_v03(ckpt, device=dev)
        except Exception as exc:
            if global_trace is not None:
                global_trace.fail(
                    kind=TraceKind.CRITIC_LOAD,
                    invocation_id=load_invocation,
                    phase=str(context.get("phase") or "UNKNOWN"),
                    scope=str(context.get("scope") or "search"),
                    failure_code=type(exc).__name__,
                    case_id=context.get("case_id"),
                    candidate_id=context.get("candidate_id"),
                    transaction_sha=context.get("transaction_sha"),
                    fold_id=context.get("fold_id"),
                    checkpoint_sha=checkpoint_sha,
                    parent_invocation_id=context.get("parent_invocation_id"),
                )
            raise
        if global_trace is not None:
            global_trace.complete(
                kind=TraceKind.CRITIC_LOAD,
                invocation_id=load_invocation,
                phase=str(context.get("phase") or "UNKNOWN"),
                scope=str(context.get("scope") or "search"),
                case_id=context.get("case_id"),
                candidate_id=context.get("candidate_id"),
                transaction_sha=context.get("transaction_sha"),
                fold_id=context.get("fold_id"),
                checkpoint_sha=checkpoint_sha,
                parent_invocation_id=context.get("parent_invocation_id"),
            )
        try:
            nid = int(
                meta.get("normal_class_id")
                if meta.get("normal_class_id") is not None
                else normal_class_id
            )
            moved = {
                k: (v.to(dev) if torch.is_tensor(v) else v)
                for k, v in batch.items()
                if k
                in {
                    "event_mean",
                    "event_max",
                    "case_age_hours",
                    "view_id",
                    "zone_id",
                    "local_hour",
                    "padding_mask",
                    "dino_vec",
                    "dino_mask",
                }
            }
            with torch.no_grad():
                out = risk_logit_from_batch(model, moved, normal_class_id=nid, use_binary=True)
            out["checkpoint"] = str(ckpt)
            out["semantic_critic_input_sha"] = semantic_critic_input_sha(batch)
            return out
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    setattr(_fn, "_cf1s_supports_trace_context", True)
    return _fn


@dataclass
class CheckpointLoadRegistry:
    """Trace-derived checkpoint loads for fold isolation proofs."""

    allowed_paths: Optional[set] = None
    closure_frozen: bool = False
    forbidden_fold2_paths: set = field(default_factory=set)
    loads: List[Dict[str, Any]] = field(default_factory=list)

    def record_load(self, path: Path) -> None:
        p = str(Path(path).resolve()) if Path(path).exists() else str(path)
        if self.allowed_paths is not None and p not in self.allowed_paths and str(path) not in self.allowed_paths:
            # also allow unresolved string match
            allowed_str = {str(x) for x in self.allowed_paths}
            if p not in allowed_str and str(path) not in allowed_str:
                raise CoreContractError(f"checkpoint load not in phase allowlist: {path}")
        if not self.closure_frozen and str(path) in self.forbidden_fold2_paths:
            raise CoreContractError(f"fold2 checkpoint load before closure forbidden: {path}")
        self.loads.append({"path": str(path), "closure_frozen": self.closure_frozen})

    def load_count_for(self, paths: Sequence[Any]) -> int:
        wanted = {str(p) for p in paths}
        return sum(1 for row in self.loads if row["path"] in wanted)


def lock_thresholds_from_search_baseline_noise(
    max_search_baseline_pure_repeat_risk_noise: float,
    *,
    max_search_identity_reconstruction_error: float = 0.0,
    minimum_effect_floor: float = 0.001,
) -> Dict[str, Any]:
    """Cohort-level lock from Search runtime noise and fold-scoped identity error."""
    runtime_noise = float(max_search_baseline_pure_repeat_risk_noise)
    identity_err = float(max_search_identity_reconstruction_error)
    ceiling = 0.001
    if runtime_noise > ceiling:
        raise CoreContractError(
            f"runtime_repeat_noise {runtime_noise} exceeds ceiling {ceiling}; NOT_EVALUABLE"
        )
    if identity_err > ceiling:
        raise CoreContractError(
            f"identity_reconstruction_error {identity_err} exceeds ceiling {ceiling}; NOT_EVALUABLE"
        )
    value = max(float(minimum_effect_floor), 3.0 * runtime_noise, 3.0 * identity_err)
    if value > 0.003:
        raise CoreContractError(f"locked threshold {value} exceeds 0.003; lock blocked")
    thr = {
        "locked_min_effect_abs_delta": float(value),
        "locked_min_incremental_abs_gain": float(value),
        "incremental_threshold": float(value),
        "max_search_runtime_repeat_noise": runtime_noise,
        "max_search_identity_reconstruction_error": identity_err,
        "minimum_effect_floor": float(minimum_effect_floor),
    }
    return {
        **thr,
        "threshold_calibration_scope": "DEVELOPMENT3_SEARCH_BASELINE_REPEATS_ONLY",
        "threshold_locked_before_holdout": True,
        "threshold_locked_before_selection_blind_reevaluation": True,
        "holdout_noise_use": "EVALUABILITY_GATE_ONLY",
        "holdout_noise_must_not_change_threshold": True,
        "selection_blind_noise_use": "EVALUABILITY_GATE_ONLY",
        "selection_blind_noise_must_not_change_threshold": True,
        "threshold_policy_sha256": sha256_json(thr),
    }


def evaluate_scope_gate(
    *,
    scope: str,
    identity_pass: bool,
    noise_ceiling_pass: bool,
) -> Dict[str, Any]:
    reasons: List[str] = []
    is_reeval = scope in ("holdout", "selection_blind_reevaluation", "fold2")
    if not identity_pass:
        reasons.append(
            ScopeGateReason.SEARCH_IDENTITY_FAILED.value
            if not is_reeval
            else ScopeGateReason.HOLDOUT_IDENTITY_FAILED.value
        )
    if not noise_ceiling_pass:
        reasons.append(
            ScopeGateReason.SEARCH_NOISE_CEILING_FAILED.value
            if not is_reeval
            else ScopeGateReason.HOLDOUT_NOISE_CEILING_FAILED.value
        )
    allow_candidate_forward = len(reasons) == 0
    if not allow_candidate_forward:
        reasons.append(ScopeGateReason.CANDIDATE_FORWARD_SKIPPED_BY_SCOPE_GATE.value)
    return {
        "scope": scope,
        "allow_candidate_effect_forward": allow_candidate_forward,
        "reasons": reasons,
    }


def run_cold_forward_pipeline(
    *,
    events: list,
    edits: Any = None,
    validated_transaction: Optional[ValidatedRawTransaction] = None,
    reencoder: Any,
    fold_forward_fn: Callable[..., Mapping[str, Any]],
    checkpoint_paths: Sequence[Any],
    fold_ids: Sequence[int],
    requested_fold_ids: Sequence[int],
    sidecar_by_event_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
    baseline_batch_template: Optional[Mapping[str, Any]] = None,
    trace: Optional[FoldForwardTrace] = None,
    global_trace: Optional[GlobalForwardTrace] = None,
    scope: str = "search",
    transaction_mode: str = "CANDIDATE",
    forward_kind: ForwardKind = ForwardKind.CANDIDATE_EFFECT_FORWARD,
    counters: Optional[ForwardCounters] = None,
    batch_size: int = 8,
    case_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    phase: str = "SEARCH_EFFECTS",
    invocation_nonce: str = "0",
    # Explicitly rejected: completed diagnosis_batch injection
    diagnosis_batch: Any = None,
) -> Dict[str, Any]:
    """Single CF1S_COLD_FORWARD_V1 path for baseline/identity/candidates."""
    if diagnosis_batch is not None:
        raise CoreContractError(
            "completed diagnosis_batch injection forbidden; "
            "pass sidecar_by_event_id / baseline_batch_template only"
        )
    tx = validated_transaction
    if tx is None and isinstance(edits, ValidatedRawTransaction):
        tx = edits
    if tx is None:
        reject_unchecked_token_dict(edits)
        raise CoreContractError("validated_transaction required for production pipeline")
    reject_unchecked_token_dict(tx)

    inv_id = (
        f"pipe::{case_id or 'case'}::{candidate_id or tx.canonical_validated_transaction_sha[:12]}"
        f"::{forward_kind.value}::{scope}::{'-'.join(str(x) for x in requested_fold_ids)}"
        f"::{invocation_nonce}"
    )
    if global_trace is not None:
        allowed = global_trace.begin_operation(
            kind=TraceKind.PIPELINE_INVOCATION,
            invocation_id=inv_id,
            phase=phase,
            scope=scope,
            allow=True,
            case_id=case_id,
            candidate_id=candidate_id,
            transaction_sha=tx.canonical_validated_transaction_sha,
            extra={"forward_kind": forward_kind.value},
        )
        if not allowed:
            raise CoreContractError("pipeline invocation DENIED")
        global_trace.start(
            kind=TraceKind.PIPELINE_INVOCATION,
            invocation_id=inv_id,
            phase=phase,
            scope=scope,
            case_id=case_id,
            candidate_id=candidate_id,
            transaction_sha=tx.canonical_validated_transaction_sha,
            extra={"forward_kind": forward_kind.value},
        )

    # Legacy counters retained for compatibility; authoritative counts come from global_trace.
    if counters is not None:
        counters.record(forward_kind, scope=scope)

    try:
        result = production_apply_edits_and_score(
            events=events,
            validated_transaction=tx,
            reencoder=reencoder,
            fold_forward_fn=_wrap_fold_forward_with_trace(
                fold_forward_fn,
                global_trace=global_trace,
                phase=phase,
                scope=scope,
                case_id=case_id,
                candidate_id=candidate_id,
                transaction_sha=tx.canonical_validated_transaction_sha,
                parent_invocation_id=inv_id,
            ),
            checkpoint_paths=checkpoint_paths,
            fold_ids=fold_ids,
            requested_fold_ids=requested_fold_ids,
            sidecar_by_event_id=sidecar_by_event_id,
            baseline_batch_template=baseline_batch_template,
            trace=trace,
            scope=scope,
            transaction_mode=tx.transaction_mode or transaction_mode,
            batch_size=batch_size,
        )
    except Exception as exc:
        if global_trace is not None:
            global_trace.fail(
                kind=TraceKind.PIPELINE_INVOCATION,
                invocation_id=inv_id,
                phase=phase,
                scope=scope,
                failure_code=type(exc).__name__,
                case_id=case_id,
                candidate_id=candidate_id,
                transaction_sha=tx.canonical_validated_transaction_sha,
                extra={"forward_kind": forward_kind.value},
            )
        raise

    if global_trace is not None:
        global_trace.complete(
            kind=TraceKind.PIPELINE_INVOCATION,
            invocation_id=inv_id,
            phase=phase,
            scope=scope,
            case_id=case_id,
            candidate_id=candidate_id,
            transaction_sha=tx.canonical_validated_transaction_sha,
            extra={"forward_kind": forward_kind.value},
        )
    result["pipeline_id"] = PIPELINE_ID
    result["forward_kind"] = forward_kind.value
    result["reencode_mode"] = REENCODE_MODE
    result["trace_invocation_id"] = inv_id
    return result


def _wrap_fold_forward_with_trace(
    fold_forward_fn: Callable[..., Mapping[str, Any]],
    *,
    global_trace: Optional[GlobalForwardTrace],
    phase: str,
    scope: str,
    case_id: Optional[str],
    candidate_id: Optional[str],
    transaction_sha: Optional[str],
    parent_invocation_id: str,
) -> Callable[..., Mapping[str, Any]]:
    if global_trace is None:
        return fold_forward_fn

    def _wrapped(
        path: Any,
        diagnosis_batch: Mapping[str, Any],
        *,
        _cf1s_fold_id: Optional[int] = None,
    ) -> Mapping[str, Any]:
        ckpt = str(path)
        checkpoint_sha = (
            sha256_file(Path(path))
            if Path(path).is_file()
            else canonical_json_sha256({"unresolved_checkpoint_path": ckpt})
        )
        inv = f"ckpt::{parent_invocation_id}::{ckpt}"
        allowed = global_trace.begin_operation(
            kind=TraceKind.CHECKPOINT_FORWARD,
            invocation_id=inv,
            phase=phase,
            scope=scope,
            allow=True,
            case_id=case_id,
            candidate_id=candidate_id,
            transaction_sha=transaction_sha,
            fold_id=_cf1s_fold_id,
            checkpoint_sha=checkpoint_sha,
            parent_invocation_id=parent_invocation_id,
        )
        if not allowed:
            raise CoreContractError("checkpoint forward DENIED")
        global_trace.start(
            kind=TraceKind.CHECKPOINT_FORWARD,
            invocation_id=inv,
            phase=phase,
            scope=scope,
            case_id=case_id,
            candidate_id=candidate_id,
            transaction_sha=transaction_sha,
            fold_id=_cf1s_fold_id,
            checkpoint_sha=checkpoint_sha,
            parent_invocation_id=parent_invocation_id,
        )
        try:
            if getattr(fold_forward_fn, "_cf1s_supports_trace_context", False):
                out = fold_forward_fn(
                    path,
                    diagnosis_batch,
                    _trace_context={
                        "phase": phase,
                        "scope": scope,
                        "case_id": case_id,
                        "candidate_id": candidate_id,
                        "transaction_sha": transaction_sha,
                        "fold_id": _cf1s_fold_id,
                        "parent_invocation_id": inv,
                    },
                )
            else:
                out = fold_forward_fn(path, diagnosis_batch)
        except Exception as exc:
            global_trace.fail(
                kind=TraceKind.CHECKPOINT_FORWARD,
                invocation_id=inv,
                phase=phase,
                scope=scope,
                failure_code=type(exc).__name__,
                case_id=case_id,
                candidate_id=candidate_id,
                transaction_sha=transaction_sha,
                fold_id=_cf1s_fold_id,
                checkpoint_sha=checkpoint_sha,
                parent_invocation_id=parent_invocation_id,
            )
            raise
        global_trace.complete(
            kind=TraceKind.CHECKPOINT_FORWARD,
            invocation_id=inv,
            phase=phase,
            scope=scope,
            case_id=case_id,
            candidate_id=candidate_id,
            transaction_sha=transaction_sha,
            fold_id=_cf1s_fold_id,
            checkpoint_sha=checkpoint_sha,
            parent_invocation_id=parent_invocation_id,
        )
        return out

    setattr(_wrapped, "_cf1s_accepts_fold_id", True)
    return _wrapped


def attribution_policy_lock() -> Dict[str, Any]:
    return {
        "folds": "search_only",
        "token_score": "absolute_ixg",
        "fold_aggregation": "median",
        "event_aggregation": "max_token",
        "top_k": 10,
        "tie_break": ["event_time_epoch", "event_id", "token_index"],
        "selection_blind_attribution_forbidden": True,
        "holdout_attribution_forbidden": True,
    }


RISK_DEFINITION_V1 = {
    "risk_definition": "SIGMOID_ABNORMAL_LOGIT",
    "target_head": "abnormal_binary",
    "probability_transform": "SIGMOID",
    "risk_metric": "abnormal_probability",
    "risk_probability_tolerance": RISK_PROBABILITY_TOLERANCE,
    "compute_dtype": "float32",
}
