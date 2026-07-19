"""CF-1S shared execution context and callback bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


class ProductionContractViolation(RuntimeError):
    """Hard-fail when production entry contracts are violated before candidate work."""


EffectFn = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]
UnaryEditFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]
AtomicTargetFn = Callable[[Mapping[str, Any], Any], Sequence[Any]]
IdentityRunner = Callable[[], Mapping[str, Any]]
ParityRunner = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class CF1SCallbackBundle:
    atomic_target_fn: AtomicTargetFn
    ground_fn: UnaryEditFn
    invert_fn: UnaryEditFn
    gate0_fn: UnaryEditFn
    gate4_fn: UnaryEditFn
    retokenize_fn: UnaryEditFn
    stage_a_reencode_fn: Callable[..., Mapping[str, Any]]
    search_effect_fn: EffectFn
    holdout_effect_fn: EffectFn
    identity_runner: IdentityRunner
    parity_runner: ParityRunner


@dataclass
class CF1SCallCounters:
    holdout_effect_calls: int = 0
    holdout_call_before_candidate_freeze_count: int = 0
    holdout_call_before_search_closure_complete_count: int = 0
    fixture_default_callback_use_count: int = 0
    default_adjacency_target_use_count: int = 0
    simulated_result_count: int = 0
    pre_fallback_effect_used_for_metric_count: int = 0
    search_value_used_as_holdout_count: int = 0
    missing_parent_default_true_count: int = 0
    missing_effect_default_zero_count: int = 0
    parent_candidate_added_after_freeze_count: int = 0
    search_parent_completion_used_for_selection: bool = False
    search_parent_completion_used_for_beam_reranking: bool = False
    search_parent_completion_used_for_candidate_replacement: bool = False


@dataclass
class CF1SExecutionContext:
    case_id: str
    events: Sequence[Mapping[str, Any]]
    edit_policy: Mapping[str, Any]
    search_policy: Mapping[str, Any]
    saliency_policy: Mapping[str, Any]
    acceptance_policy: Mapping[str, Any]
    callbacks: CF1SCallbackBundle
    provenance_mode: str  # FIXTURE | PRODUCTION
    uses_fixture_defaults: bool
    arm: str = "CONTROL_LIKE"
    cutoff_time: Optional[str] = None
    model_effects_evaluated: bool = False
    counters: CF1SCallCounters = field(default_factory=CF1SCallCounters)
    production_contract_checked_before_candidate_generation: bool = False
    candidate_freeze_complete: bool = False
    parent_completion_search_complete: bool = False
    evaluation_closure_locked: bool = False
    selected_manifest_locked: bool = False

    def assert_production_contract(self) -> None:
        self.production_contract_checked_before_candidate_generation = True
        if str(self.provenance_mode).upper() != "PRODUCTION":
            return
        missing = _missing_callbacks(self.callbacks)
        if missing:
            raise ProductionContractViolation(
                f"production callback bundle incomplete: {missing}"
            )
        if self.uses_fixture_defaults:
            raise ProductionContractViolation(
                "production context uses_fixture_defaults=true"
            )
        if self.counters.fixture_default_callback_use_count > 0:
            raise ProductionContractViolation(
                "fixture_default_callback_use_count>0 before candidate generation"
            )


def _missing_callbacks(bundle: CF1SCallbackBundle) -> list[str]:
    required = (
        "atomic_target_fn",
        "ground_fn",
        "invert_fn",
        "gate0_fn",
        "gate4_fn",
        "retokenize_fn",
        "stage_a_reencode_fn",
        "search_effect_fn",
        "holdout_effect_fn",
        "identity_runner",
        "parity_runner",
    )
    missing = []
    for name in required:
        if getattr(bundle, name, None) is None:
            missing.append(name)
    return missing


def wrap_holdout_effect_fn(
    ctx: CF1SExecutionContext,
    holdout_fn: EffectFn,
) -> EffectFn:
    """Count holdout calls and hard-fail if invoked before search closure complete."""

    def _wrapped(edits: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        ctx.counters.holdout_effect_calls += 1
        if not ctx.candidate_freeze_complete:
            ctx.counters.holdout_call_before_candidate_freeze_count += 1
            raise ProductionContractViolation(
                "holdout_effect_fn called before candidate freeze"
            )
        if not ctx.parent_completion_search_complete:
            ctx.counters.holdout_call_before_search_closure_complete_count += 1
            raise ProductionContractViolation(
                "holdout_effect_fn called before parent_completion_search_complete"
            )
        out = dict(holdout_fn(edits))
        if bool(out.get("simulated")):
            ctx.counters.simulated_result_count += 1
            if str(ctx.provenance_mode).upper() == "PRODUCTION":
                raise ProductionContractViolation(
                    "simulated=true result in production holdout scoring"
                )
        return out

    return _wrapped


def wrap_search_effect_fn(
    ctx: CF1SExecutionContext,
    search_fn: EffectFn,
) -> EffectFn:
    """Strip holdout fields from search scoring and reject simulated production results."""

    def _wrapped(edits: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        out = dict(search_fn(edits))
        if bool(out.get("simulated")):
            ctx.counters.simulated_result_count += 1
            if str(ctx.provenance_mode).upper() == "PRODUCTION":
                raise ProductionContractViolation(
                    "simulated=true result in production search scoring"
                )
        # Search results must not carry holdout fields.
        out.pop("delta_risk_holdout_by_fold", None)
        out.pop("delta_risk_holdout_aggregate", None)
        out.pop("atomic_parent_deltas_holdout", None)
        return out

    return _wrapped


def fixture_pass_through(_: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "gate0_pass": True,
        "gate4_pass": True,
        "raw_grounding_status": "EXACT",
        "simulated": True,
    }


def fixture_stage_a_noop(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    return {"ok": True, "simulated": True, "stage_a_output_hash": "fixture"}


def fixture_identity_runner() -> Dict[str, Any]:
    return {
        "risks": [0.1, 0.1, 0.1],
        "logits": [0.0, 0.0, 0.0],
        "forced_risk_delta": 0.0,
        "forced_logit_delta": 0.0,
        "simulated": True,
    }


def fixture_parity_runner(
    _edits: Sequence[Mapping[str, Any]],
    effect: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "parity_audited": False,
        "parity_status": "NOT_EVALUABLE",
        "full_reencode_fallback_used": False,
        "effect_source": "FIXTURE_UNAUDITED",
        "finalized_effect_values": True,
        "finalized_effect": dict(effect),
        "simulated": True,
    }
