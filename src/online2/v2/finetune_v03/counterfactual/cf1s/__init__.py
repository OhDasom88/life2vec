"""CF-1S saliency-guided multi-event sequence-edit feasibility (non-causal)."""

from .acceptance import evaluate_cohort_acceptance, evaluate_selector_acceptance
from .beam import compose_with_beam, screen_single_event_atomics
from .code_lock import build_cf1s_lock_artifacts
from .edit_policy import EditClass, FeatureEditRecord, load_edit_policy
from .execution_context import (
    CF1SCallbackBundle,
    CF1SExecutionContext,
    ProductionContractViolation,
)
from .fold_effect import required_strict_majority_count, stable_model_sensitivity_valid
from .interaction import classify_interaction, stable_interaction_fields
from .locus_universe import build_common_eligible_universe, canonicalize_raw_mg_locus
from .production_preflight import production_preflight
from .readiness import build_production_readiness
from .selectors import (
    allocate_feature_round_robin,
    rank_loci_for_selector,
)
from .status import overall_cf1s_status, primary_feasibility_from_selectors

__all__ = [
    "EditClass",
    "FeatureEditRecord",
    "load_edit_policy",
    "build_common_eligible_universe",
    "canonicalize_raw_mg_locus",
    "allocate_feature_round_robin",
    "rank_loci_for_selector",
    "classify_interaction",
    "stable_interaction_fields",
    "required_strict_majority_count",
    "stable_model_sensitivity_valid",
    "overall_cf1s_status",
    "primary_feasibility_from_selectors",
    "compose_with_beam",
    "screen_single_event_atomics",
    "build_cf1s_lock_artifacts",
    "CF1SCallbackBundle",
    "CF1SExecutionContext",
    "ProductionContractViolation",
    "production_preflight",
    "build_production_readiness",
    "evaluate_selector_acceptance",
    "evaluate_cohort_acceptance",
]
