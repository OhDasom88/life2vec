"""World-model "Outcome model" role, scoped to bounded (existing CF1S
2-event) edits only -- per README §12.1 role table and the 2026-07-24 scope
decision to NOT build dynamics/plausibility/uncertainty models or pursue RL
this round.

This is a thin, intentionally-not-clever wrapper: the outcome model role is
already fully implemented by `counterfactual/evaluation/diagnosis_critic.py`
(`score_folds`/`ensemble_delta_r`, fold-ensembled risk_before/after). Nothing
new is computed here -- this module exists so `world_model/` has a named,
discoverable entry point matching the role table, instead of silently relying
on a counterfactual/-tree import that a reader wouldn't think to look for
under "world model."

Do NOT add dynamics/plausibility/action-conditioned/uncertainty heads to this
module without first passing `promotion_gate.py` -- see its docstring.
"""

from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.evaluation.diagnosis_critic import (  # noqa: F401
    ensemble_delta_r,
    risk_from_batch,
    score_folds,
)

__all__ = ["score_folds", "ensemble_delta_r", "risk_from_batch"]
