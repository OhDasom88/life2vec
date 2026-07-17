"""Immutable fold execution context — mandatory input for fold-aware CF stages.

Runtime traces are derived from context usage, not ad-hoc fold-id logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


EVAL_CROSSFIT = "crossfit_3fold"
EVAL_OOF_REF = "oof_reference_only"


@dataclass(frozen=True)
class FoldExecutionContext:
    search_fold_ids: Tuple[int, ...]
    holdout_fold_ids: Tuple[int, ...]
    evaluation_mode: str
    case_seen_by_fold: Mapping[int, bool]
    cohort: str  # problem_set | example_set
    case_id: str
    effective_epsilon: float
    oof_fold_id: Optional[int] = None
    configured_epsilon: float = 0.01
    calibration_hash: Optional[str] = None
    risk_metric: str = "abnormal_probability"

    def __post_init__(self) -> None:
        mode = str(self.evaluation_mode)
        cohort = str(self.cohort)
        if cohort not in {"example_set", "problem_set"}:
            raise ValueError(f"unknown cohort={cohort}")
        if mode not in {EVAL_CROSSFIT, EVAL_OOF_REF}:
            raise ValueError(f"unknown evaluation_mode={mode}")
        # Cohort × mode contract (v3 P0)
        if cohort == "example_set" and mode != EVAL_OOF_REF:
            raise ValueError(
                f"InvalidFoldExecutionContext: example_set requires {EVAL_OOF_REF}, got {mode}"
            )
        if cohort == "problem_set" and mode != EVAL_CROSSFIT:
            raise ValueError(
                f"InvalidFoldExecutionContext: problem_set requires {EVAL_CROSSFIT}, got {mode}"
            )
        search = tuple(int(x) for x in self.search_fold_ids)
        holdout = tuple(int(x) for x in self.holdout_fold_ids)
        if mode == EVAL_CROSSFIT:
            if set(search) & set(holdout):
                raise ValueError(f"search/holdout overlap: search={search} holdout={holdout}")
            if not search:
                raise ValueError("crossfit requires non-empty search_fold_ids")
            # problem cases must not claim training seen-ness as all-false via override;
            # allow all-false only when genuinely unseen (problem_set).
        if mode == EVAL_OOF_REF:
            if self.oof_fold_id is None:
                raise ValueError("oof_reference_only requires oof_fold_id")
            if holdout and set(holdout) != {int(self.oof_fold_id)}:
                raise ValueError(
                    f"oof mode holdout must be {{oof_fold_id}}; got {holdout} vs {self.oof_fold_id}"
                )
            if search and set(search) != {int(self.oof_fold_id)}:
                raise ValueError(
                    f"oof mode search must be {{oof_fold_id}}; got {search} vs {self.oof_fold_id}"
                )
            # OOF fold must be unseen; other folds typically seen for example_set
            seen_map = {int(k): bool(v) for k, v in dict(self.case_seen_by_fold).items()}
            if seen_map.get(int(self.oof_fold_id), True):
                raise ValueError(
                    f"oof_fold_id={self.oof_fold_id} must be unseen in case_seen_by_fold"
                )
        object.__setattr__(self, "search_fold_ids", search)
        object.__setattr__(self, "holdout_fold_ids", holdout)
        object.__setattr__(
            self,
            "case_seen_by_fold",
            {int(k): bool(v) for k, v in dict(self.case_seen_by_fold).items()},
        )

    @property
    def cf_validity_allowed(self) -> bool:
        """Example OOF-only must not claim independent cf_valid."""
        return self.independently_evaluable and self.cohort == "problem_set"

    @property
    def independently_evaluable(self) -> bool:
        return self.evaluation_mode == EVAL_CROSSFIT

    def assert_holdout_unused_for_selection(self, used_fold_ids: Sequence[int]) -> None:
        """Raise if any holdout fold was used during attribution/locus/ranking."""
        if not self.independently_evaluable:
            return
        leaked = set(int(x) for x in used_fold_ids) & set(self.holdout_fold_ids)
        if leaked:
            raise RuntimeError(f"holdout fold leakage into selection: {sorted(leaked)}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["case_seen_by_fold"] = {str(k): v for k, v in self.case_seen_by_fold.items()}
        d["independently_evaluable"] = self.independently_evaluable
        return d

    def usage_trace(self, stage: str, **extra: Any) -> Dict[str, Any]:
        row = {
            "stage": stage,
            "case_id": self.case_id,
            "cohort": self.cohort,
            "evaluation_mode": self.evaluation_mode,
            "search_fold_ids": list(self.search_fold_ids),
            "holdout_fold_ids": list(self.holdout_fold_ids),
            "oof_fold_id": self.oof_fold_id,
            "effective_epsilon": self.effective_epsilon,
        }
        row.update(extra)
        return row


def build_fold_execution_context(
    *,
    case_id: str,
    cohort: str,
    evaluation_mode: str,
    case_seen_by_fold: Mapping[int, bool],
    search_fold_ids: Sequence[int],
    holdout_fold_ids: Sequence[int],
    effective_epsilon: float,
    oof_fold_id: Optional[int] = None,
    configured_epsilon: float = 0.01,
    calibration_hash: Optional[str] = None,
    risk_metric: str = "abnormal_probability",
) -> FoldExecutionContext:
    return FoldExecutionContext(
        search_fold_ids=tuple(int(x) for x in search_fold_ids),
        holdout_fold_ids=tuple(int(x) for x in holdout_fold_ids),
        evaluation_mode=str(evaluation_mode),
        case_seen_by_fold=dict(case_seen_by_fold),
        cohort=str(cohort),
        case_id=str(case_id),
        effective_epsilon=float(effective_epsilon),
        oof_fold_id=None if oof_fold_id is None else int(oof_fold_id),
        configured_epsilon=float(configured_epsilon),
        calibration_hash=calibration_hash,
        risk_metric=str(risk_metric),
    )
