"""§12.3 world-model -> RL-environment promotion gate, enforced in code.

The plan (`README.md` §12.3, mirrored in `../edit_policy_rl/README.md`) lists
7 criteria that must ALL pass before `world_model/` may be used as an RL
environment rather than a bounded-edit auxiliary scorer. As of this module's
creation (2026-07-24), NONE of the models that would produce real evidence
for these criteria exist (dynamics/plausibility/action-conditioned/
uncertainty are explicitly out of scope this round -- see README "범위 결정
(2026-07-24)"). Rather than leave that as a README sentence someone can miss,
this module makes it structurally impossible to construct a "promoted"
decision without evidence for every criterion: `PromotionDecision.__post_init__`
raises if `promoted=True` while any criterion is unpassed, and
`current_known_status()` returns the actual, currently-all-blocked gate.

Fail-closed by construction: a criterion with no evidence is FAILED, not
skipped or assumed-pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class PromotionCriterion:
    key: str
    description: str
    passed: bool
    evidence_note: str


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    criteria: Tuple[PromotionCriterion, ...]

    def __post_init__(self) -> None:
        unpassed = [c.key for c in self.criteria if not c.passed]
        if self.promoted and unpassed:
            raise ValueError(
                f"promoted=True but criteria unpassed: {unpassed} -- fail-closed violation, "
                "a PromotionDecision cannot claim promotion without every criterion passing"
            )

    @property
    def blocking_reasons(self) -> Tuple[str, ...]:
        return tuple(f"{c.key}: {c.evidence_note}" for c in self.criteria if not c.passed)


# §12.3 criteria, in the order the plan lists them.
CRITERIA_SPEC: Tuple[Tuple[str, str], ...] = (
    (
        "next_state_beats_baseline",
        "next-state 예측이 persistence/seasonal baseline보다 우수함이 실측 보고됨",
    ),
    (
        "multistep_rollout_error_reported",
        "multi-step rollout horizon별 오류가 실측 보고됨",
    ),
    (
        "action_conditioning_learns_real_effect",
        "action conditioning이 실제 반응 차이를 학습함이 실측 확인됨",
    ),
    (
        "uncertainty_calibration_passes",
        "uncertainty calibration 검증을 통과함",
    ),
    (
        "ood_detection_works",
        "OOD action/state 탐지가 실측 확인됨",
    ),
    (
        "rollout_physically_consistent",
        "실제 trajectory와 rollout의 물리·시간 일관성이 확인됨",
    ),
    (
        "holdout_blind_eval_passed",
        "development acceptance 고정 후 holdout(Validation20) blind 평가를 통과함",
    ),
)


def evaluate_promotion_gate(
    evidence: Mapping[str, Optional[bool]],
    evidence_notes: Mapping[str, str],
) -> PromotionDecision:
    """`evidence[key] is True` is the ONLY way a criterion passes. Missing key,
    None, or False all fail it -- this is deliberate (fail-closed)."""
    criteria = tuple(
        PromotionCriterion(
            key=key,
            description=desc,
            passed=bool(evidence.get(key) is True),
            evidence_note=evidence_notes.get(key, "no evidence provided"),
        )
        for key, desc in CRITERIA_SPEC
    )
    return PromotionDecision(promoted=all(c.passed for c in criteria), criteria=criteria)


def current_known_status() -> PromotionDecision:
    """The ACTUAL current gate state, per the 2026-07-24 codebase audit.

    Every criterion is False because the models that would produce evidence
    for it were explicitly not built this round (see README §12.1 role
    table -- plausibility/dynamics/action-conditioned/uncertainty are all
    "신규", none implemented). The one criterion with real measured evidence
    (`holdout_blind_eval_passed`-adjacent) is the CF1S 55-case validation:
    44/55 reached CONSTRUCTIBLE_SELECTED but ALL 44 were
    CONTROL_WITHIN_LOCKED_THRESHOLD (0 cases exceeded the locked threshold
    under 2-event edits, `outputs/cf1s_core/ALL55_CASE_FINDINGS.json`,
    confirmed still current as of 2026-07-24, no later re-measurement found).
    That's evidence the ACTION-RESPONSE SIGNAL itself is sparse in the
    current bounded edit range, not just that a model is unbuilt -- it's the
    reason §14 recommends action-coverage expansion (3+ event edits) before
    investing further in dynamics/plausibility models, which this round
    explicitly deferred.
    """
    no_evidence = "model not built this round (2026-07-24 scope: bounded-edit scorer only, no RL)"
    evidence = {key: False for key, _ in CRITERIA_SPEC}
    notes = {key: no_evidence for key, _ in CRITERIA_SPEC}
    notes["holdout_blind_eval_passed"] = (
        "CF1S 55-case validation ran (Dev3 3 + Primary32 32 + Validation20 20), but the outcome "
        "was 0/44 CONSTRUCTIBLE_SELECTED cases exceeding the locked effect threshold -- signal "
        "sparse in the current 2-event edit range, not a pass"
    )
    return evaluate_promotion_gate(evidence, notes)
