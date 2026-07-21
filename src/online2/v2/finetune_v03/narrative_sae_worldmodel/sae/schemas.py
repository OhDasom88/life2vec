"""§9.1/§9.7 SAE feature 상태 머신과 인과 증거 등급 + §4.1 `SAEFeatureRef`.

SAE latent는 자동으로 monosemantic feature로 확정하지 않는다 — 다음 상태를
코드로 강제한다:

    SAE_LATENT → CANDIDATE_SEMANTIC_FEATURE → EVALUATED_SEMANTIC_FEATURE
              → INTERVENTION_SUPPORTED_FEATURE → CIRCUIT_SUPPORTED_FEATURE

그리고 인과 증거 등급(E0~E5)에서 **E3(개입 실험) 미만은 인과적 특징이라고
부르지 않는다** — 이걸 문서로만 적어두지 않고 ``assert_causal_claim_allowed``가
실제로 막는다. CF1S의 `disposition_profiles.py`와 같은 설계 원칙(자기선언
금지, 재계산 기반 판정 — 저장된 라벨을 그대로 믿지 않는다)을 공유한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureState(str, Enum):
    SAE_LATENT = "SAE_LATENT"
    CANDIDATE_SEMANTIC_FEATURE = "CANDIDATE_SEMANTIC_FEATURE"
    EVALUATED_SEMANTIC_FEATURE = "EVALUATED_SEMANTIC_FEATURE"
    INTERVENTION_SUPPORTED_FEATURE = "INTERVENTION_SUPPORTED_FEATURE"
    CIRCUIT_SUPPORTED_FEATURE = "CIRCUIT_SUPPORTED_FEATURE"


# 상태 승격 순서 - 건너뛰기 금지(예: SAE_LATENT에서 바로 INTERVENTION_SUPPORTED로 못 감).
_STATE_ORDER = [
    FeatureState.SAE_LATENT,
    FeatureState.CANDIDATE_SEMANTIC_FEATURE,
    FeatureState.EVALUATED_SEMANTIC_FEATURE,
    FeatureState.INTERVENTION_SUPPORTED_FEATURE,
    FeatureState.CIRCUIT_SUPPORTED_FEATURE,
]
_STATE_RANK = {state: rank for rank, state in enumerate(_STATE_ORDER)}


def assert_valid_promotion(current: FeatureState, next_state: FeatureState) -> None:
    """한 단계씩만 승격 가능. 강등(재평가로 신뢰도가 떨어짐)은 언제나 허용한다."""
    current_rank = _STATE_RANK[current]
    next_rank = _STATE_RANK[next_state]
    if next_rank > current_rank + 1:
        raise ValueError(
            f"cannot promote {current.value} -> {next_state.value}: "
            f"must pass through {_STATE_ORDER[current_rank + 1].value} first"
        )


class CausalGrade(str, Enum):
    E0_OBSERVED = "E0_OBSERVED"
    E1_ASSOCIATED = "E1_ASSOCIATED"
    E2_PREDICTIVE = "E2_PREDICTIVE"
    E3_INTERVENTION = "E3_INTERVENTION"
    E4_MEDIATED = "E4_MEDIATED"
    E5_CIRCUIT = "E5_CIRCUIT"


_CAUSAL_GRADE_ORDER = [
    CausalGrade.E0_OBSERVED,
    CausalGrade.E1_ASSOCIATED,
    CausalGrade.E2_PREDICTIVE,
    CausalGrade.E3_INTERVENTION,
    CausalGrade.E4_MEDIATED,
    CausalGrade.E5_CIRCUIT,
]
_CAUSAL_GRADE_RANK = {grade: rank for rank, grade in enumerate(_CAUSAL_GRADE_ORDER)}
_MINIMUM_CAUSAL_GRADE = CausalGrade.E3_INTERVENTION


def assert_causal_claim_allowed(grade: CausalGrade) -> None:
    """E3 미만이면 "인과적"이라는 단어를 쓰는 호출부를 여기서 막는다.

    SAE Top-K 활성값만으로(E0/E1/E2) circuit tracing이나 인과 개입 효과를
    주장하는 코드 경로가 있으면 이 함수 호출에서 예외가 난다 — 계획서 §9.7
    ("E3 미만을 인과적 특징이라고 부르지 않으며")을 조용한 문서 문구가 아니라
    실행 시점의 가드로 만든 것.
    """
    if _CAUSAL_GRADE_RANK[grade] < _CAUSAL_GRADE_RANK[_MINIMUM_CAUSAL_GRADE]:
        raise ValueError(
            f"{grade.value} is below {_MINIMUM_CAUSAL_GRADE.value} — "
            "cannot make a causal claim without intervention evidence (계획서 §9.7)"
        )


@dataclass(frozen=True)
class SAEFeatureRef:
    """계획서 §4.1. SAE version·latent ID·activation·validation state.

    ``state``는 저장된 라벨이 아니라 항상 재계산돼야 한다(disposition_profiles.py와
    동일 원칙) — 이 dataclass는 그 재계산 결과를 담는 그릇일 뿐, 이 객체를
    직접 수정해서 상태를 "승급"시키는 코드를 만들지 않는다.
    """

    sae_version: str  # "SAE_PRETRAIN"|"SAE_FINETUNE"|"SAE_WORLD"|"SAE_POLICY"
    latent_id: int
    state: FeatureState
    causal_grade: CausalGrade = CausalGrade.E0_OBSERVED
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latent_id < 0:
            raise ValueError("latent_id must not be negative")
        if (
            self.state is FeatureState.INTERVENTION_SUPPORTED_FEATURE
            and _CAUSAL_GRADE_RANK[self.causal_grade] < _CAUSAL_GRADE_RANK[_MINIMUM_CAUSAL_GRADE]
        ):
            raise ValueError(
                "INTERVENTION_SUPPORTED_FEATURE state requires causal_grade >= E3_INTERVENTION"
            )
