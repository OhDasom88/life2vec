"""사람 검토 우선순위 큐 (계획서 §5.3).

optimizer step 중에는 사람이 학습 데이터를 바꾸지 않는다 — 이 모듈은 dataset
version을 고정하기 전, 검토 대상을 7개 기준으로 골라 우선순위를 매기는 순수
함수 계층이다. 실행 트리거나 UI는 ``../ui/data_grounding_curation/``이 담당한다.

7개 기준 중 지금 이 저장소에서 실제로 계산 가능한 신호와, 아직 다른 모듈이
없어서 값을 받아야만 하는 신호를 구분한다(허위로 채우지 않는다):

- 계산 가능(이 모듈이 직접 판정): 모델·규칙·전문가 판정 불일치, 신규·저빈도
  개념(빈도 테이블 필요), 인과·추천 문장 포함, 근거·반례 동시 존재
- 외부 입력 필요(아직 구현 안 된 모듈의 산출물, ``ReviewContext``로 주입):
  예측 영향도 -> `finetune_v03` critic, concept split/merge 후보 -> `../concept_governance/`,
  holdout 유사·권한 불명확 -> split-membership 플러밍(아직 없음)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .schemas import CAUSAL_STATUS_DEFAULT, Narrative, RECOMMENDATION_STATUS_DEFAULT

# 근거 없이 만든 가중치다 — §5.4 평가로 실측 라벨이 쌓이면 재보정 대상이다.
# split 권한 문제(SPLIT_ACCESS_AMBIGUOUS)를 최우선으로 둔 것은 계획서 전체가
# holdout-blind를 fail-closed 원칙(§0)으로 두기 때문이며, 이것만은 재보정
# 이전에도 최우선순위를 유지해야 한다.
_REASON_WEIGHTS: dict[str, float] = {
    "SPLIT_ACCESS_AMBIGUOUS": 3.0,
    "CAUSAL_OR_RECOMMENDATION_PRESENT": 2.0,
    "SUPPORT_AND_CONTRADICTION_BOTH_PRESENT": 2.0,
    "HIGH_PREDICTION_IMPACT": 1.5,
    "MODEL_EXPERT_DISAGREEMENT": 1.5,
    "CONCEPT_SPLIT_MERGE_CANDIDATE": 1.0,
    "LOW_FREQUENCY_CONCEPT": 1.0,
}


@dataclass(frozen=True)
class ReviewReason:
    code: str
    detail: str


@dataclass(frozen=True)
class ReviewContext:
    """다른(아직 없는) 모듈이 채워야 하는 per-narrative 외부 신호.

    필드가 ``None``/``False``면 "신호 없음"이 아니라 "아직 계산 안 됨"으로
    취급한다 — 즉 이 상태에서는 해당 기준이 검토 큐에 절대 걸리지 않는다.
    """

    prediction_impact: float | None = None
    concept_split_merge_candidate: bool = False
    split_ambiguous: bool = False


@dataclass(frozen=True)
class ReviewQueueEntry:
    narrative: Narrative
    reasons: tuple[ReviewReason, ...]
    priority_score: float

    @property
    def requires_review(self) -> bool:
        return bool(self.reasons)


def evaluate_review_reasons(
    narrative: Narrative,
    context: ReviewContext = ReviewContext(),
    *,
    template_frequency: Mapping[str, int] | None = None,
    low_frequency_threshold: int = 50,
    prediction_impact_threshold: float = 0.5,
) -> tuple[ReviewReason, ...]:
    reasons: list[ReviewReason] = []

    # 1. 모델·규칙·전문가 판정 불일치 — 템플릿 저자의 expert_confidence와
    #    §5.2 어댑터의 sequence_recommendation이 서로 반대 방향을 가리키는 경우.
    expert_confidence = str(narrative.confidence.get("template_expert_confidence", ""))
    if expert_confidence == "높음" and narrative.sequence_recommendation in {"REVIEW", "EXCLUDE"}:
        reasons.append(
            ReviewReason(
                "MODEL_EXPERT_DISAGREEMENT",
                f"expert_confidence=높음, sequence_recommendation={narrative.sequence_recommendation}",
            )
        )
    elif expert_confidence == "낮음" and narrative.sequence_recommendation == "INCLUDE":
        reasons.append(
            ReviewReason(
                "MODEL_EXPERT_DISAGREEMENT",
                "expert_confidence=낮음, sequence_recommendation=INCLUDE",
            )
        )

    # 2. 신규 또는 저빈도 개념 — template_frequency를 준 경우에만 판정한다.
    if template_frequency is not None and narrative.supporting_windows:
        template_id = narrative.supporting_windows[0].narrative_template_id
        count = template_frequency.get(template_id, 0)
        if count and count < low_frequency_threshold:
            reasons.append(
                ReviewReason(
                    "LOW_FREQUENCY_CONCEPT",
                    f"template={template_id} count={count} < {low_frequency_threshold}",
                )
            )

    # 3. 인과·추천 문장이 포함된 서사 — 기본값(NOT_ESTABLISHED/NOT_GENERATED)을
    #    벗어난 narrative만 해당. §5.2 어댑터가 만드는 narrative는 항상 기본값을
    #    유지하므로, 이 기준은 causal_status/recommendation_status를 실제로
    #    갱신하는 상위 단계(예: 전문가 승인 이후)에서만 발동한다.
    if (
        narrative.causal_status != CAUSAL_STATUS_DEFAULT
        or narrative.recommendation_status != RECOMMENDATION_STATUS_DEFAULT
    ):
        reasons.append(
            ReviewReason(
                "CAUSAL_OR_RECOMMENDATION_PRESENT",
                f"causal_status={narrative.causal_status}, "
                f"recommendation_status={narrative.recommendation_status}",
            )
        )

    # 4. 근거와 반대 근거가 동시에 강한 사례 — contradicting_windows를 채우는
    #    로직이 아직 없으므로(§5.1 검색은 REJECT 후보를 반례로 승격하지 않는다)
    #    현재는 항상 False로 평가된다. 반례 탐지가 구현되면 자동으로 작동한다.
    if narrative.supporting_windows and narrative.contradicting_windows:
        reasons.append(
            ReviewReason(
                "SUPPORT_AND_CONTRADICTION_BOTH_PRESENT",
                f"{len(narrative.supporting_windows)} supporting vs "
                f"{len(narrative.contradicting_windows)} contradicting",
            )
        )

    # 5. 예측 영향도가 큰 사례 — finetune critic이 아직 없어 외부 주입.
    if context.prediction_impact is not None and context.prediction_impact >= prediction_impact_threshold:
        reasons.append(
            ReviewReason(
                "HIGH_PREDICTION_IMPACT",
                f"prediction_impact={context.prediction_impact:.3f} >= {prediction_impact_threshold}",
            )
        )

    # 6. concept split/merge 후보 — ../concept_governance/가 아직 없어 외부 주입.
    if context.concept_split_merge_candidate:
        reasons.append(
            ReviewReason("CONCEPT_SPLIT_MERGE_CANDIDATE", "flagged by concept_governance")
        )

    # 7. holdout과 유사하나 학습 권한이 불명확한 사례 — split-membership
    #    플러밍이 아직 없어 외부 주입.
    if context.split_ambiguous:
        reasons.append(
            ReviewReason(
                "SPLIT_ACCESS_AMBIGUOUS",
                "holdout-similar with unclear training authorization",
            )
        )

    return tuple(reasons)


def _priority_score(reasons: tuple[ReviewReason, ...]) -> float:
    return sum(_REASON_WEIGHTS.get(reason.code, 1.0) for reason in reasons)


def build_review_queue(
    items: Iterable[tuple[Narrative, ReviewContext]],
    *,
    template_frequency: Mapping[str, int] | None = None,
    low_frequency_threshold: int = 50,
    prediction_impact_threshold: float = 0.5,
) -> list[ReviewQueueEntry]:
    """검토가 필요한 narrative만 걸러 우선순위(내림차순)로 정렬한다."""
    entries: list[ReviewQueueEntry] = []
    for narrative, context in items:
        reasons = evaluate_review_reasons(
            narrative,
            context,
            template_frequency=template_frequency,
            low_frequency_threshold=low_frequency_threshold,
            prediction_impact_threshold=prediction_impact_threshold,
        )
        if reasons:
            entries.append(ReviewQueueEntry(narrative, reasons, _priority_score(reasons)))
    entries.sort(key=lambda entry: entry.priority_score, reverse=True)
    return entries
