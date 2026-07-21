"""§8.2 modality별 ablation.

event/token 단위 attribution은 이미 ``counterfactual/attribution/``
(``token_ixg_v03.py``의 gradient 기반 IxG, ``aggregation.py``/``fold_consensus.py``의
역할 가중 집계·fold 간 consensus)에 구현돼 있다 — diagnosis 목적(binary
abnormal logit)에 한정되지만 재구현하지 않는다. grep으로 확인한 결과
modality 단위 ablation은 저장소 어디에도 없었다 — 그것만 여기서 만든다.

모델을 modality 제외하고 다시 순전파하는 것 자체는 하지 않는다 — 그건 실제
학습된 멀티모달 모델이 있어야 하는 일인데(``../multimodal_pretrain/``가
아직 손실 함수만 있고 학습 루프에 연결되지 않았다), 아직 없다. 대신 이미
계산된 두 예측값(modality 포함 vs 제외)을 받아 delta를 계산·집계하는 순수
함수만 제공한다 — 실제 순전파는 호출자(평가 루프) 책임이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModalityAblationRecord:
    modality: str
    sequence_id: str
    prediction_with_modality: float
    prediction_without_modality: float

    @property
    def delta(self) -> float:
        return self.prediction_with_modality - self.prediction_without_modality

    @property
    def absolute_delta(self) -> float:
        return abs(self.delta)


def summarize_modality_ablation(
    records: Sequence[ModalityAblationRecord],
) -> dict[str, dict[str, float]]:
    """modality별 delta 통계(n/mean_delta/mean_absolute_delta) — 어느 modality가
    실제 예측에 기여하는지 순위 매기는 데 쓴다.
    """
    by_modality: dict[str, list[ModalityAblationRecord]] = {}
    for record in records:
        by_modality.setdefault(record.modality, []).append(record)

    summary: dict[str, dict[str, float]] = {}
    for modality, items in by_modality.items():
        deltas = [item.delta for item in items]
        absolute_deltas = [item.absolute_delta for item in items]
        summary[modality] = {
            "n": float(len(items)),
            "mean_delta": sum(deltas) / len(deltas),
            "mean_absolute_delta": sum(absolute_deltas) / len(absolute_deltas),
        }
    return summary


def find_dead_modalities(
    summary: Mapping[str, Mapping[str, float]], *, threshold: float = 1e-6
) -> list[str]:
    """제외해도 예측이 거의 안 바뀌는 modality(§7.4 "dead modality" 로깅 항목과 연결).

    ``mean_absolute_delta``가 ``threshold`` 미만이면 사실상 아무 기여도 안
    하고 있다는 뜻이다 — modality가 입력에는 있지만 모델이 무시하고 있을
    가능성(embedding이 항상 0이거나, 손실 가중치가 너무 낮거나 등).
    """
    return sorted(
        modality
        for modality, stats in summary.items()
        if stats["mean_absolute_delta"] < threshold
    )


def rank_modalities_by_contribution(
    summary: Mapping[str, Mapping[str, float]],
) -> list[tuple[str, float]]:
    """``mean_absolute_delta`` 내림차순 (modality, mean_absolute_delta) 목록."""
    return sorted(
        ((modality, stats["mean_absolute_delta"]) for modality, stats in summary.items()),
        key=lambda item: item[1],
        reverse=True,
    )
