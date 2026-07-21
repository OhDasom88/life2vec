"""L_total = λ1·L_MLM + λ2·L_SOP + λ3·L_time + λ4·L_time-text + λ5·L_cross-modal (계획서 §7.2).

``L_MLM``/``L_SOP``는 여기서 계산하지 않는다 — ``scripts/online2_v2/run_v2_pretrain_loop.py``가
``src/tasks/grouped_mlm.py``의 ``GroupedMLM``으로 이미 계산하고 있는 실제 값을
**입력으로만** 받는다(재계산 금지, 중복 학습 로직 방지). 이 모듈은 그 두 값과
``losses.py``의 신규 3개 손실을 하나의 스칼라로 결합하고, breakdown을 W&B 로깅용
딕셔너리로 남긴다.

손실 적용 여부는 가중치 0으로 끄는 것과 다르게 다룬다 — 계획서 §7.2 마지막
문장("해당 loss를 적용하지 않는 선택지도 포함")대로, 가중치가 0보다 크면
그 손실 값이 반드시 제공돼야 한다(``None``이면 fail-closed로 에러). 이건
"이 스텝에 깜빡하고 안 넘겼다"와 "이 손실을 의도적으로 껐다"를 구분하기 위함이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PretrainLossWeights:
    """λ1..λ5. 새로 추가된 3개(time/time_text/cross_modal)는 검증되지 않은
    손실이라 기본값 0.0(꺼짐)으로 시작한다 — MLM/SOP만 기본으로 켜져 있다.
    """

    mlm: float = 1.0
    sop: float = 1.0
    time_reconstruction: float = 0.0
    time_text_contrastive: float = 0.0
    cross_modal_matching: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mlm", "sop", "time_reconstruction", "time_text_contrastive", "cross_modal_matching"):
            if getattr(self, name) < 0:
                raise ValueError(f"loss weight must not be negative: {name}={getattr(self, name)}")


@dataclass(frozen=True)
class ComposedLoss:
    total: Tensor
    breakdown: dict[str, float]  # 가중치 적용 *전* 각 손실의 스칼라 값 — W&B 로깅용


def compose_total_loss(
    weights: PretrainLossWeights,
    *,
    mlm_loss: Tensor | None = None,
    sop_loss: Tensor | None = None,
    time_reconstruction_loss: Tensor | None = None,
    time_text_contrastive_loss: Tensor | None = None,
    cross_modal_matching_loss: Tensor | None = None,
) -> ComposedLoss:
    """가중치가 0보다 큰 손실은 반드시 제공돼야 한다 — 없으면 즉시 에러(fail-closed)."""
    components = {
        "mlm": (weights.mlm, mlm_loss),
        "sop": (weights.sop, sop_loss),
        "time_reconstruction": (weights.time_reconstruction, time_reconstruction_loss),
        "time_text_contrastive": (weights.time_text_contrastive, time_text_contrastive_loss),
        "cross_modal_matching": (weights.cross_modal_matching, cross_modal_matching_loss),
    }

    total: Tensor | None = None
    breakdown: dict[str, float] = {}
    for name, (weight, value) in components.items():
        if weight <= 0.0:
            continue
        if value is None:
            raise ValueError(
                f"weight for {name!r} is {weight} (> 0) but no loss value was provided — "
                "either compute it this step or set its weight to 0.0 to explicitly disable it"
            )
        breakdown[name] = float(value.detach()) if isinstance(value, Tensor) else float(value)
        contribution = weight * value
        total = contribution if total is None else total + contribution

    if total is None:
        raise ValueError("all loss weights are 0.0 — nothing to optimize")

    return ComposedLoss(total=total, breakdown=breakdown)
