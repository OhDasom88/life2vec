"""§7.2 사전학습 손실 중 실제로 신규인 3개 (계획서 §7.2, ``L_time``/``L_time-text``/``L_cross-modal``).

착수 전 조사 결과, ``L_MLM``과 ``L_SOP``는 이미 구현돼 실제로 학습되고 있었다
(``src/tasks/grouped_mlm.py``의 ``GroupedMLM`` + ``scripts/online2_v2/run_v2_pretrain_loop.py``의
``model.cls_w * sop_loss + model.mlm_w * mlm_loss``). 이 모듈은 그걸 다시 만들지
않는다 — ``loss_composition.py``가 그 두 손실을 (재계산이 아니라) **입력으로**
받아 여기 새로 만드는 3개와 결합한다.

- ``L_time``(시계열 복원): 아무 데도 없었다. online1 쪽에 Cube latent 전용
  cosine 복원 손실 프로토타입(``src/online1/pretrain_smoke.py``)이 있었지만
  online1/online2 격리 원칙(각 문서가 서로의 산출물을 참고용으로만 쓰기로
  이미 합의)상 재사용하지 않고, 같은 수학적 형태(코사인 대신 MSE, 마스킹된
  위치만)를 참고해 online2용으로 새로 만들었다.
- ``L_time-text``(시계열-서사 대조학습): 저장소 어디에도 InfoNCE/CLIP류 코드가
  없었다 — 완전 신규.
- ``L_cross-modal``(시계열-서사-이미지-Cube 관계): online1에 recall@K **평가**
  지표 스텁만 있고 실제 학습 손실은 없었다 — 완전 신규.

세 손실 다 순수 torch 함수다. 실제 GPU 학습 루프(``run_v2_pretrain_loop.py``)에
연결하는 건 이 모듈의 범위 밖이다(README "아직 없는 것" 참조) — 여기서는
손실 계산 자체의 정확성만 보장한다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def time_reconstruction_loss(
    predicted_values: Tensor, target_values: Tensor, valid_mask: Tensor | None = None
) -> Tensor:
    """마스킹된 위치에서 연속값(정규화된 raw value)을 회귀하는 MSE.

    MLM은 이산 토큰(구간화된 bin)만 복원한다 — bin 안에서 실제 값이 어디에
    있었는지는 버려진다. 이 손실은 그 정보를 보존하려는 것이다.

    ``valid_mask``: 범주형 토큰처럼 연속값이 없는 위치를 제외한다(전부 False면
    0을 반환 — 이 스텝에 유효한 타깃이 없다는 뜻이지 학습이 잘 됐다는 뜻이
    아니다, 호출자가 별도로 카운트해야 한다).
    """
    if predicted_values.shape != target_values.shape:
        raise ValueError(
            f"shape mismatch: predicted={tuple(predicted_values.shape)} "
            f"target={tuple(target_values.shape)}"
        )
    if valid_mask is None:
        return F.mse_loss(predicted_values, target_values)
    if valid_mask.shape != predicted_values.shape:
        raise ValueError("valid_mask must match predicted_values shape")
    if not bool(valid_mask.any()):
        return predicted_values.new_zeros(())
    return F.mse_loss(predicted_values[valid_mask], target_values[valid_mask])


def multi_positive_info_nce(
    anchor_embeddings: Tensor,
    candidate_embeddings: Tensor,
    positive_mask: Tensor,
    *,
    temperature: float = 0.07,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """multi-positive InfoNCE — time-text와 cross-modal 둘 다 이 함수로 구현한다.

    ``positive_mask[i, j] = True``면 후보 j가 앵커 i의 양성이다. 계획서 §7.3이
    요구하는 "동일 의미가 여러 구간에 존재할 수 있으므로 multi-positive를
    허용한다"를 그대로 구현한다 — 앵커당 양성이 여러 개면 그 로그합을 쓴다
    (soft/multi-label InfoNCE, Khosla et al. SupCon과 동일한 형태).

    앵커에 양성이 하나도 없으면(``positive_mask`` 해당 행이 전부 False) 그
    앵커는 손실 계산에서 제외한다 — 0으로 나누기를 피하고, 대신 몇 개가
    제외됐는지는 호출자가 ``positive_mask.any(dim=1)``으로 직접 확인해야 한다.
    """
    if anchor_embeddings.shape[0] != positive_mask.shape[0]:
        raise ValueError("anchor_embeddings and positive_mask row count mismatch")
    if candidate_embeddings.shape[0] != positive_mask.shape[1]:
        raise ValueError("candidate_embeddings and positive_mask column count mismatch")

    has_positive = positive_mask.any(dim=1)
    if not bool(has_positive.any()):
        return anchor_embeddings.new_zeros(())

    anchor_norm = F.normalize(anchor_embeddings, dim=-1)
    candidate_norm = F.normalize(candidate_embeddings, dim=-1)
    similarity = anchor_norm @ candidate_norm.T / temperature  # (n_anchor, n_candidate)

    log_denominator = torch.logsumexp(similarity, dim=1)  # (n_anchor,)
    masked_similarity = similarity.masked_fill(~positive_mask, float("-inf"))
    log_positive_sum = torch.logsumexp(masked_similarity, dim=1)  # (n_anchor,)

    per_anchor_loss = log_denominator - log_positive_sum  # -log(sum_pos / sum_all)
    per_anchor_loss = per_anchor_loss[has_positive]

    if sample_weights is not None:
        weights = sample_weights[has_positive]
        denom = weights.sum().clamp_min(1e-12)
        return (per_anchor_loss * weights).sum() / denom

    return per_anchor_loss.mean()


def time_text_contrastive_loss(
    time_embeddings: Tensor,
    text_embeddings: Tensor,
    positive_mask: Tensor,
    *,
    temperature: float = 0.07,
    pair_weights: Tensor | None = None,
) -> Tensor:
    """시계열↔서사 대조학습(§7.2 L_time-text). ``contrastive_pairs.py``의
    pair 품질 등급을 ``pair_weights``로 넘기면 품질 낮은 pair의 기여를 줄인다.
    """
    return multi_positive_info_nce(
        time_embeddings,
        text_embeddings,
        positive_mask,
        temperature=temperature,
        sample_weights=pair_weights,
    )


def cross_modal_matching_loss(
    embeddings_by_modality: Mapping[str, Tensor],
    positive_mask: Tensor,
    *,
    temperature: float = 0.07,
    modality_pairs: Sequence[tuple[str, str]] | None = None,
) -> Tensor:
    """시계열·서사·이미지·Cube 중 여러 modality 쌍에 InfoNCE를 적용해 평균한다(§7.2 L_cross-modal).

    같은 배치 인덱스가 여러 modality에 걸쳐 같은 관측을 가리킨다고 가정하고,
    ``positive_mask``(n, n)는 모든 modality 쌍에 공통으로 쓴다(같은 관측
    집합이므로). 특정 modality가 이번 배치에 없으면(``embeddings_by_modality``에
    없음) 그 modality가 낀 쌍은 건너뛴다 — modality missingness를 여기서
    강제하지 않고 호출자가 이미 걸러서 넘긴다고 가정한다.
    """
    available = set(embeddings_by_modality.keys())
    if modality_pairs is None:
        names = sorted(available)
        modality_pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    pair_losses = []
    for left, right in modality_pairs:
        if left not in available or right not in available:
            continue
        pair_losses.append(
            multi_positive_info_nce(
                embeddings_by_modality[left],
                embeddings_by_modality[right],
                positive_mask,
                temperature=temperature,
            )
        )
    if not pair_losses:
        return torch.zeros(())
    return torch.stack(pair_losses).mean()
