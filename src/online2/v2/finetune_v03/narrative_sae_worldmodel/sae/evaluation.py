"""§9.4 SAE 평가 6영역 중 라벨 없이 계산 가능한 두 영역(복원, 희소성) + downstream fidelity.

의미(semantic)/분해(splitting·absorption 등)/안정성(stability)/기능(ablation·
steering) 4개 영역은 사람 라벨(예: `../narrative_grounding/`의 검토 결과)이나
실제 개입 실험이 있어야 계산할 수 있다 — 아직 없다(README "아직 없는 것" 참조).
"SAE reconstruction과 sparsity가 좋아도 의미적 순도가 확보됐다고 판정하지
않는다"(계획서 §9.4)를 지키려면 이 두 영역만으로 SAE 품질을 확정하지 않아야
한다 — 이 모듈의 함수들은 그 자체로 "이 SAE는 좋다"를 결론 내지 않는다.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_activation_frequency(codes: Tensor) -> Tensor:
    """feature별로 배치 안에서 몇 % 샘플이 활성화시켰는지 (dict_size,)."""
    return (codes > 0).float().mean(dim=0)


def reconstruction_metrics(x: Tensor, reconstruction: Tensor) -> dict[str, float]:
    """reconstruction error(MSE), explained variance."""
    mse = torch.nn.functional.mse_loss(reconstruction, x).item()
    total_variance = x.var(dim=0, unbiased=False).sum().item()
    residual_variance = (x - reconstruction).var(dim=0, unbiased=False).sum().item()
    explained_variance = 1.0 - residual_variance / total_variance if total_variance > 0 else 0.0
    return {"mse": mse, "explained_variance": explained_variance}


def sparsity_metrics(codes: Tensor) -> dict[str, float]:
    """평균 L0(샘플당 활성 feature 개수), 평균 activation frequency, dead feature ratio."""
    active = codes > 0
    mean_l0 = active.sum(dim=-1).float().mean().item()
    frequency = compute_activation_frequency(codes)
    dead_feature_ratio = (frequency == 0).float().mean().item()
    return {
        "mean_l0": mean_l0,
        "mean_activation_frequency": frequency.mean().item(),
        "dead_feature_ratio": dead_feature_ratio,
    }


def downstream_fidelity(original_output: Tensor, reconstructed_output: Tensor) -> dict[str, float]:
    """원본 activation으로 낸 downstream 출력과 SAE 재구성으로 낸 출력의 차이.

    ``../representation_tracking/attribution.py``의 modality ablation과 같은
    패턴 — 실제 downstream 모델(예: finetune critic)을 원본/재구성 각각으로
    순전파하는 건 호출자 책임이고, 여기서는 이미 계산된 두 출력을 받아 delta만
    계산한다.
    """
    if original_output.shape != reconstructed_output.shape:
        raise ValueError(
            f"shape mismatch: original={tuple(original_output.shape)} "
            f"reconstructed={tuple(reconstructed_output.shape)}"
        )
    delta = original_output - reconstructed_output
    flat_original = original_output.flatten().double()
    flat_reconstructed = reconstructed_output.flatten().double()
    if flat_original.numel() > 1 and flat_original.std() > 0 and flat_reconstructed.std() > 0:
        correlation = float(torch.corrcoef(torch.stack([flat_original, flat_reconstructed]))[0, 1])
    else:
        correlation = float("nan")
    return {
        "mean_absolute_delta": float(delta.abs().mean()),
        "correlation": correlation,
    }
