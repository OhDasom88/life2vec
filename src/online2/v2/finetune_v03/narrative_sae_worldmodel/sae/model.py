"""§9.2/§9.3 SAE 아키텍처. dictionary expansion factor, L1/Top-K sparsity 지원.

새 아키텍처를 발명하지 않는다 — 표준 SAE 구성(encoder: 선형 + ReLU 또는
Top-K, decoder: 선형, pre-encoder bias 차감 — Bricken et al. 2023, Anthropic
"Towards Monosemanticity"의 표준 관례)을 그대로 쓴다. BatchTopK(§9.3이 언급하는
세 번째 sparsity 옵션)는 배치 전체에서 top-k를 뽑는 변형인데, 이 모듈은 아직
샘플별 Top-K까지만 구현했다 — README "아직 없는 것" 참조.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SPARSITY_MODES = frozenset({"L1", "TOPK"})


def _topk_activation(pre_activation: Tensor, top_k: int) -> Tensor:
    """ReLU 이후 값 기준 상위 top_k만 남기고 나머지는 0으로."""
    activated = F.relu(pre_activation)
    dict_size = activated.shape[-1]
    if top_k >= dict_size:
        return activated
    values, indices = torch.topk(activated, top_k, dim=-1)
    result = torch.zeros_like(activated)
    result.scatter_(-1, indices, values)
    return result


class SparseAutoencoder(nn.Module):
    """activation(input_dim) -> codes(dict_size, 희소) -> reconstruction(input_dim)."""

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        *,
        sparsity_mode: str = "L1",
        top_k: int | None = None,
        tied_weights: bool = False,
    ) -> None:
        super().__init__()
        if sparsity_mode not in SPARSITY_MODES:
            raise ValueError(f"invalid sparsity_mode: {sparsity_mode!r}")
        if sparsity_mode == "TOPK" and (top_k is None or top_k <= 0):
            raise ValueError("TOPK sparsity_mode requires a positive top_k")
        if input_dim <= 0 or dict_size <= 0:
            raise ValueError("input_dim and dict_size must be positive")

        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_mode = sparsity_mode
        self.top_k = top_k
        self.tied_weights = tied_weights

        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder_weight = nn.Parameter(torch.empty(dict_size, input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(dict_size))
        nn.init.kaiming_uniform_(self.encoder_weight, a=5**0.5)

        if tied_weights:
            self.decoder_weight = None
        else:
            self.decoder_weight = nn.Parameter(torch.empty(input_dim, dict_size))
            nn.init.kaiming_uniform_(self.decoder_weight, a=5**0.5)

    def _decoder_weight(self) -> Tensor:
        return self.encoder_weight.t() if self.tied_weights else self.decoder_weight

    def encode(self, x: Tensor) -> Tensor:
        centered = x - self.pre_bias
        pre_activation = F.linear(centered, self.encoder_weight, self.encoder_bias)
        if self.sparsity_mode == "L1":
            return F.relu(pre_activation)
        return _topk_activation(pre_activation, self.top_k)

    def decode(self, codes: Tensor) -> Tensor:
        return F.linear(codes, self._decoder_weight()) + self.pre_bias

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codes = self.encode(x)
        reconstruction = self.decode(codes)
        return reconstruction, codes


@dataclass(frozen=True)
class SAELossOutput:
    total: Tensor
    reconstruction_loss: float
    sparsity_loss: float


def sae_loss(
    x: Tensor, reconstruction: Tensor, codes: Tensor, *, sparsity_mode: str, l1_coefficient: float = 0.0
) -> SAELossOutput:
    """복원 MSE + (L1 모드일 때만) 희소성 페널티.

    Top-K는 활성 개수 자체가 구조적으로 제한되므로 별도 L1 페널티가 필요
    없다(표준 관례) — ``sparsity_mode="TOPK"``이면 ``sparsity_loss``는 항상 0.
    """
    if sparsity_mode not in SPARSITY_MODES:
        raise ValueError(f"invalid sparsity_mode: {sparsity_mode!r}")
    reconstruction_loss = F.mse_loss(reconstruction, x)
    if sparsity_mode == "L1":
        sparsity_term = l1_coefficient * codes.abs().sum(dim=-1).mean()
    else:
        sparsity_term = torch.zeros((), device=x.device)
    total = reconstruction_loss + sparsity_term
    return SAELossOutput(
        total=total,
        reconstruction_loss=float(reconstruction_loss.detach()),
        sparsity_loss=float(sparsity_term.detach()) if isinstance(sparsity_term, Tensor) else float(sparsity_term),
    )
