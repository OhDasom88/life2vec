"""§9.3 dead feature resampling.

SAE 학습에서 흔히 쓰는 표준 기법이다: 오래 활성화되지 않은 feature의
encoder/decoder 가중치를 재초기화해 dictionary 활용도를 높인다. 새 기법을
발명하지 않는다 — 표준적인 "재초기화" 방식을 그대로 구현했다(더 정교한
"재구성 오차가 큰 샘플 방향으로 재초기화"는 하지 않는다 — README "아직
없는 것" 참조).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import SparseAutoencoder


def resample_dead_features(
    model: SparseAutoencoder, activation_frequency: Tensor, *, threshold: float = 0.0
) -> int:
    """``activation_frequency``(dict_size,)가 ``threshold`` 이하인 feature를 재초기화한다.

    반환값: 재초기화된 feature 개수. 0이면 죽은 feature가 없었다는 뜻이다.
    """
    if activation_frequency.shape != (model.dict_size,):
        raise ValueError(
            f"activation_frequency shape {tuple(activation_frequency.shape)} "
            f"does not match dict_size {model.dict_size}"
        )
    dead_mask = activation_frequency <= threshold
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    with torch.no_grad():
        fresh_encoder = torch.empty_like(model.encoder_weight)
        nn.init.kaiming_uniform_(fresh_encoder, a=5**0.5)
        model.encoder_weight.data[dead_mask] = fresh_encoder[dead_mask]
        model.encoder_bias.data[dead_mask] = 0.0

        if not model.tied_weights:
            fresh_decoder = torch.empty_like(model.decoder_weight)
            nn.init.kaiming_uniform_(fresh_decoder, a=5**0.5)
            model.decoder_weight.data[:, dead_mask] = fresh_decoder[:, dead_mask]

    return n_dead
