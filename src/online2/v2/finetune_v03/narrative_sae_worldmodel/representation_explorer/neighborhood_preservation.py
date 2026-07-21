"""projection 품질 지표 — 원공간 이웃 관계가 3D로 얼마나 보존됐는지 (계획서 §10).

착수 전 조사 결과 저장소 어디에도 이런 지표 계산이 없었다(``trustworthiness``/
``continuity``/``neighborhood_preservation`` grep 결과 0건) — 진짜 신규다.
새 지표를 발명하지 않고 표준 지표(``sklearn.manifold.trustworthiness``)를
그대로 쓴다.
"""

from __future__ import annotations

import numpy as np


def compute_neighborhood_preservation(
    original: np.ndarray, projected: np.ndarray, *, n_neighbors: int = 5
) -> float:
    """``sklearn.manifold.trustworthiness`` — 값이 1에 가까울수록 원공간의
    k-근접 이웃 관계가 projection에서도 잘 보존됐다는 뜻. 0.5 근처면 projection이
    원공간 구조를 거의 반영하지 못한다는 뜻이므로, 그런 projection의 군집을
    해석하는 건 특히 위험하다(§10 disclaimer가 이 값을 항상 옆에 두라고
    요구하는 이유).
    """
    from sklearn.manifold import trustworthiness

    if original.shape[0] != projected.shape[0]:
        raise ValueError(
            f"point count mismatch: original={original.shape[0]} projected={projected.shape[0]}"
        )
    if original.shape[0] <= n_neighbors:
        raise ValueError(
            f"need more than n_neighbors={n_neighbors} points, got {original.shape[0]}"
        )
    return float(trustworthiness(original, projected, n_neighbors=n_neighbors))
