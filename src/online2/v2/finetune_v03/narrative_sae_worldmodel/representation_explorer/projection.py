"""UMAP/PCA 3D projection 실행 (계획서 §10).

착수 전 조사 결과 저장소에 실제 UMAP 호출은 있었다
(``analysis/visualisation/person_space.ipynb``, ``umap.UMAP(**param).fit_transform(act)``)
— 하지만 파라미터·seed·checkpoint를 기록하지 않는 1회성 노트북 셀이었고
재사용 가능한 함수가 아니었다. 여기서 하는 일은 그 호출을 재사용 가능한
함수로 감싸고, ``metadata.ProjectionArtifact``가 강제하는 필드(원본 차원,
방법·파라미터·seed, checkpoint/layer, color label 출처, neighborhood
preservation)를 빠짐없이 채우는 것이다 — projection 알고리즘 자체는
``scikit-learn``/``umap-learn``을 그대로 쓴다(새로 만들지 않음).

UMAP은 무거운 import라 실제로 이 모듈의 함수를 호출할 때만 지연 import한다
(``Qwen3EmbeddingProvider``의 torch/transformers 지연 import와 같은 패턴).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .metadata import ProjectionArtifact, ProjectionMetadata
from .neighborhood_preservation import compute_neighborhood_preservation

_SUPPORTED_METHODS = frozenset({"PCA", "UMAP"})


def _run_pca(activations: np.ndarray, *, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=3, random_state=seed).fit_transform(activations)


def _run_umap(activations: np.ndarray, *, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=3, random_state=seed, n_neighbors=n_neighbors, min_dist=min_dist
    )
    return reducer.fit_transform(activations)


def project_activations(
    activations: np.ndarray,
    point_ids: Sequence[str],
    *,
    method: str,
    checkpoint_id: str,
    layer: str,
    color_label_source: str,
    seed: int = 0,
    trustworthiness_n_neighbors: int = 5,
    method_params: Mapping[str, Any] | None = None,
) -> ProjectionArtifact:
    """activations(n_points, original_dim) -> 3D ``ProjectionArtifact``.

    ``method``: ``"PCA"`` 또는 ``"UMAP"``. UMAP 파라미터는
    ``method_params``로 넘긴다(예: ``{"n_neighbors": 15, "min_dist": 0.1}``).
    neighborhood preservation은 매 호출마다 자동 계산돼 메타데이터에 실린다 —
    "계산은 했는데 기록을 깜빡했다"가 구조적으로 불가능하다.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method!r} (choose from {sorted(_SUPPORTED_METHODS)})")
    if activations.shape[0] != len(point_ids):
        raise ValueError(
            f"activations rows ({activations.shape[0]}) and point_ids "
            f"({len(point_ids)}) length mismatch"
        )

    params = dict(method_params or {})
    if method == "PCA":
        coordinates = _run_pca(activations, seed=seed)
    else:
        coordinates = _run_umap(
            activations,
            seed=seed,
            n_neighbors=params.get("n_neighbors", 15),
            min_dist=params.get("min_dist", 0.1),
        )

    preservation = compute_neighborhood_preservation(
        activations, coordinates, n_neighbors=trustworthiness_n_neighbors
    )

    metadata = ProjectionMetadata(
        original_dimensionality=activations.shape[1],
        method=method,
        params=params,
        seed=seed,
        checkpoint_id=checkpoint_id,
        layer=layer,
        color_label_source=color_label_source,
        neighborhood_preservation=preservation,
    )
    return ProjectionArtifact(
        metadata=metadata,
        point_ids=tuple(point_ids),
        coordinates=tuple(tuple(float(v) for v in row) for row in coordinates),
    )
