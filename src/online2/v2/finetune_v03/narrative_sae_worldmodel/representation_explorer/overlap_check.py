"""projection 공간에서 TRAIN/DEVELOPMENT/HOLDOUT 겹침을 참고용으로 확인 (계획서 §10).

착수 전 조사 결과 split overlap 확인 자체는 있었다(``scripts/online2_audit/run_phase_b1.py``)
— 하지만 farm 해시로 만든 ID/버킷 겹침만 세는 데이터 무결성 감사였지 projection
공간(3D 좌표) 겹침이 아니었다. 여기서 하는 건 그 3D 좌표 공간에서 서로 다른
split의 포인트가 얼마나 뒤섞여 있는지 참고용으로 보는 것뿐이다 — 계획서 §10이
명시하듯 이건 진단 도구지 leakage 판정 도구가 아니다. **실제 데이터 접근 권한
위반(leakage) 판정은 이 모듈이 아니라 CF1S의 ``core_fold_provenance``/
``case_fold_provenance`` 계약이 한다.**
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .metadata import ProjectionArtifact


def split_overlap_in_projection_space(
    artifact: ProjectionArtifact,
    split_by_point_id: Mapping[str, str],
    *,
    k: int = 5,
) -> dict[str, dict[str, float]]:
    """각 포인트의 k-근접 이웃 중 다른 split에 속한 비율을, split별로 평균낸다.

    비율이 높을수록(예: TRAIN 포인트의 이웃 다수가 HOLDOUT) 그 split이
    projection 공간에서 다른 split과 뒤섞여 있다는 뜻이다 — **참고용**이다.
    이 비율이 높다고 곧바로 데이터 누수를 의미하지 않는다(원래 표현 공간이
    그렇게 생겼을 수도 있다), 반대로 낮다고 누수가 없다는 보장도 아니다
    (projection이 원공간을 왜곡했을 수 있다 — ``artifact.metadata.
    neighborhood_preservation``을 항상 같이 봐야 하는 이유).
    """
    missing = [pid for pid in artifact.point_ids if pid not in split_by_point_id]
    if missing:
        raise ValueError(f"split_by_point_id missing entries for: {missing[:5]}...")

    n_points = len(artifact.point_ids)
    if n_points <= k:
        raise ValueError(f"need more than k={k} points, got {n_points}")

    coordinates = np.array(artifact.coordinates)
    splits = [split_by_point_id[pid] for pid in artifact.point_ids]

    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)  # +1: 자기 자신 포함
    _, indices = neighbors.kneighbors(coordinates)

    other_split_fraction_by_index: list[float] = []
    for i, neighbor_indices in enumerate(indices):
        own_split = splits[i]
        neighbor_splits = [splits[j] for j in neighbor_indices if j != i][:k]
        other_count = sum(1 for s in neighbor_splits if s != own_split)
        other_split_fraction_by_index.append(other_count / len(neighbor_splits))

    by_split: dict[str, list[float]] = {}
    for split_name, fraction in zip(splits, other_split_fraction_by_index):
        by_split.setdefault(split_name, []).append(fraction)

    return {
        split_name: {
            "n": float(len(fractions)),
            "mean_other_split_neighbor_fraction": sum(fractions) / len(fractions),
        }
        for split_name, fractions in by_split.items()
    }
