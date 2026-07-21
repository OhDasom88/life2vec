"""§10 3D projection 산출물의 공통 메타데이터 + 상시 disclaimer.

계획서 §10: "UI에는 원본 차원, projection 방법·파라미터·seed, checkpoint/layer,
색상 label의 출처와 neighborhood preservation 지표를 표시한다. 3D 거리나
군집만으로 개념의 존재, 단일 의미성 또는 인과성을 확정하지 않는다."

착수 전 조사 결과 이 강제 자체가 저장소 어디에도 없었다(진짜 신규) —
`analysis/visualisation/person_space.ipynb`에 실제 UMAP 호출이 있긴 하지만
파라미터/seed/checkpoint 기록이나 disclaimer 없이 그냥 그려보는 1회성
노트북 셀이었다. 이 모듈은 그 강제를 타입으로 만든다: 이 필드들이 없으면
``ProjectionArtifact``를 아예 만들 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DISCLAIMER = (
    "3D 거리나 군집만으로 개념의 존재, 단일 의미성 또는 인과성을 확정하지 않는다. "
    "데이터 분포·이상치 탐색, split 겹침 확인, modality 분포 비교, 거시적 변화 확인 "
    "용도로만 참고한다 (계획서 §10)."
)


@dataclass(frozen=True)
class ProjectionMetadata:
    original_dimensionality: int
    method: str  # "PCA"|"UMAP"
    params: Mapping[str, Any]
    seed: int
    checkpoint_id: str
    layer: str
    color_label_source: str  # 색상 label이 어디서 왔는지(예: "split", "sae_feature_42")
    neighborhood_preservation: float  # trustworthiness, [0, 1]

    def __post_init__(self) -> None:
        if self.original_dimensionality <= 0:
            raise ValueError("original_dimensionality must be positive")
        if not self.method:
            raise ValueError("method must not be empty")
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty — projection without a known source is not traceable")
        if not self.layer:
            raise ValueError("layer must not be empty")
        if not self.color_label_source:
            raise ValueError("color_label_source must not be empty — every projection needs labeled provenance")
        if not (0.0 <= self.neighborhood_preservation <= 1.0):
            raise ValueError(
                f"neighborhood_preservation must be in [0, 1], got {self.neighborhood_preservation}"
            )


@dataclass(frozen=True)
class ProjectionArtifact:
    """3D 좌표 + 강제된 메타데이터 + disclaimer. 메타데이터 없이는 생성 자체가 안 된다."""

    metadata: ProjectionMetadata
    point_ids: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    disclaimer: str = field(default=DISCLAIMER)

    def __post_init__(self) -> None:
        if len(self.point_ids) != len(self.coordinates):
            raise ValueError(
                f"point_ids ({len(self.point_ids)}) and coordinates "
                f"({len(self.coordinates)}) length mismatch"
            )
        if any(len(point) != 3 for point in self.coordinates):
            raise ValueError("every coordinate must have exactly 3 components")
        if self.disclaimer != DISCLAIMER:
            raise ValueError("disclaimer must not be overridden or removed")
