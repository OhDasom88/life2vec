# explorer_3d UI

**근거**: 계획서 §15, §10 · **백엔드**: `../../representation_explorer/` · **상태**: 미착수

## 기능

거시 분포·이상치 탐색용 3D(UMAP/PCA) 뷰. `representation_explorer.metadata`가 강제하는 메타데이터(원본 차원, 방법·파라미터·seed, checkpoint/layer, 색상 label 출처, neighborhood preservation 지표) 없는 projection artifact는 표시하지 않는다.

## 산출물

projection artifact (조회 전용, 이 UI 자체는 artifact를 생성하지 않고 `representation_explorer`가 생성한 것을 렌더링만 함)

## 구현 계획

1. projection scatter 뷰 + 메타데이터 패널(항상 함께 표시).
2. train/development/holdout 겹침 확인 토글.
3. modality별 분포 비교, 사전학습/미세조정 전후 비교 뷰.
4. **가드**: "3D 거리·군집만으로 개념 존재/단일 의미성/인과성을 확정하지 않는다"는 disclaimer를 화면에 상시 고정 표시.
