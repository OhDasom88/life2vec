# representation_explorer

**근거**: 계획서 §10 (3D 표현 탐색)
**Phase**: Phase 4 (SAE Pilot과 병행) · **범위**: 현재 필수 (§3.1) · **상태**: 미착수

## 목적

UMAP/PCA 등 3D projection을 **참고용으로만** 제공한다. 데이터 분포/군집 탐색, 이상치 탐지, train/development/holdout 겹침 확인, modality별 분포 비교, 사전학습·미세조정 전후 거시적 변화 확인이 용도의 전부다. **3D 거리나 군집만으로 개념의 존재·단일 의미성·인과성을 확정하지 않는다** — 이 제약을 코드가 강제해야 한다(모든 출력에 disclaimer 메타데이터 첨부, 개념 확정 API 노출 금지).

## 구현 계획

1. `projection.py` — UMAP/PCA 실행, 파라미터·seed 기록.
2. `overlap_check.py` — train/development/holdout split 겹침 확인 뷰.
3. `neighborhood_preservation.py` — projection 품질 지표(원공간 대비 이웃 보존율).
4. `metadata.py` — 모든 projection 산출물에 원본 차원, 방법·파라미터·seed, checkpoint/layer, 색상 label 출처, neighborhood preservation 지표를 함께 기록하는 공통 wrapper. 이 메타데이터 없는 projection artifact는 UI(`../ui/explorer_3d/`)에 노출하지 않는다.

## 의존성

- 기존: 없음 (신규)
- 신규: `../representation_tracking/`(입력), `../sae/`(선택적, 색상 label 소스)

## Acceptance 연결

E1–E2 (계획서 §17-E)
