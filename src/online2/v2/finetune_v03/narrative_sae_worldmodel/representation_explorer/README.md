# representation_explorer

**근거**: 계획서 §10 (3D 표현 탐색)
**Phase**: Phase 4 (SAE Pilot과 병행) · **범위**: 현재 필수 (§3.1) · **상태**: 구현 완료(원 4개 파일, 29개 테스트 통과, scikit-learn/umap-learn 사용) + `label_separation_probe.py` 추가(2026-07-23, 실제 OOF 표현으로 검증 완료)

## 목적

UMAP/PCA 등 3D projection을 **참고용으로만** 제공한다. 데이터 분포/군집 탐색, 이상치 탐지, train/development/holdout 겹침 확인, modality별 분포 비교, 사전학습·미세조정 전후 거시적 변화 확인이 용도의 전부다. **3D 거리나 군집만으로 개념의 존재·단일 의미성·인과성을 확정하지 않는다** — 이 제약을 코드가 강제한다(아래 참조).

## ⚠ 착수 전 점검 결과 — 이번엔 진짜로 전부 신규였다

`analysis/visualisation/person_space.ipynb`에 실제 `umap.UMAP(...).fit_transform(...)` 호출이 있긴 했지만, 파라미터·seed·checkpoint를 기록하지 않고 disclaimer도 없는 1회성 탐색 노트북 셀이었다(재사용 가능한 함수 아님). `neighborhood_preservation`/`trustworthiness`/`continuity` 관련 코드는 저장소 어디에도 없었다. `scripts/online2_audit/run_phase_b1.py`에 split overlap 확인이 있었지만 farm 해시 버킷 카운팅이었지 projection 공간(3D 좌표) 겹침이 아니었다. `umap-learn==0.5.12`/`scikit-learn==1.7.2`는 이미 life2vec 환경에 설치돼 있어 바로 쓸 수 있었다.

## 구현한 것

- [`metadata.py`](metadata.py) — `ProjectionMetadata`/`ProjectionArtifact`. 원본 차원·방법·파라미터·seed·checkpoint_id·layer·color_label_source·neighborhood_preservation 중 하나라도 비어 있으면 **객체 생성 자체가 안 된다**(`__post_init__`에서 검증) — "메타데이터 없는 projection artifact는 노출하지 않는다"를 타입으로 강제한 것. `disclaimer` 필드는 고정 문자열이고 다른 값으로 덮어쓰려 하면 즉시 에러 — 개념 확정성 경고를 지울 수 없게 만들었다.
- [`neighborhood_preservation.py`](neighborhood_preservation.py) — `sklearn.manifold.trustworthiness`를 그대로 사용(새 지표 발명 안 함). identity projection(원본=투영)은 1.0, 무작위로 섞은 투영은 그보다 낮음을 실제 계산으로 확인.
- [`projection.py`](projection.py) — `project_activations(activations, point_ids, method="PCA"|"UMAP", ...)`. PCA는 `sklearn.decomposition.PCA`, UMAP은 `umap-learn`을 그대로 쓴다. 호출할 때마다 `neighborhood_preservation`을 자동 계산해 메타데이터에 싣는다 — "계산은 했는데 기록을 깜빡했다"가 구조적으로 불가능. UMAP은 무거운 import라 함수 호출 시점에만 지연 import(`Qwen3EmbeddingProvider`와 같은 패턴).
- [`overlap_check.py`](overlap_check.py) — `split_overlap_in_projection_space(artifact, split_by_point_id, k=...)`. 각 포인트의 k-근접 이웃 중 다른 split 비율을 split별로 집계. **참고용이라는 점을 코드 주석에 명시** — 이 비율이 높다고 곧바로 데이터 누수가 아니고, 낮다고 누수가 없다는 보장도 아니다(projection이 원공간을 왜곡했을 수 있어 `neighborhood_preservation`을 항상 같이 봐야 함). 실제 leakage 판정은 이 모듈이 아니라 CF1S의 `core_fold_provenance` 계약이 한다.

## 추가: label_separation_probe.py — 정상/비정상 concept-space 분리 실측 (2026-07-23)

사용자 요청("병명이 있는 경우와 없는 경우가 제대로 잘 구분되어 있는지 확인")에 답하기 위해 [`label_separation_probe.py`](label_separation_probe.py)를 추가했다. `metadata.py`와 같은 원칙(강제된 `checkpoint_ids`/`label_source`, disclaimer 덮어쓰기 금지)을 따르되, 3D projection과 다른 질문을 던진다 — "군집이 보기에 갈려 보이는가"가 아니라 **"cross-validated linear probe로 실제 분리 가능한가"**(`E2_PREDICTIVE` 수준 실측치, AUROC/실루엣). 반드시 **out-of-fold** 표현에 대해서만 계산하도록 타입으로 강제한다(in-sample이면 자명하게 좋아 보여 무의미).

`run_diagnosis_finetune_v03.py`가 매 fold마다 계산은 하지만 CSV에서는 버리던 `z_proj`/`h_binary`를 이제 `*_oof_representations.npz`로 저장하도록 고쳐서(코드 변경 최소, 새 학습 불필요), 실제 3-fold cv_repeated 실행(`outputs/online2/v2_finetune_v03/runs/probe_oof_3fold`, 35개 라벨된 케이스 전체 OOF 커버) 결과에 `scripts/online2_v2/v03/probe_normal_abnormal_separation_v03.py`를 돌렸다.

**실측 결과 — 예상보다 약하다, 그리고 이건 정직하게 봐야 할 신호다**:

| 표현 | n | pos(비정상) | CV AUROC | 실루엣 |
|---|---|---|---|---|
| `h_binary` (binary head 직전 pooled) | 35 | 31 | 0.415 ± 0.154 | -0.002 |
| `z_proj` (SupCon/prototype projection) | 35 | 31 | 0.536 ± 0.249 | -0.037 |

두 표현 다 **거의 우연 수준(0.5) 또는 그 이하**다 — 공식 CV 리포트의 binary AUROC 0.911(`DIAGNOSIS_FINETUNE_V03_CV_REPORT.md`)과 크게 다르다. 모순이 아니다: 0.911은 attention pooling + binary head 전체가 함께 만든 성능이고, 이 probe는 그 head를 다시 훈련시키지 않고 "pooled 표현 자체가 이미 선형적으로 갈려 있는가"만 본다 — **정상/비정상 분리력의 대부분이 표현 자체보다 head의 비선형 변환에서 나온다**는 뜻일 수 있다. 다만 이 probe는 35건 중 정상이 4건뿐인 극단적 불균형에서 5-fold조차 못 채워 4-fold로 줄었고(각 fold당 정상 1건), fold별 AUROC가 0.143~0.75로 크게 흔들린다 — **표본이 너무 작아 이 수치 자체의 신뢰구간이 넓다는 것도 같이 봐야 한다.** 결론은 "표현이 나쁘다"가 아니라 "이 질문에 안정적으로 답하려면 정상 케이스가 지금보다 많이 필요하다"는 쪽에 가깝다.

## 아직 없는 것

- `../ui/explorer_3d/`(실제 화면) — 이 모듈은 순수 계산 계층만 구현했다. UI는 계획서가 예정한 대로 다음 단계.
- `../sae/`(선택적 color label 소스)가 아직 없어서, SAE feature 기준 색상은 지금은 쓸 수 없다(`color_label_source`에 `"split"`/`"narrative_template_id"` 등 이미 있는 라벨만 넣을 수 있음).
- 실제 `../representation_tracking/`의 `ActivationRef`/캡시된 activation을 이 모듈에 직접 흘려 넣는 배치 스크립트는 아직 없다 — `project_activations`는 `np.ndarray`를 받는 순수 함수라 연결 자체는 어렵지 않지만, 아직 안 짰다.

## 의존성

- 기존: `scikit-learn`, `umap-learn` (둘 다 이미 설치돼 있었음, 재사용)
- 신규 데이터 소스: `../representation_tracking/`(activation 입력, 아직 연결 안 됨), `../sae/`(선택적 color label, 아직 없음)

## Acceptance 연결

E1–E2 (계획서 §17-E)
