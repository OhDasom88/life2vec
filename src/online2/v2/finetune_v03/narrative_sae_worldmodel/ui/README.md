# ui

**근거**: 계획서 §15 (UI 구성)
**상태**: `data_grounding_curation/` + `pipeline_explorer/` 구현. 스택은 Streamlit으로 확정(`requirements.txt`에 추가) — 나머지 6개 서브앱도 이 스택을 따른다.

## 공통 원칙

UI 화면 상태는 정본(source of truth)으로 사용하지 않는다. 모든 변경은 각 서브앱이 아니라 대응 백엔드 패키지의 versioned artifact와 immutable transaction으로 저장하고, UI는 그 결과를 조회·트리거만 한다.

## 서브앱 맵

| 서브앱 | 근거 백엔드 | 생성 artifact |
|---|---|---|
| [data_grounding_curation/](data_grounding_curation/README.md) | `../narrative_grounding/` | grounding decision — **구현 완료**(§5.3 검토 큐, §5.1 검색 화면은 아직) |
| [pipeline_explorer/](pipeline_explorer/README.md) | `../narrative_grounding/` + `../sae/` + 6-arm 재학습 산출물 | (조회 전용, artifact 생성 안 함) — **구현 완료**(원시데이터↔토큰↔시퀀스 추적, 인코딩·dead-ratio 트레이스, SAE 개념/시퀀스 유사도, 학습 버전 브라우저 4탭). 원래 `sequence_composer_ui`/`sae_feature_dashboard`/`run_artifact_control`로 나뉘어 있던 계획 중 **조회 영역만** 통합 구현 — 아래 세 항목 참조 |
| [sequence_composer_ui/](sequence_composer_ui/README.md) | 기존 `cf1s/core_raw_transaction.py` + `../sequence_curation/` | sequence transaction — 원시↔토큰↔시퀀스 **조회**는 `pipeline_explorer/`가 구현. **편집 액션**(`curation_actions` 트리거)은 여전히 미착수 |
| [explorer_3d/](explorer_3d/README.md) | `../representation_explorer/` | projection artifact |
| [sae_feature_dashboard/](sae_feature_dashboard/README.md) | `../sae/` | feature evaluation — encoding trace·dead ratio·**근사적** concept 유사도는 `pipeline_explorer/`가 구현. feature 상태 머신 뷰·`concept_mapping`(N:M)·`feature_alignment` 뷰는 그 백엔드 모듈 자체가 없어 여전히 미착수 |
| [concept_governance_ui/](concept_governance_ui/README.md) | 기존 `cf1s/core_locks.py` + `../concept_governance/` | ontology/mapping version |
| [world_model_validator/](world_model_validator/README.md) | `../world_model/` | readiness report |
| [edit_policy_lab/](edit_policy_lab/README.md) | `../edit_policy_rl/` + 기존 CF1S quarantine/promotion 뷰어 | edit transaction |
| [run_artifact_control/](run_artifact_control/README.md) | W&B run/sweep, 기존 `outputs/cf1s_core/` | run/selection manifest — 로컬 run(`run_manifest_v2.json`) 비교는 `pipeline_explorer/`가 구현. W&B 연동·§4.2 추적 키 체인 트리 뷰는 미착수 |

## 착수 순서

백엔드 모듈이 없는 UI는 만들 수 없다. 각 서브앱은 대응 백엔드 패키지가 최소 동작 가능한 상태(Phase 해당 단계 acceptance 일부 통과)가 된 뒤 착수한다 — 즉 UI 8종을 한 번에 만들지 않고 Phase 1~7 진행과 병행해 하나씩 붙인다. 예외적으로 `pipeline_explorer/`는 세 서브앱(sequence_composer_ui/sae_feature_dashboard/run_artifact_control)에 걸친 **조회 전용** 기능을 사용자 요청으로 먼저 통합 구현했다 — "여러 화면에 흩어져 있으면 전체 파이프라인을 유기적으로 못 본다"는 문제 제기가 이유였고, 이건 각 서브앱을 개별적으로 순서대로 만드는 것보다 조회 기능을 먼저 하나로 묶는 게 더 유용하다는 판단에 따른 계획 변경이다.
