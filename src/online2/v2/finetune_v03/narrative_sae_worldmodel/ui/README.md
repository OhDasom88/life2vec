# ui

**근거**: 계획서 §15 (UI 구성)
**상태**: 미착수, 현재 저장소에 UI 프레임워크 흔적 없음(신규 스택 선정 필요 — Streamlit/Gradio 등, 미결정)

## 공통 원칙

UI 화면 상태는 정본(source of truth)으로 사용하지 않는다. 모든 변경은 각 서브앱이 아니라 대응 백엔드 패키지의 versioned artifact와 immutable transaction으로 저장하고, UI는 그 결과를 조회·트리거만 한다.

## 서브앱 맵

| 서브앱 | 근거 백엔드 | 생성 artifact |
|---|---|---|
| [data_grounding_curation/](data_grounding_curation/README.md) | `../narrative_grounding/` | grounding decision |
| [sequence_composer_ui/](sequence_composer_ui/README.md) | 기존 `cf1s/core_raw_transaction.py` + `../sequence_curation/` | sequence transaction |
| [explorer_3d/](explorer_3d/README.md) | `../representation_explorer/` | projection artifact |
| [sae_feature_dashboard/](sae_feature_dashboard/README.md) | `../sae/` | feature evaluation |
| [concept_governance_ui/](concept_governance_ui/README.md) | 기존 `cf1s/core_locks.py` + `../concept_governance/` | ontology/mapping version |
| [world_model_validator/](world_model_validator/README.md) | `../world_model/` | readiness report |
| [edit_policy_lab/](edit_policy_lab/README.md) | `../edit_policy_rl/` + 기존 CF1S quarantine/promotion 뷰어 | edit transaction |
| [run_artifact_control/](run_artifact_control/README.md) | W&B run/sweep, 기존 `outputs/cf1s_core/` | run/selection manifest |

## 착수 순서

백엔드 모듈이 없는 UI는 만들 수 없다. 각 서브앱은 대응 백엔드 패키지가 최소 동작 가능한 상태(Phase 해당 단계 acceptance 일부 통과)가 된 뒤 착수한다 — 즉 UI 8종을 한 번에 만들지 않고 Phase 1~7 진행과 병행해 하나씩 붙인다.
