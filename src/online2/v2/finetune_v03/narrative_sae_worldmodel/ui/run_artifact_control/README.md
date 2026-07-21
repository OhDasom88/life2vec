# run_artifact_control

**근거**: 계획서 §15 · **백엔드**: W&B run/sweep, 기존 `outputs/cf1s_core/` 아티팩트 구조 · **상태**: 로컬 run(`run_manifest_v2.json`) 목록·config diff·지표 비교는 [`../pipeline_explorer/`](../pipeline_explorer/README.md) 4번 탭으로 구현됨(이번 세션 run들이 전부 `--no-wandb`로 실행돼 W&B 자체가 없음). **W&B 연동, §4.2 추적 키 체인 트리 뷰(아래 "구현 계획")는 미착수.**

## 기능

- W&B run/sweep 목록 및 lineage 뷰
- artifact hash 체인 조회 (raw_point_id → ... → edit_transaction_id, §4.2)
- selection manifest 조회

## 산출물

run/selection manifest

## 구현 계획

1. 기존 `outputs/cf1s_core/` 아티팩트 디렉토리 구조를 그대로 인덱싱하는 방식으로 시작(별도 저장소 새로 만들지 않음).
2. §4.2 추적 키 체인을 트리 뷰로 시각화 — `raw_point_id → window_id → event_id → narrative_id → token_span_id → sequence_id → checkpoint_id/layer/position → sae_id/feature_id → prediction_id → edit_transaction_id`.
3. `event_id` 이후 사슬은 이미 CF1S `canonical_json_sha256` 체인으로 구현돼 있으므로, 이 UI가 신규로 시각화해야 할 구간은 `narrative_id`, `sae_id/feature_id` 두 축뿐이라는 점을 우선순위에 반영.
