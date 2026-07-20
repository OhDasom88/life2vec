# edit_policy_lab

**근거**: 계획서 §15 · **백엔드**: `../../edit_policy_rl/` + 기존 CF1S quarantine/promotion 뷰어 · **상태**: 미착수

## 기능

- 편집 diff 뷰 (raw/event/token/SAE/output, `edit_policy_rl.sae_linkage` 출력)
- reward 분해 뷰 (soft score 8종 + hard constraint 결과)
- gate 통과 이력 (OOD 정책 5단계 중 어디서 걸렸는지)
- policy 비교 (여러 policy 후보의 편집 결과 나란히 비교)

## 산출물

edit transaction

## 구현 계획

1. 기존 CF1S quarantine/promotion 결과 뷰어를 확장하는 방식으로 시작(신규 뷰어를 처음부터 만들지 않음).
2. `edit_policy_rl.hard_constraint`의 REJECT 사유를 reward 그래프와 분리된 별도 패널로 표시(soft score와 섞이지 않게 — hard constraint 위반은 음수 reward가 아니라 즉시 차단이라는 계획서 원칙을 UI가 반영).
3. `NOVEL_FEATURE_COMBINATION` 분류된 편집 결과는 전문가 검토 대기 상태로 별도 표시.
