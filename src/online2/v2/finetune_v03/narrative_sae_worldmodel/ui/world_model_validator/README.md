# world_model_validator

**근거**: 계획서 §15 · **백엔드**: `../../world_model/` · **상태**: 미착수

## 기능

- rollout 결과 뷰 (multi-step, horizon별 오류)
- uncertainty calibration 뷰
- OOD 탐지 결과 (`world_model`의 5단계 OOD 정책 상태별 분포)
- `world_model.promotion_gate`의 §12.3 승격 기준 7개 항목 체크리스트 — 통과/미통과를 항목별로 표시

## 산출물

readiness report (RL 환경 승격 여부 판정의 근거 문서)

## 구현 계획

1. 승격 기준 7개 항목 대시보드 — 하나라도 미충족이면 화면 상단에 "RL 환경 미승격, bounded edit 보조 scorer로만 사용" 상태를 고정 표시.
2. horizon별 rollout 오류 누적 그래프.
3. §14 실측(55건 중 임계값 초과 인과효과 0건) 요약을 이 화면에 컨텍스트로 항상 노출 — readiness 판정이 이 실측과 모순되지 않는지 사람이 바로 확인 가능하게 한다.
