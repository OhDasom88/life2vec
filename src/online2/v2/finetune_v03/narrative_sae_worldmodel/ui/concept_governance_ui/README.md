# concept_governance_ui

**근거**: 계획서 §15 · **백엔드**: 기존 `cf1s/core_locks.py` + `../../concept_governance/` · **상태**: 미착수

## 기능

- N:M mapping 뷰 (concept ↔ SAE feature/narrative)
- split/merge 후보 목록 (`concept_governance.split_merge_candidates` 출력)
- 승인 워크플로우 UI (`concept_governance.governance_workflow` 각 단계 진행 상태)

## 산출물

ontology/mapping version

## 구현 계획

1. 변경 유형 5종(해석 특징/ontology/event/vocab/데이터 규칙)별 후보 리스트.
2. **vocab 변경 승인 화면에는 `concept_governance.vocab_impact_estimator`의 재인증 소요시간 추정치를 승인 버튼과 같은 화면에 강제 표시** — 이 수치 없이 승인 액션을 노출하지 않는다.
3. 승인 이력(누가/언제/근거 보고서) 조회.
