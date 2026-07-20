# concept_governance

**근거**: 계획서 §11 (개념 분할·병합 및 재학습)
**Phase**: Phase 5 · **범위**: 현재 필수 (§3.1, 관리 체계 자체) · **상태**: 미착수 (기존 `cf1s/core_locks.py` 원칙 확장)

## 목적

SAE 특징/ontology/event/vocab/데이터 규칙 변경을 버전 관리하고, 변경 유형별로 적절한 재학습 범위를 판정한다. SAE dictionary 크기 증가만으로 분할·병합이 해결됐다고 판정하지 않는다.

## 변경 유형별 기본 대응 (§11.1)

| 유형 | 예 | 기본 대응 |
|---|---|---|
| 해석 특징 변경 | SAE 고온 특징 세분화 | SAE 재학습·probe |
| Ontology 변경 | HIGH_VPD를 생육단계별 분리 | ontology version·재매핑 |
| Event 변경 | 이벤트 조건·window 변경 | 영향 시퀀스 재생성 |
| Vocab 변경 | token 분할·병합·bin 변경 | 재토큰화·embedding 재학습 |
| 데이터 규칙 변경 | 전체 시퀀스 생성 규칙 변경 | continual/full pretraining 판정 |

## 운영상 중대 영향 — vocab 변경 비용

Vocab 변경은 CF1S의 `code_tree_sha256`/`policy_tree_sha256`에는 영향 없지만 `tokenizer_sha256`/`vocab_sha256`를 바꿔 **기존 55건(Dev3 3 + Primary32 32 + Validation20 20) 전체의 stable lock을 무효화**한다. `governance_workflow.py`의 승인 단계에서 이 재인증 비용을 실행 전에 반드시 수치로(예상 소요시간) 보고해야 한다.

## 구현 계획

1. `change_classifier.py` — 변경 5유형 분류.
2. `split_merge_candidates.py` — 후보 탐지 규칙 6종(§11.2): 반대 prediction effect / 생육단계·시간대별 의미 분리 / 하나의 feature에 여러 개념 혼합 / 하나의 개념이 여러 feature에 분산 / 전문가 불일치 반복 / 두 개념이 조건·활성·예측효과에서 지속적으로 동일.
3. `governance_workflow.py` — 승인 절차: 후보 탐지 → 근거 보고서 → 전문가/개발 승인 → ontology/vocab/SAE 새 version → 영향 데이터 재생성 → 필요 범위 재학습 → 이전 version과 회귀 비교 → (vocab 변경 시) CF1S 55건 재인증.
4. `vocab_impact_estimator.py` — vocab 변경이 55건 stable lock에 미치는 영향 사전 시뮬레이션, 재인증 소요시간 추정 리포트 생성.

## 의존성

- 기존: `cf1s/core_locks.py`(해시 잠금 원칙, `REQUIRED_STABLE_LOCK_SHA_FIELDS`)
- 신규: `../sae/`(분할·병합 후보 신호원)

## 리스크

vocab 변경 승인은 55건 전체 재인증을 유발하는 고비용 결정이다. 재인증 비용 보고 없이 승인 워크플로우를 통과시키지 않는다.
