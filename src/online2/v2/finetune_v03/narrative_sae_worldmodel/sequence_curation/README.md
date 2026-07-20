# sequence_curation

**근거**: 계획서 §6.2, §6.3 (시퀀스 절단, 학습 전 편집)
**Phase**: Phase 2 · **범위**: 현재 필수 (§3.1) · **상태**: 미착수

## 목적

사전학습 입력 시퀀스를 이벤트 경계를 유지한 채 생성·절단하고, 저품질/오류 서사 시퀀스를 학습셋에서 편집(제외/마스킹 등)한다. CF1S(`counterfactual/cf1s/`)의 편집과 액션 이름이 겹치지만 **목적이 다르다**: CF1S는 미세조정 이후 인과검증용 최소 변경 편집(`core_raw_transaction.py`), 여기는 사전학습 입력 curation용 편집이다.

## 스키마 통합 방침

기존 CF1S `EditTransaction`(`core_raw_transaction.py`의 `ValidatedRawTransaction`)을 두 벌로 만들지 않는다. `edit_purpose: PRETRAIN_CURATION | CAUSAL_VERIFICATION | RL_POLICY` 필드를 기존 스키마에 추가하는 방식으로 통합하고(하위 호환, 기존 CF1S 값은 `CAUSAL_VERIFICATION`), 이 모듈은 `PRETRAIN_CURATION` 값을 생성하는 쪽만 구현한다. `RL_POLICY` 값은 `../edit_policy_rl/`에서 채운다.

## 구현 계획

1. `sequence_cut.py` — 이벤트 경계 유지 절단 (§6.2): max sequence length 초과 시 첫 구간은 앞에서부터, 마지막 구간은 뒤에서부터 제한에 최대한 가깝게 자름. 이벤트 내부 토큰은 임의 절단 금지. 중간 구간이 필요하면 event overlap policy를 별도 version으로 기록.
2. `curation_actions.py` — `INCLUDE|EXCLUDE|MASK|REPLACE|SPLIT|MERGE|REORDER|REWINDOW` 8개 action. 원본을 덮어쓰지 않고 immutable transaction 저장, 편집 전후 sequence hash·편집 이유·적용 규칙·영향받은 split·vocab 영향·재학습 필요성 기록.
3. `review_to_action.py` — `../narrative_grounding/review_queue.py`의 사람 검토 결과(`REVIEW` 판정)를 여기 action으로 변환하는 연결 지점.

## 의존성

- 기존: `cf1s/core_raw_transaction.py` (해시·immutable transaction 패턴 참고)
- 신규: `../narrative_grounding/` (curation 대상 판단 입력)

## Acceptance 연결

A1, A2 (계획서 §17-A)
