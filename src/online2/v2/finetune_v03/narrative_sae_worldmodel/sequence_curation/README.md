# sequence_curation

**근거**: 계획서 §6.2, §6.3 (시퀀스 절단, 학습 전 편집)
**Phase**: Phase 2 · **범위**: 현재 필수 (§3.1) · **상태**: 구현 완료(§6.2 절단 알고리즘, §6.3 편집 트랜잭션, §5.3→§6.3 연결), 104개 테스트 통과

## 목적

사전학습 입력 시퀀스를 이벤트 경계를 유지한 채 생성·절단하고, 저품질/오류 서사 시퀀스를 학습셋에서 편집(제외/마스킹 등)한다. CF1S(`counterfactual/cf1s/`)의 편집과 액션 이름이 겹치지만 **목적이 다르다**: CF1S는 미세조정 이후 인과검증용 최소 변경 편집(`core_raw_transaction.py`), 여기는 사전학습 입력 curation용 편집이다.

## ⚠ 착수 전 점검 결과 — 두 가지가 원래 계획과 달라졌다

1. **§6.2 절단 알고리즘은 실제로 신규였다.** life2vec의 핵심이 "긴 이벤트 시퀀스를 max_length로 자르는" 작업이라 이미 어딘가 있을 거라 예상하고 찾아봤지만, 정확히 이 알고리즘(긴 시퀀스를 여러 청크로 **분할** — 첫 청크는 앞에서부터, 마지막은 뒤에서부터, 중간은 버전 관리되는 overlap 정책)은 없었다. 가장 가까운 선례는 둘 다 "하나의 윈도우만 만드는" 절단이었다: `src/tasks/base.py`의 `Task.clip_document()`(뒤에서부터 채우는 단일 절단, `accumulate(reversed(lengths))` 패턴을 여기서 재사용)와 `src/online2/builder.py`의 `emit_template_sequence`(토큰 예산 초과 시 앞에서 이벤트를 버리는 단일 절단). 둘 다 여러 청크로 나누지는 않는다.
2. **스키마 통합 방침을 바꿨다.** 원래 계획은 CF1S `ValidatedRawTransaction`에 `edit_purpose` 필드를 추가해 통합하는 것이었다. 착수 전 확인 결과 이건 위험하다 — CF1S 해시(`canonical_validated_transaction_sha` 등)는 55건(Dev3 3 + Primary32 32 + Validation20 20) stable lock의 무결성 근거이고, frozen dataclass에 필드 하나만 추가해도 재직렬화 시 해시가 달라져 55건 전체 재인증을 유발할 수 있다(`concept_governance/README.md`가 vocab 변경에 대해 이미 문서화한 것과 같은 종류의 리스크를, 이번엔 CF1S 코드 자체를 건드려서 만들게 된다). 그래서 CF1S 프로덕션 파일은 건드리지 않고, 같은 설계 원칙(불변 트랜잭션, 해시 체인, fail-closed)을 공유하는 **별도** 스키마를 만들었다 — `edit_purpose` enum으로 구분하자던 문구와 다른 선택이지만, CF1S의 hash-lock 안정성을 지키기 위한 의도적 이탈이다.

## §6.2 시퀀스 절단 (구현 완료)

[`sequence_cut.py`](sequence_cut.py) — `cut_into_bounded_chunks(events, max_length, overlap_policy_id=..., overlap_events=...)`. 이벤트(`TokenEvent`, 내부 토큰은 절대 쪼개지 않는 최소 단위) 목록을 받아 `SequenceChunk`(`SOLE|FIRST|MIDDLE|LAST`) 목록으로 분할한다. 이벤트 하나가 `max_length`를 넘으면 잘라내지 않고 통째로 포함한 뒤 `SINGLE_EVENT_EXCEEDS_MAX_LENGTH` 플래그를 남긴다(online2 builder.py의 `quality_flags` 관례와 동일).

## §6.3 학습 전 편집 (구현 완료)

[`curation_actions.py`](curation_actions.py) — `CurationTransaction` + `build_curation_transaction(...)`. `INCLUDE|EXCLUDE|MASK|REPLACE|SPLIT|MERGE|REORDER|REWINDOW` 8개 action, 액션별 필수 `parameters` 키를 `__post_init__`에서 검증한다. 해시는 새로 만들지 않고 `src/online2/canonical.py`의 `stable_id`/`canonical_json`을 재사용(온라인2 코퍼스와 동일한 네임스페이스 해싱). `transaction_id`는 (sequence_id, action, before_sha, after_sha, reason, applied_rule_ref)의 해시라 같은 편집을 두 번 만들면 같은 ID가 나온다(멱등).

## §5.3 → §6.3 연결 (구현 완료)

[`review_to_action.py`](review_to_action.py) — `narrative_grounding`의 `decisions.jsonl` 레코드를 curation transaction으로 승격한다: `ACCEPT` → `INCLUDE`(토큰 불변), `REJECT` → `EXCLUDE`(토큰 비움). `SKIP`은 아직 결정이 아니므로 트랜잭션을 만들지 않는다(`None` 반환). decisions.jsonl에는 실제 토큰이 없으므로(§5.3 UI는 narrative 객체만 다룸) 토큰 조회는 호출자 책임 — `transactions_from_decisions(decisions, tokens_by_sequence_id=..., affected_split=...)`가 못 찾은 항목은 조용히 건너뛴다.

## 아직 없는 것

- 위 세 조각을 실제로 실행하는 배치 스크립트/UI(`tokens_by_sequence_id`를 online2 `sequences.parquet`/`event_tokens.parquet`에서 실제로 채워 넣는 부분)는 아직 없다. 지금은 순수 함수 계층만 구현·검증됐다.
- `SequenceChunk`를 실제 `multimodal_pretrain/` 입력 파이프라인에 연결하는 부분도 아직 없다(그 모듈 자체가 미착수).

## 의존성

- 기존: `src/online2/canonical.py`(해시 유틸 재사용), `src/tasks/base.py`(절단 패턴 참고, 코드 재사용은 아님)
- 신규: `../narrative_grounding/`(curation 대상 판단 입력)
- CF1S(`counterfactual/cf1s/core_raw_transaction.py`)는 참고만 하고 직접 의존하지 않음(위 "착수 전 점검 결과" 참조)

## Acceptance 연결

A1, A2 (계획서 §17-A)
