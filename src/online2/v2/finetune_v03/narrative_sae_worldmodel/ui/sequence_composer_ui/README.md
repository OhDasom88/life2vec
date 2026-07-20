# sequence_composer_ui

**근거**: 계획서 §15 · **백엔드**: 기존 `cf1s/core_raw_transaction.py` + `../../sequence_curation/` · **상태**: 미착수

## 기능

이벤트 단위 include/exclude/mask/split/merge 조작 화면. `sequence_curation.curation_actions`의 8개 action(`INCLUDE|EXCLUDE|MASK|REPLACE|SPLIT|MERGE|REORDER|REWINDOW`)을 사람이 직접 트리거할 수 있게 노출한다.

## 산출물

sequence transaction (immutable, 편집 전후 hash 포함)

## 구현 계획

1. 시퀀스 타임라인 뷰 (이벤트 경계 표시).
2. action 트리거 UI + 편집 이유 입력 필드(필수 — `sequence_curation.curation_actions`가 기록하는 편집 이유).
3. 편집 이력 조회(immutable transaction 로그).

## 주의

이 UI는 `edit_purpose=PRETRAIN_CURATION` 편집만 다룬다. `CAUSAL_VERIFICATION`(CF1S 기존) 편집 UI는 이미 존재하는 CF1S quarantine/promotion 뷰어를 그대로 쓰고 여기서 중복 구현하지 않는다.
