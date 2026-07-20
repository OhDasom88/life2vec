# data_grounding_curation UI

**근거**: 계획서 §15 · **백엔드**: `../../narrative_grounding/` · **상태**: 미착수

## 기능

- 서사↔시계열 양방향 조회 (`narrative_grounding.text_to_window` / `window_to_text` 결과 뷰)
- 근거 window와 반대 근거(contradicting_windows) 동시 표시
- `../../narrative_grounding/review_queue.py`의 우선순위 큐 — 사람 검토 화면 (`AUTO_ACCEPT_CANDIDATE`/`REVIEW`/`QUARANTINE`/`REJECT` 분기 표시 및 사람 override)

## 산출물

grounding decision (사람 검토 결과, immutable). 이 UI에서의 결정은 dataset version을 고정하기 전에만 반영한다 — optimizer step 중 학습 데이터 변경 금지(§5.3).

## 구현 계획

1. `narrative_grounding.schemas`의 `Narrative`/`DataWindow`를 읽기 전용으로 렌더링하는 뷰.
2. 검토 큐 리스트 + 개별 케이스 상세(근거/반례/confidence) + 승인·반려 액션.
3. 액션 결과를 `narrative_grounding.review_queue`가 기대하는 형식으로 기록.
