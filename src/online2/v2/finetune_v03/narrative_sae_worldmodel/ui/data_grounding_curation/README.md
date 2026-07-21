# data_grounding_curation UI

**근거**: 계획서 §15 · **백엔드**: `../../narrative_grounding/` · **상태**: 구현 완료(§5.3 검토 큐 부분) — §5.1 텍스트 질의 검색 화면은 아직

## 실행

```bash
conda run -n life2vec streamlit run \
  src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/data_grounding_curation/app.py
```

`streamlit`은 life2vec conda 환경에 이번에 추가 설치했다(이전엔 없었음). `outputs/online2/build-v8-active80-r3/`(967K sequence 실측 빌드 산출물)가 있어야 동작한다.

## 구성

- [`review_batch.py`](review_batch.py) — Streamlit 의존성 없는 순수 로직(배치 샘플링, 큐 구성, 결정 영속화). 단독 테스트됨(`tests/v2/narrative_sae_worldmodel/test_review_batch.py`).
- [`app.py`](app.py) — 위를 감싸는 얇은 Streamlit 렌더링 레이어.

## 기능 (구현 완료)

- farm_id 선택 → 해당 farm의 전체 인스턴스에서 템플릿당 N개 샘플링 → `narrative_grounding.review_queue.build_review_queue`로 검토 큐 구성
- 반례 탐지(`narrative_grounding.contradictions.augment_with_contradictions`)는 **샘플이 아니라 farm 전체**를 candidate pool로 써서 실제 존재하는 반례를 놓치지 않음
- 검토 화면: observation/derived_state/interpretation/confidence, 근거 window, 반례 window(있으면), 검토 사유(우선순위 코드+detail)
- ACCEPT/REJECT/SKIP 버튼 → `outputs/online2/narrative_grounding_review/decisions.jsonl`에 append-only 기록(같은 narrative를 다시 검토해도 기존 줄을 덮어쓰지 않음, 마지막 줄이 현재 상태)
- 이미 결정된 항목은 재실행해도 큐에서 자동 제외

## 실측 검증

`streamlit.testing.v1.AppTest`로 헤드리스 스모크 테스트 완료(브라우저 없이 실제 스크립트 실행): farm F130230(전체 16,573개 인스턴스) 기준 샘플 222개, 검토 큐 154개, ACCEPT 클릭 → 대기 154→153, 예외 없음. 이 과정에서 반례 후보가 478개까지 잡히는 케이스를 발견했다(아래 한계 참조).

## 알려진 한계

- **배치 로드가 느리다(~40초)**: farm 전체(최대 16K+ 인스턴스)를 candidate pool로 반례 탐지에 쓰기 때문. 세션 상태에 캐시해 배치당 1회만 계산하도록 했지만, "배치 로드" 버튼을 누를 때마다 다시 걸린다. farm 단위가 아니라 사전 인덱싱(예: 시간순 정렬 인덱스)으로 최적화할 여지가 있음 — 지금은 정확성을 우선하고 속도는 다음 단계로 미룸.
- **반례 후보가 과도하게 많이 잡히는 템플릿이 있음**: 실측 중 "1 supporting vs 478 contradicting"인 사례를 발견했다. `contradictions.py`가 부분 문자열 매칭이라 흔한 confounder 키워드(예: 짧은 단어)가 넓게 걸리는 것으로 보인다 — 템플릿 단위 dedup이나 매칭 키워드 특이도 가중치가 다음 개선 대상.
- **§5.1(텍스트 질의로 window 검색) 화면은 아직 없다**: 지금 UI는 §5.3(큐 기반 검토)만 다룬다. `narrative_grounding.text_to_window.search_text_to_window`를 호출하는 별도 탭/화면이 필요.
- **결정이 §5.4 지표로 아직 연결되지 않음**: `decisions.jsonl`이 쌓이면 `narrative_grounding.evaluation.expert_acceptance_rate`/`auto_accept_error_and_review_rate`의 실제 입력이 될 수 있으나, 그 파이프라인(로그 -> 지표 계산 -> 리포트)은 아직 연결하지 않았다.
