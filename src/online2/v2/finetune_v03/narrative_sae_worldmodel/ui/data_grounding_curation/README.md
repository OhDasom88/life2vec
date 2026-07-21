# data_grounding_curation UI

**근거**: 계획서 §15 · **백엔드**: `../../narrative_grounding/` · **상태**: §5.1(텍스트→window 검색), §5.3(사람 검토 큐) 두 탭 모두 구현 완료, 실제 GPU/실제 코퍼스로 검증됨

## 실행

```bash
conda run -n life2vec streamlit run \
  src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/data_grounding_curation/app.py
```

`streamlit`은 life2vec conda 환경에 이번에 추가 설치했다(이전엔 없었음). `outputs/online2/build-v8-active80-r3/`(967K sequence 실측 빌드 산출물)가 있어야 동작한다.

## 구성

- [`review_batch.py`](review_batch.py) — §5.3용 Streamlit 의존성 없는 순수 로직(배치 샘플링, 큐 구성, 결정 영속화). 단독 테스트됨(`tests/v2/narrative_sae_worldmodel/test_review_batch.py`).
- `../../narrative_grounding/text_to_window.py`의 `search_text_to_window_over_table` — §5.1용 로직(순수 함수, 별도 UI 레이어 없이 그대로 재사용).
- [`app.py`](app.py) — 위 둘을 감싸는 얇은 Streamlit 렌더링 레이어. `st.tabs`로 두 화면을 분리한다.

## 탭 1 — §5.3 사람 검토 큐 (구현 완료)

- farm_id 선택 → 해당 farm의 전체 인스턴스에서 템플릿당 N개 샘플링 → `narrative_grounding.review_queue.build_review_queue`로 검토 큐 구성
- 반례 탐지(`narrative_grounding.contradictions.augment_with_contradictions`)는 **샘플이 아니라 farm 전체**를 candidate pool로 써서 실제 존재하는 반례를 놓치지 않음
- 검토 화면: observation/derived_state/interpretation/confidence, 근거 window, 반례 window(있으면), 검토 사유(우선순위 코드+detail)
- ACCEPT/REJECT/SKIP 버튼 → `outputs/online2/narrative_grounding_review/decisions.jsonl`에 append-only 기록(같은 narrative를 다시 검토해도 기존 줄을 덮어쓰지 않음, 마지막 줄이 현재 상태)
- 이미 결정된 항목은 재실행해도 큐에서 자동 제외

## 탭 2 — §5.1 텍스트 → window 검색 (구현 완료)

- 자유 텍스트 질의를 받아 `search_text_to_window_over_table`(text_to_window.py) 호출 — 조회 전용, 아무것도 기록하지 않음
- `Qwen3EmbeddingProvider`(실제 모델, `st.cache_resource`로 서버 프로세스당 1회만 로드)로 80개 템플릿을 임베딩한 `TemplateEmbeddingIndex`도 함께 캐시
- 질의에서 farm 힌트(`F######`)를 찾으면 그걸로 967K행 전체 대신 해당 farm만 pyarrow 컬럼 필터로 좁혀 응답 시간을 지킴
- 결과는 결합 점수 내림차순, `🟢 AUTO_ACCEPT_CANDIDATE / 🟡 REVIEW / 🟠 QUARANTINE / 🔴 REJECT` 배지와 함께 expander로 표시(observation, 근거 window, template_similarity/structured_score 분해)
- **life2vec conda 환경에 `transformers`가 이번에 새로 필요해져서 추가 설치**(온라인2-embedding-build 환경과 동일 버전 4.57.6으로 맞춤, `requirements.txt` 반영)

## 실측 검증

`streamlit.testing.v1.AppTest`로 헤드리스 스모크 테스트 완료(브라우저 없이 실제 스크립트 실행, 실제 GPU/실제 967K 코퍼스):

- §5.3: farm F130230(전체 16,573개 인스턴스) 기준 샘플 222개, 검토 큐 154개, ACCEPT 클릭 → 대기 154→153, 예외 없음. 반례 후보가 478개까지 잡히는 케이스 발견(아래 한계 참조).
- §5.1: 질의 "F130230 zone 1에서 실내온도가 급격히 점프했다" → 30개 후보, 전부 `[A01](실내온도 급격한 점프)` 템플릿, combined_score=0.935로 전부 `AUTO_ACCEPT_CANDIDATE`(재보정된 임계값 0.90을 넘음) — 질의 의미와 실제 검색 결과가 정확히 일치함을 확인.

## 알려진 한계

- **§5.3 배치 로드가 느리다(~40초)**: farm 전체(최대 16K+ 인스턴스)를 candidate pool로 반례 탐지에 쓰기 때문. 세션 상태에 캐시해 배치당 1회만 계산하도록 했지만, "배치 로드" 버튼을 누를 때마다 다시 걸린다.
- **§5.1 최초 검색이 느리다(~15초)**: Qwen3-Embedding-0.6B 모델 로드 + 80개 템플릿 임베딩이 최초 1회만 발생(이후 `st.cache_resource`로 즉시 응답). farm 힌트 없는 질의는 랭킹된 템플릿 전체 인스턴스를 필터해야 해서 `max_candidate_rows`(기본 5000)로 응답성을 우선하고 정확도를 일부 희생한다.
- **반례 후보가 과도하게 많이 잡히는 템플릿이 있음**: 실측 중 "1 supporting vs 478 contradicting"인 사례를 발견했다. `contradictions.py`가 부분 문자열 매칭이라 흔한 confounder 키워드(예: 짧은 단어)가 넓게 걸리는 것으로 보인다 — 템플릿 단위 dedup이나 매칭 키워드 특이도 가중치가 다음 개선 대상.
- **결정이 §5.4 지표로 아직 연결되지 않음**: `decisions.jsonl`이 쌓이면 `narrative_grounding.evaluation.expert_acceptance_rate`/`auto_accept_error_and_review_rate`의 실제 입력이 될 수 있으나, 그 파이프라인(로그 -> 지표 계산 -> 리포트)은 아직 연결하지 않았다.
- **§5.1 탭에서 검색한 결과를 §5.3 검토 큐로 보내는 연결이 없다**: 두 탭이 독립적으로 동작한다. 검색 결과 중 애매한 것(REVIEW/QUARANTINE)을 검토 큐에 편입하는 기능은 다음 개선 대상.
