# narrative_grounding

**근거**: 계획서 §5 (시계열–서사 양방향 Grounding)
**Phase**: Phase 1 · **범위**: 현재 필수 (§3.1) · **상태**: 부분 구현 — §5.2(시계열→서사)는 어댑터 계층까지 완료, §5.1(서사→시계열)/§5.3/§5.4는 미착수

## ⚠ 착수 전 점검 결과 — §5.2와 §6은 이미 대부분 구현되어 있었다

착수 전 `narratives/v8` 산출물을 확인한 결과, 이 모듈이 "완전 신규"라는 최초 가정은 틀렸다. 실제로는:

- `/data/datasets/agrichallenge/online2/narratives/normalized_catalog_v8.csv` — 80개 ACTIVE 서사 템플릿 카탈로그(윈도우 정의, 시작·종료 조건, threshold rule, agronomic_interpretation 등 계획서 §5.2 필드와 거의 1:1 대응)
- `src/online2/materializers.py`의 `NarrativeTemplate`/`MaterializerRegistry` — 이 카탈로그를 원시 데이터에 매칭하는 규칙 기반 매처(이미 구현)
- `src/online2/builder.py`의 `CorpusBuilder` — raw point → cell → atomic value → event → sequence까지 `stable_id` 해시 사슬로 물질화(계획서 §4.2 추적 키 사슬과 동일한 설계)
- `outputs/online2/build-v8-active80-r3/` — 위 파이프라인을 실제로 돌려 **967,012개 sequence를 이미 빌드해 둔 실측 산출물**(`sequences.parquet` 등)

즉 §5.2(시계열→서사, 규칙 기반 생성)와 §6(이벤트·토큰·시퀀스 생성)의 핵심 로직은 이미 완성되어 실행까지 끝난 상태였다. 이 사실을 모르고 처음부터 다시 설계했다면 967K건 규모의 중복 파이프라인을 만들 뻔했다.

## 실제로 구현한 것 (이번 착수분)

기존 `src/online2` 산출물을 **다시 만들지 않고** 계획서 스키마로 감싸는 어댑터 계층만 추가했다.

- [`schemas.py`](schemas.py) — 계획서 §4.1/§5.2의 `DataWindow`/`Narrative` dataclass. `window_id`는 신규 발급 없이 기존 `sequence_id`를 재사용(추적 키 사슬 중복 방지). `Narrative.__post_init__`이 §17 `B1`(모든 narrative는 supporting window 또는 NO_SUPPORT를 가짐)과 "근거 없으면 인과성 주장 금지"를 코드 레벨로 강제.
- [`from_online2_corpus.py`](from_online2_corpus.py) — `sequences.parquet` 한 행 + `normalized_catalog.csv` 템플릿 행 → `Narrative`. `outputs/online2/build-v8-active80-r3/`의 실제 967K행 중 3,000개 무작위 샘플(67/80 템플릿 커버)에 대해 예외 없이 동작 확인.
- [`tests/v2/narrative_sae_worldmodel/test_narrative_grounding.py`](../../../../../../../tests/v2/narrative_sae_worldmodel/test_narrative_grounding.py) — 합성 fixture 기반 회귀 테스트 6건(윈도우 시각 역산, NOT_ESTABLISHED 기본값, 다운그레이드 플래그→REVIEW, 근거 없는 인과성 주장 차단 등).

### 알려진 한계 (다음 사람이 바로 봐야 할 것)

- `sequence_recommendation`(`INCLUDE|EXCLUDE|REVIEW`) 판정이 현재는 `quality_flags`에 다운그레이드 플래그(`CATALOG_MAX_EVENTS_APPLIED` 등) 존재 여부만 보는 매우 단순한 휴리스틱이다. 실측 샘플(3,000건)에서 `REVIEW` 2,816건, `INCLUDE` 184건, `EXCLUDE` 0건으로 편중됨 — 튜닝되지 않은 1차 규칙이므로 §5.3 검토 큐 설계 시 반드시 재검토할 것.
- `DataWindow.start_timestamp`는 `narrative_center - covered_time_span_hours`로 역산한 근사값이다. `sequence_segments.parquet`(개별 이벤트 timestamp)까지 조인하면 더 정확해진다.
- `interpretation` 필드는 인스턴스별로 새로 생성한 해석이 아니라 템플릿 저자가 미리 써 둔 `agronomic_interpretation` 텍스트를 그대로 복사한 것이다 — §5.2가 요구하는 "가능한 해석"의 최소 구현이며, 인스턴스 데이터(실제 값 크기 등)를 반영한 해석 생성은 아직 없다.

## 아직 신규로 구현해야 하는 것 (§5.1, §5.3, §5.4)

`src/online2`가 커버하지 못하는 부분은 실제로 신규다:

- **§5.1 서사→시계열**: 임의의 자연어 서사 문장이 입력으로 들어왔을 때 근거 window를 검색하는 기능. `src/online2`는 반대 방향(카탈로그 규칙 → 매칭)만 하며, 텍스트 질의 기반 검색·랭킹(ontology 규칙, Time-Text 임베딩, 전문가 mapping 결합)은 아직 없다. 임베딩 모델/인덱스 선택 등 별도 설계가 필요한 큰 작업.
- **§5.3 사람 검토 큐**: 위 "알려진 한계"의 `sequence_recommendation` 휴리스틱을 대체/보강하는 우선순위 큐. 7개 배정 기준(§5.3) 미구현.
- **§5.4 Grounding 평가**: Recall@K, Precision@K, temporal IoU, cycle consistency, unsupported claim rate 등. 미구현.

## 데이터 계약 (계획서 §4.1)

- `DataWindow`: 시작·종료, 포함 point, 집계·결측·변화점 정보 — `schemas.py`에 구현, `from_online2_corpus.window_from_sequence_row`가 채움
- `Narrative`: 관측·파생 사실·해석·인과 상태·추천 상태·근거 window — `schemas.py`에 구현, `from_online2_corpus.narrative_from_sequence_row`가 채움

## 다음 구현 순서

1. (완료) `schemas.py`, `from_online2_corpus.py` — 위 참조
2. `text_to_window.py` (§5.1, 서사→시계열) — 임의 서사 문장을 입력받아 `DataWindow` 후보를 검색·랭킹. ontology 규칙 + Time-Text 임베딩 + 전문가 mapping 결합, `AUTO_ACCEPT_CANDIDATE|REVIEW|QUARANTINE|REJECT` 분기. SAE feature 랭킹 신호는 `../sae/`가 사전학습 이후에나 존재하므로 1차 구현에서는 제외하고 Phase 4 이후 추가.
3. `review_queue.py` (§5.3) — 사람 검토 우선순위 규칙 7종. `sequence_recommendation` 휴리스틱을 대체/보강하는 실질적 판정 로직.
4. `evaluation.py` (§5.4) — Recall@K, Precision@K, temporal IoU, cycle consistency, unsupported claim rate 등.

## 의존성

- 기존: `src/online2/materializers.py`, `src/online2/builder.py`, `src/online2/catalog.py`, `outputs/online2/build-v8-active80-r3/`(빌드 산출물)
- 신규: 없음 (2번부터)
- 하위 소비자: `sequence_curation/`(REVIEW 판정 입력), `multimodal_pretrain/`(pair 소스)

## Acceptance 연결

B1–B4 (계획서 §17-B)

## 중단 조건 연결

Grounding unsupported claim 허용치 초과 시 다음 Phase 진행 금지 (§18)
