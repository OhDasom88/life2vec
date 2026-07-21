# narrative_grounding

**근거**: 계획서 §5 (시계열–서사 양방향 Grounding)
**Phase**: Phase 1 · **범위**: 현재 필수 (§3.1) · **상태**: §5.1/§5.2/§5.3/§5.4 구현 완료 + §5.1 임계값 실측 재보정 완료 + `../ui/data_grounding_curation/`(§5.1/§5.3 두 탭) 실제 GPU·실제 코퍼스로 연결 완료. 남은 건 아래 "알려진 한계"·"남은 작업" 참조

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

## §5.1 서사→시계열 (구현 완료)

[`text_to_window.py`](text_to_window.py) — 967K건 인스턴스를 전부 임베딩하지 않고 2단계로 처리한다.

1. **Time-Text 임베딩 단계**: 80개 narrative 템플릿 텍스트(`normalized_catalog.csv`)만 1회 임베딩(`TemplateEmbeddingIndex`)해 질의와 어떤 템플릿이 의미적으로 가까운지 랭킹. 텍스트 임베딩 모델은 새로 고르지 않고 `scripts/online2_v2/generate_external_embeddings.py`가 이미 채택한 `Qwen/Qwen3-Embedding-0.6B`를 `Qwen3EmbeddingProvider`로 재사용(동일 pooling 규칙). torch/transformers는 `embed()` 최초 호출 시에만 지연 import.
2. **ontology 규칙 단계**: 질의 텍스트에서 `F######` farm_id, `zone N`/`구역 N` 패턴을 정규식으로 추출(`extract_structured_hints`)해, 랭킹된 템플릿의 실제 인스턴스(`sequences.parquet` 행) 중 farm/zone이 일치하는 것을 가점.
3. 두 점수를 0.7:0.3 가중합(`combined_score`)해 `AUTO_ACCEPT_CANDIDATE|REVIEW|QUARANTINE|REJECT`로 분기(`_classify`).

**한계**: 분기 임계값(0.75/0.5/0.3)과 가중치(0.7/0.3)는 실측 라벨 없이 정한 1차 placeholder다. §5.4 `auto_accept_error_and_review_rate`로 사람 검증 라벨이 쌓이면 재보정해야 한다. `Qwen3EmbeddingProvider`는 실제로 실행해보지 않았다(GPU/모델 다운로드 필요) — 테스트는 결정론적 fake provider로 랭킹·분기 로직만 검증했다.

## §5.3 사람 검토 큐 (구현 완료, 일부는 외부 입력 대기)

[`review_queue.py`](review_queue.py) — 7개 배정 기준(§5.3) 중 지금 계산 가능한 것과 아직 다른 모듈이 없어 값을 주입받아야 하는 것을 구분했다(허위로 채우지 않음).

| 기준 | 상태 |
|---|---|
| 모델·규칙·전문가 판정 불일치 | 계산됨 — `confidence["template_expert_confidence"]` vs `sequence_recommendation` 반대 방향 |
| 신규·저빈도 개념 | 계산됨(단, `template_frequency` 테이블을 호출자가 줘야 함) |
| 인과·추천 문장 포함 | 계산됨 — `causal_status`/`recommendation_status`가 기본값을 벗어난 경우만 |
| 근거·반례 동시 존재 | 항상 False — `contradicting_windows`를 채우는 반례 탐지 로직이 아직 없음 |
| 예측 영향도 큼 | `ReviewContext.prediction_impact` 외부 주입 필요(finetune critic 없음) |
| concept split/merge 후보 | `ReviewContext.concept_split_merge_candidate` 외부 주입 필요(`../concept_governance/` 없음) |
| holdout 유사·권한 불명확 | `ReviewContext.split_ambiguous` 외부 주입 필요(split-membership 플러밍 없음) |

우선순위 가중치(`_REASON_WEIGHTS`)도 실측 없이 정한 1차 값이다 — `SPLIT_ACCESS_AMBIGUOUS`만은 계획서 fail-closed 원칙(§0) 때문에 재보정 이후에도 최우선을 유지해야 한다.

## 반례(contradicting_windows) 탐지 (구현 완료)

[`contradictions.py`](contradictions.py) — `contradicting_windows`가 항상 빈 튜플이라던 이전 한계를 해소했다. 새 매칭 규칙을 발명하지 않고 전문가가 이미 각 템플릿에 붙여 둔 `confounders` 텍스트(`normalized_catalog.csv`, 80개 템플릿 전부 비어있지 않음)를 근거로 쓴다: narrative_id가 다른 템플릿의 window가 (a) 같은 farm, (b) 시간 겹침(`temporal_iou > 0`), (c) 그 템플릿 설명 텍스트에 이 narrative의 confounders 키워드가 부분 문자열로 등장 — 세 조건을 모두 만족하면 "경쟁 설명"으로 `contradicting_windows` 후보에 올린다.

실제 데이터(farm F130230, 16,573개 인스턴스)로 검증: A01("실내온도 급격한 점프", confounders="실제 급변;환기")이 같은 시간대의 C01/C02("환기" 관련 원인반응 템플릿)와 매칭되는 것을 확인했다 — 온도 점프가 센서 이상이 아니라 실제 환기 작동 때문일 수 있다는, 전문가가 이미 알고 있던 경쟁 설명이 자동으로 표면화된다.

`augment_with_contradictions(narrative, catalog, candidate_rows)`는 `Narrative`가 frozen이라 원본을 바꾸지 않고 `contradicting_windows`가 채워진 사본을 반환한다. 이제 `../review_queue.py`의 "근거·반례 동시 존재" 기준이 실제로 발동할 수 있다(이전에는 `contradicting_windows`가 항상 비어 있어 죽은 코드였음).

**한계**: 부분 문자열 매칭이라 오탐/누락이 있다(예: "환기"가 category 텍스트에 우연히 등장하면 실제로 무관해도 매칭됨). 그 자체로 "이 narrative는 틀렸다"는 결론이 아니라 사람 검토로 넘기는 신호다. 같은 경쟁 템플릿의 여러 인스턴스가 겹치면 중복 매칭이 그대로 반환된다(템플릿 단위 dedup은 아직 없음, UI 표시 단계에서 그룹핑 필요).

## §5.4 Grounding 평가 (지표 함수 구현 완료, 라벨 데이터는 없음)

[`evaluation.py`](evaluation.py) — Recall@K, Precision@K, temporal IoU, feature-set overlap(Jaccard), farm/zone scope accuracy, cycle consistency, unsupported claim rate, expert acceptance rate, auto-accept 오류율/사람 검토율 9개 지표를 순수 함수로 구현. **사람이 라벨링한 정답 grounding pair는 아직 없다** — 이 함수들은 §5.3 검토 큐를 사람이 실제로 처리하기 시작하면 그 결과를 입력으로 소비할 준비가 된 상태이며, 지금은 합성 데이터로만 검증했다.

## 데이터 계약 (계획서 §4.1)

- `DataWindow`: 시작·종료, 포함 point, 집계·결측·변화점 정보 — `schemas.py`에 구현, `from_online2_corpus.window_from_sequence_row`가 채움
- `Narrative`: 관측·파생 사실·해석·인과 상태·추천 상태·근거 window — `schemas.py`에 구현, `from_online2_corpus.narrative_from_sequence_row`가 채움

## 남은 작업

1. ~~`Qwen3EmbeddingProvider`를 실제로 돌려 임계값 재보정~~ — 완료 (`scripts/online2_v2/calibrate_narrative_grounding_thresholds.py`, 0.90/0.75/0.55).
2. ~~§5.3 검토 큐를 사람이 실제로 처리하는 UI 연결~~ — 완료 (`../ui/data_grounding_curation/`, §5.1 검색 탭도 함께).
3. ~~`contradicting_windows`(반례) 채우는 로직~~ — 완료 (`contradictions.py`).
4. ~~`decisions.jsonl` -> `evaluation.py` 실제 계산 파이프라인 연결~~ — `expert_acceptance_rate` 부분은 완료(`decision_metrics.py`, UI §5.4 탭에서 실시간 반영).
5. ~~§5.1 검색 결과를 §5.3 검토 큐로 보내는 연결~~ — 완료 (`search_to_review.py`, §5.1 탭의 "REVIEW/QUARANTINE을 §5.3 큐로 보내기" 버튼). `decision_metrics.summarize_decisions`가 `grounding_search_confirmation`으로 §5.1 판정 등급별 사람 확인율을 실시간 계산한다.
6. **`evaluation.auto_accept_error_and_review_rate`는 여전히 완전히 연결되지 않았다** — 의도적. 이 지표는 AUTO_ACCEPT_CANDIDATE·REJECT까지 포함한 전체 후보 모집단이 있어야 `human_review_rate`가 의미를 가지는데, `search_to_review.py`는 REVIEW/QUARANTINE만 큐에 올리도록 설계돼 있다(AUTO_ACCEPT는 이미 확신, REJECT는 이미 배제 — §5.1 자체의 존재 이유). 전체 검색 결과를 매번 로깅하는 별도 파이프라인이 있어야 완전히 채워진다 — 다음 우선순위.
7. SAE feature 기반 랭킹 신호는 `../sae/`가 사전학습 이후에나 존재하므로 Phase 4 이후 `text_to_window.py`에 추가.
8. `../ui/data_grounding_curation/README.md`에 정리된 성능 한계(§5.3 배치 로드 ~40초, §5.1 최초 검색 ~15초, 과매칭 반례) 개선.

## 의존성

- 기존: `src/online2/materializers.py`, `src/online2/builder.py`, `src/online2/catalog.py`, `outputs/online2/build-v8-active80-r3/`(빌드 산출물), `scripts/online2_v2/generate_external_embeddings.py`(텍스트 임베딩 모델 선택 재사용)
- 외부 주입 대기: `../concept_governance/`(concept split/merge 신호), finetune critic(예측 영향도), split-membership 플러밍(holdout 유사도)
- 하위 소비자: `sequence_curation/`(REVIEW 판정 입력), `multimodal_pretrain/`(pair 소스)

## Acceptance 연결

B1–B4 (계획서 §17-B) — B1(supporting window/NO_SUPPORT)과 B2(관측·해석·인과 구분)는 `schemas.py`가 코드 레벨로 강제. B3(양방향 retrieval)는 §5.1/§5.2 어댑터로 가능해졌으나 cycle consistency 실측은 아직 없음(§5.4 `cycle_consistency` 함수는 준비됨). B4(자동 승인 오류율·검토율)는 §5.4 `auto_accept_error_and_review_rate`로 계산 가능하나 라벨 데이터 부재로 미실측.

## 중단 조건 연결

Grounding unsupported claim 허용치 초과 시 다음 Phase 진행 금지 (§18)
