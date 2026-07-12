---
title: "online2 사전학습용 life2vec 코퍼스 구축 파이프라인"
subtitle: "이벤트 단위 세그먼트·멀티모달 임베딩·MLM/OP 반영 개정판"
date: "2026-07-12"
lang: ko-KR
---

# 0. 문서 목적

본 문서는 `agrichallenge/online2`의 공개 데이터 전체를 이용해 life2vec형 사전학습 코퍼스를 구축하기 위한 실행 설계다.

최종 목표는 정확히 10만 개를 맞추는 것이 아니다. 공개 데이터에서 합리적으로 구성할 수 있는 시간·공간·관계·모달리티 맥락을 최대한 빠짐없이 시퀀스로 물질화하고, 기존 life2vec의 **token-level random masking MLM**과 **Order Prediction 이하 OP** 구조에 투입하는 것이다.

공개 사전학습 풀은 다음과 같다.

```text
PUBLIC_PRETRAIN_POOL =
    example_set observations
  + example_set images
  + example_set public interpretations
  + problem_set observations
  + problem_set images
```

`example_set`과 `problem_set`의 차이는 정상·이상 여부가 아니다.

- `example_set`: 정황과 공개 해석이 함께 존재
- `problem_set`: 정황은 존재하지만 최종 해석은 공개되지 않음

숨겨진 problem 해석이나 평가 정답은 사용하지 않는다.

# 1. 계층과 핵심 정의

```text
Cell → Token → Event = Segment → Sequence → Corpus
```

- **Cell**: 원본 CSV의 개별 값
- **Token**: cell 또는 파생 상태를 나타내는 원자 기호
- **Event = Segment**: 특정 시간과 공간 범위에서 관측되거나 정의된 하나의 장면
- **Sequence**: 서사의 중심점에 따라 선택된 이벤트 배열
- **Corpus**: 서로 다른 서사 중심과 맥락을 가진 시퀀스 집합

세그먼트는 이벤트보다 큰 묶음이 아니다. **세그먼트 하나가 이벤트 하나**다.

모든 일반 관측 이벤트는 다음 기준으로 구분한다.

```text
Event identity =
    observation_timestamp
  × spatial_scope
  × event_view
```

같은 timestamp라도 farm 또는 zone이 다르면 기본적으로 다른 지역 이벤트다.

# 2. Cell → Token

## 2.1 수치형 토큰

수치형 변수는 고정된 bin edge registry를 이용해 토큰화한다.

```text
IN_TEMP|ABS_B12
IN_RH|GLOBAL_Q80_90
ROOT_EC|FARM_REL_LOW
VENT_L|ABS_B06
```

권장 원칙:

1. public pretrain pool에서 bin edge를 한 번 적합하고 고정
2. 절대값 bin과 상대 분위수 bin을 구분
3. farm 내 상대 상태가 필요한 경우 별도 상대 토큰 사용
4. 원시값은 sidecar에 보존
5. binning registry의 버전과 적합 출처를 저장

## 2.2 시간·공간 토큰

```text
FARM|F001
ZONE|Z01
HOUR|12
DAY_OFFSET|D07
GAP_H|06
GAP_D|07
```

online2의 기본 신뢰 해상도는 보통 1시간이므로 분 단위의 가상 관측을 보간해 만들지 않는다.

## 2.3 파생 상태 토큰

파생값은 계산 구간과 기준 시점을 명시한다.

```text
DELTA_IN_TEMP|DT_H01|UP_LARGE
RH90_CONTINUOUS|DUR_H04
ROOT_EC_TREND|LOOKBACK_H12|DOWN
```

필수 sidecar:

```text
feature_window_start
feature_window_end
lookback_hours
uses_future_data
```

미래값을 사용한 파생 토큰을 과거 이벤트에 넣지 않는다.

# 3. Event = Segment: 시간·공간 유일성

## 3.1 지역 이벤트

지역 이벤트는 정확히 하나의 farm과 하나의 zone에 귀속된다.

```text
LOCAL_EVENT_KEY =
    observation_timestamp
  + farm_id
  + zone_id
  + event_view
```

예시:

```text
SEG_F1_Z1_T12
[
  FARM|F001
  ZONE|Z01
  HOUR|12
  IN_RH|B80
]

SEG_F2_Z1_T12
[
  FARM|F002
  ZONE|Z01
  HOUR|12
  IN_RH|B70
]
```

농장 1의 습도 80%와 농장 2의 습도 70%는 각각 어느 위치의 정보인지 명시되어야 하므로 별도 이벤트다.

위치 마커 없이 다음처럼 합치면 안 된다.

```text
[IN_RH|B80; IN_RH|B70]
```

같은 시점의 지역 이벤트들은 별도 이벤트로 유지하되 같은 `same_time_group_id`를 갖는다.

## 3.2 공간 집계 이벤트

서사의 중심점이 특정 농장의 값이 아니라 해당 시점의 전체 공간 분포라면 여러 지역 이벤트로부터 집계 이벤트를 만들 수 있다.

```text
SEG_ALL_FARMS_RH_T12
[
  SPATIAL_SCOPE|ALL_FARMS
  MEMBER_COUNT|2
  RH_MEAN|B75
  RH_MIN|B70
  RH_MAX|B80
  RH_RANGE|B10
]
```

이 경우 `[70;80]`과 `[80;70]`은 같은 전반적 분포다.

집계 이벤트 조건:

- 개별 farm·zone 귀속이 서사 목적에 필요하지 않음
- parent 지역 이벤트의 순열에 대해 결과가 불변
- 평균·범위·분위수·히스토그램·count 또는 set pooling 사용
- `parent_segment_ids`와 `aggregation_policy` 보존
- 위치 마커 없는 원시 값 목록 나열 금지

## 3.3 자연어 해석 이벤트

전문가 해석과 같은 자연어는 다수의 subword token으로 직접 펼치지 않고 이미지 이벤트와 유사한 **외부 임베딩 슬롯**으로 처리한다.

```text
SEG_INTERPRETATION
[
  EVENT_KIND|INTERPRETATION
  MODALITY|NATURAL_LANGUAGE
  TEXT_EMBED_SLOT
  INTERPRETATION_SOURCE|EXPERT
]
```

권장 구조:

```text
interpretation text
+ context prompt
→ frozen pretrained LLM
→ pooled contextual embedding
→ trainable projection
→ trainable normalization or gating
→ sequence embedding dimension
```

```text
z_text = LayerNorm(W_text · FrozenLLM(context_prompt; text) + b_text)
```

기본 원칙:

- Frozen LLM 가중치는 고정
- projection·normalization·gating은 학습 가능
- raw text는 sidecar에 보존
- context prompt의 버전과 hash를 저장
- LLM model ID와 embedding reference를 저장
- 자연어 이벤트가 참조하는 관측 이벤트를 `reference_segment_ids`로 보존

필수 메타데이터:

```text
raw_text_ref
text_embedding_ref
llm_model_id
context_prompt_template_id
context_prompt_hash
annotation_timestamp
reference_segment_ids
```

해석 이벤트의 `observation_timestamp`는 연결된 관측 또는 이미지의 기준 시점에 정렬할 수 있다. 실제 해석 작성 시점은 `annotation_timestamp`로 별도 보존한다.

## 3.4 이미지 이벤트

이미지도 개별 이미지 ID를 vocabulary로 만들지 않고 공통 특수 슬롯을 사용한다.

```text
SEG_IMAGE
[
  EVENT_KIND|IMAGE_OBSERVATION
  MODALITY|IMAGE
  IMAGE_EMBED_SLOT
  ALIGN_CONF|MEDIUM
]
```

```text
image
→ frozen pretrained VLM
→ pooled visual embedding
→ trainable projection
→ trainable normalization or gating
→ sequence embedding dimension
```

필수 메타데이터:

```text
image_id
image_embedding_ref
vlm_model_id
alignment_confidence
timestamp_precision
reference_segment_ids
```

VLM은 기본적으로 고정한다. projection 이후 계층과 sequence model은 학습 가능하다.

## 3.5 관계 이벤트

관계 유형 자체가 서사의 한 장면일 때 관계를 별도 이벤트로 표현한다.

```text
SEG_REL
[
  EVENT_KIND|RELATION
  REL_TYPE|CONTEXT_INTERPRETATION
  LEFT_ROLE|CONTEXT
  RIGHT_ROLE|INTERPRETATION
]
```

관계 이벤트는 실제 event ID를 vocabulary token으로 만들지 않는다. 연결 대상은 메타데이터로 관리한다.

```text
linked_segment_ids
relation_anchor_timestamp
relation_scope
```

관계 이벤트의 시간은 다음 중 하나로 명시한다.

- `shared_timestamp`: 연결 이벤트들이 동일 시점
- `later_event_anchor`: 두 이벤트 중 뒤 시점에 귀속
- `reference_observation`: 해석이 참조하는 관측 시점
- `logical_anchor`: 물리적 관측시점이 아닌 관계 배치용 논리 시점

복잡한 token 간·event 간·sequence 간 관계를 모두 명시적 그래프로 만들 필요는 없다. 다양한 서사에서 이벤트를 반복적으로 배치해 관계가 암묵적으로 표현되도록 한다.

시퀀스 전체가 일관된 관계를 가질 때만 background token을 추가한다.

```text
BG_REL|EXAMPLE_CONTEXT_TO_INTERPRETATION
BG_REL|EXAMPLE_PROBLEM_SIMILARITY
BG_SCOPE|CROSS_FARM_SAME_TIME
BG_ORDER|SPARSE_CHRONOLOGICAL
```

background token은 개별 이벤트에 속하지 않고 sequence prefix에 배치한다.

# 4. Event → Sequence

## 4.1 연속성과 비연속성

시퀀스의 이벤트는 연속적일 필요가 없다.

```text
Day 1 12:00 EVENT
→ GAP_D|06
→ Day 7 12:00 EVENT
→ GAP_D|07
→ Day 14 12:00 EVENT
```

짧은 시퀀스도 긴 시간 범위를 나타낼 수 있으므로 다음을 분리해 기록한다.

```text
effective_token_count
covered_time_span_hours
```

권장 길이:

- 16~64: 단일 대비 또는 장기 도약 압축
- 65~256: 국소 반응·일간 프로파일
- 257~640: 다중 이벤트·cross-zone
- 641~2560: 수일·다중 모티프
- 2561~5120: 장기 생육·다중 엔티티 맥락

길이를 채우기 위해 같은 이벤트를 반복하지 않는다.

## 4.2 동일 시점 다중 공간

같은 timestamp에 여러 farm·zone 이벤트가 존재할 수 있다.

```text
TIME_GROUP_T12
[
  SEG_F1_Z1_T12
  SEG_F1_Z2_T12
  SEG_F2_Z1_T12
]
```

각 이벤트는 위치 정보를 보존한다. 이벤트 배열의 내부 순서는 시간 선후 관계가 아니다.

서사의 중심점이 전체 분포라면 이 지역 이벤트들로부터 별도 공간 집계 이벤트를 만들 수 있다.

## 4.3 MLM을 고려한 맥락 구성

기존 life2vec MLM decoder는 mask 위치의 문맥화된 token encoding만 이용해 원래 token을 복원한다.

```text
h_mask = Transformer(sequence)[mask_position]
logits = MLM_decoder(h_mask)
```

복원 시 decoder에 별도의 농장 상태·과거 환경·현재 환경을 추가 입력하지 않는다. 따라서 mask token 추론에 필요한 정보는 **시퀀스 구성 단계에서 이미 포함**되어야 한다.

예를 들어 대상 구역의 현재 actuator token을 복원하려면 서사에 필요에 따라 다음 이벤트들을 포함한다.

- 대상 farm-zone의 이전 환경과 actuator
- 대상 farm-zone의 현재 환경
- 동일 시점의 외기
- 동일 farm의 다른 zone 이벤트
- 유사 외기 조건의 다른 farm 이벤트
- 이후 반응 이벤트

모든 시퀀스가 위 항목을 전부 포함해야 하는 것은 아니다. `narrative_center`가 요구하는 증거를 의도적으로 선택한다.

물질화 단계에서 다음을 기록한다.

```text
maskable_feature_families
context_evidence_roles
context_coverage_score
```

# 5. life2vec 사전학습

## 5.1 MLM

기존 life2vec의 사전학습용 decoder 구조를 그대로 사용한다.

- token 단위 random masking
- mask 위치의 문맥화된 hidden state로 vocabulary token 복원
- 별도의 feature-specific decoder를 기본 구조에 추가하지 않음
- 세부 mask ratio와 random replacement 비율은 기존 설정을 기준으로 실험

기본 MLM 대상은 discrete vocabulary token이다.

`IMAGE_EMBED_SLOT`과 `TEXT_EMBED_SLOT`의 연속 임베딩 값은 기존 categorical decoder가 직접 복원하는 대상이 아니다. 해당 위치의 공통 modality marker가 masking될 수는 있으나 연속 임베딩 자체의 재구성 loss는 기본 life2vec MLM 밖의 선택적 실험으로 둔다.

## 5.2 OP: Arranged·Shuffled·Reversed

OP의 목적은 시퀀스 요약 벡터가 이벤트 배열의 시간적 질서를 충분히 표현하도록 만드는 것이다.

```text
sequence summary vector
→ OP classifier
→ {ARRANGED; SHUFFLED; REVERSED}
```

기존 life2vec의 OP head와 sequence summary 방식을 차용한다.

### 원본 시퀀스

시간 서사의 원본 이벤트는 timestamp 기준 non-decreasing order로 정렬한다.

```text
t1 <= t2 <= ... <= tn
```

동일 timestamp 이벤트는 같은 `same_time_group_id`를 가진다.

### OP target 생성

OP 변형은 개별 이벤트가 아니라 **서로 다른 timestamp의 time group 단위**로 수행한다.

```text
G1(t1) < G2(t2) < G3(t3)
```

- `ARRANGED`: time group을 원래 시간순으로 배치
- `REVERSED`: 서로 다른 time group의 순서를 역순으로 배치
- `SHUFFLED`: arranged와 reversed가 아닌 순열로 배치

같은 time group 내부 이벤트는 shuffle 또는 reverse되어도 시간순 위반이 아니다. 따라서 내부 순서만 바뀐 시퀀스의 target은 계속 `ARRANGED`다.

```text
[G1: A;B;C] < G2
[G1: C;A;B] < G2
```

두 입력은 모두 arranged다.

비지도 OP target 생성기에서 다음을 처리한다.

1. same-time event를 먼저 time group으로 묶음
2. group 내부 순서는 OP label 결정에서 무시
3. group 간 순서만 arranged·shuffled·reversed 판정
4. distinct time group이 1개면 `op_eligible=false`
5. 3-class OP가 필요하면 distinct time group이 최소 3개인 시퀀스를 우선 사용
6. relation event는 anchor time group과 함께 이동
7. sequence-level background token은 prefix에 고정
8. gap token은 time group 변형 정책과 일관되게 재계산하거나 group block에 귀속

관계 중심 시퀀스라도 timestamp 순으로 정렬된 event group이 충분하면 OP에 사용할 수 있다. 반대로 단일 동시간 비교만 포함하면 OP 대상이 아니다.

# 6. Example–Problem 해석 학습

example의 정황·이미지·자연어 해석을 이벤트로 구성한다.

```text
[CONTEXT EVENTS]
→ [IMAGE EVENT]
→ [INTERPRETATION EVENT]
```

정황과 해석의 관계가 핵심이면 관계 이벤트를 추가할 수 있다.

```text
[CONTEXT EVENT]
→ [RELATION EVENT: CONTEXT_INTERPRETATION]
→ [INTERPRETATION EVENT]
```

problem은 숨겨진 해석을 포함하지 않는다.

```text
[PROBLEM CONTEXT EVENTS]
→ [PROBLEM IMAGE EVENT]
→ [INTERPRETATION_QUERY EVENT]
```

example과 problem의 유사성은 다음과 같이 서사로 표현할 수 있다.

```text
[EXAMPLE CONTEXT]
→ [RELATION EVENT: SIMILAR_CONTEXT]
→ [PROBLEM CONTEXT]
```

또는 시퀀스 전체 관계가 일관되면 다음 background token을 사용한다.

```text
BG_REL|EXAMPLE_PROBLEM_SIMILARITY
```

복잡한 관계를 별도의 정답 구조로 모두 명시하지 않고, 여러 narrative에서 event 배치를 달리해 암묵적 관계 표현을 학습한다.

# 7. 생성 우선순위

1. example 정황–이미지–해석 시퀀스
2. problem 정황과 유사 example 시퀀스
3. 동일 farm-zone의 상태 전이
4. actuator 개입 전·후·지연·회복
5. 같은 timestamp의 cross-zone·cross-farm
6. 공간 분포 집계 이벤트의 시간 변화
7. 반복·대비 모티프
8. 생육조사 전후 장기 도약
9. 이미지에서 과거 센서 맥락 역추적
10. 전체 공개 데이터 coverage용 baseline window

단순 임계값 규칙은 검색 seed·전문가 기준 태그·센서 이상 후보·품질검사에 주로 사용한다.

# 8. 권장 manifest

## 8.1 Event manifest

```text
segment_id
event_kind
observation_timestamp
annotation_timestamp
timestamp_precision
time_rank
same_time_group_id
event_view
spatial_scope_type
spatial_scope_id
farm_ids
zone_ids
member_count
aggregation_policy
permutation_invariant
parent_segment_ids
linked_segment_ids
relation_type
relation_anchor_timestamp
source_set
modalities
tokens
external_embedding_refs
raw_text_ref
context_prompt_template_id
context_prompt_hash
alignment_confidence
quality_flags
```

## 8.2 Sequence manifest

```text
sequence_id
narrative_id
narrative_center
background_tokens
segment_ids
order_semantics
op_eligible
distinct_time_group_count
effective_token_count
covered_time_span_hours
entity_count
farm_count
zone_count
local_event_count
aggregate_event_count
relation_event_count
contains_example
contains_problem
contains_interpretation
contains_image
maskable_feature_families
context_evidence_roles
context_coverage_score
sampling_weight
parent_context_id
```

# 9. 검증

1. Segment와 Event가 1:1인지 확인
2. event ID가 시간·공간 범위·view에 대해 유일한지 확인
3. 지역 이벤트가 정확히 하나의 farm과 zone을 가지는지 확인
4. 같은 timestamp라도 farm 또는 zone이 다르면 별도 지역 이벤트인지 확인
5. segment 내부 일반 관측 token의 observation timestamp가 동일한지 확인
6. 공간 집계 이벤트가 permutation invariant인지 확인
7. 집계 통계량이 parent 지역 이벤트와 일치하는지 확인
8. 자연어 해석 event의 raw text·LLM·prompt·embedding provenance가 연결되는지 확인
9. 이미지 event의 VLM embedding reference와 alignment confidence가 존재하는지 확인
10. 관계 이벤트의 linked segment와 anchor semantics가 유효한지 확인
11. sequence background token이 sequence 전체 관계와 일치하는지 확인
12. 시간 서사의 event timestamp가 non-decreasing인지 확인
13. distinct time group timestamp가 strictly increasing인지 확인
14. same-time group 내부 permutation이 OP negative로 분류되지 않는지 확인
15. arranged·shuffled·reversed target이 group 단위로 올바르게 생성되는지 확인
16. distinct time group이 부족한 sequence가 OP에서 제외되는지 확인
17. MLM 입력에 mask 복원에 필요한 맥락이 sequence 안에 존재하는지 점검
18. 기존 MLM decoder 외의 추가 입력이 mask 복원 단계에 주입되지 않는지 확인
19. 1시간 데이터를 분 단위로 보간하지 않았는지 확인
20. gap token과 실제 시간차가 일치하는지 확인
21. problem hidden interpretation이 포함되지 않았는지 확인
22. token 수가 16~5120인지 확인
23. token 수와 covered time span을 분리했는지 확인
24. exact duplicate와 학습 시 random masking augmentation을 구분했는지 확인
25. 공개 farm-zone-time coverage를 보고하는지 확인

# 10. 구현 산출물

```text
events.parquet
sequences.parquet
sequence_segments.parquet
external_embeddings.parquet
binning_registry.json
prompt_registry.json
corpus_statistics.json
validation_report.json
```

학습 데이터에는 다음을 별도로 관리한다.

```text
MLM examples:
  original sequence
  epoch-level token random masking

OP examples:
  arranged sequence
  shuffled time-group sequence
  reversed time-group sequence
```

이 설계의 핵심은 관계를 별도 복잡한 그래프 정답으로 강제하기보다 이벤트와 시퀀스의 반복적 배치를 통해 암묵적으로 학습하고, MLM 복원에 필요한 모든 증거를 sequence materialization 단계에서 구성하는 것이다.
