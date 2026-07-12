# online2 life2vec 코퍼스 물질화 프롬프트 v7

당신은 스마트팜 딸기 재배 데이터 분석가이자 life2vec형 이벤트 시퀀스 설계 전문가다.

## 목표

`agrichallenge/online2`의 공개 데이터 전체를 사용하여 합리적으로 구성 가능한 이벤트와 시퀀스를 최대한 빠짐없이 물질화하라. 정확히 10만 개를 맞추지 말라.

## 계층

```text
Cell → Token → Event = Segment → Sequence → Corpus
```

세그먼트 하나는 이벤트 하나다.

## 이벤트 식별

일반 지역 이벤트:

```text
timestamp × farm_id × zone_id × event_view
```

같은 timestamp라도 farm 또는 zone이 다르면 별도 이벤트다. 동일 시점 지역 이벤트는 같은 `same_time_group_id`를 공유한다.

공간 전체 분포가 서사의 중심일 때만 parent 지역 이벤트로부터 permutation-invariant aggregate event를 생성한다.

## 자연어 해석 이벤트

전문가 해석을 raw subword sequence로 직접 넣지 말고 다음 구조를 사용하라.

```text
[TEXT_EMBED_SLOT]
```

```text
raw interpretation + versioned context prompt
→ frozen LLM
→ trainable projection and normalization
→ sequence embedding dimension
```

저장:

```text
raw_text_ref
text_embedding_ref
llm_model_id
context_prompt_template_id
context_prompt_hash
annotation_timestamp
reference_segment_ids
```

## 이미지 이벤트

```text
[IMAGE_EMBED_SLOT]
```

```text
image → frozen VLM → trainable projection and normalization
```

개별 image ID를 vocabulary token으로 만들지 말고 metadata와 embedding reference로 보존하라.

## 관계 표현

관계가 한 장면이면 relation event로 생성하라.

```text
EVENT_KIND|RELATION
REL_TYPE|...
```

연결 segment ID는 metadata에 둔다.

복잡한 token·event·sequence 관계는 다양한 narrative 배치로 암묵적으로 표현하라. 시퀀스 전체가 일관된 관계를 가질 때만 prefix background token을 추가하라.

예:

```text
BG_REL|EXAMPLE_CONTEXT_TO_INTERPRETATION
BG_REL|EXAMPLE_PROBLEM_SIMILARITY
BG_SCOPE|CROSS_FARM_SAME_TIME
```

## 시퀀스 구성

이벤트는 연속적일 필요가 없다. 실제 gap을 기록하라.

MLM에서 mask token 복원 시 decoder가 받는 직접 입력은 mask 위치의 contextualized hidden state뿐이다. 따라서 복원에 필요한 과거·현재·동일시점·타구역·타농장 맥락을 sequence 구성 단계에서 포함하라.

각 sequence에 기록:

```text
maskable_feature_families
context_evidence_roles
context_coverage_score
```

## MLM

기존 life2vec pretraining decoder를 그대로 사용한다.

- token-level random masking
- mask position hidden state로 vocabulary token 복원
- 기본 구조에 별도 context input을 추가하지 않음
- image/text continuous embedding reconstruction은 기본 MLM 대상이 아님

## OP

OP label:

```text
ARRANGED
SHUFFLED
REVERSED
```

원본 sequence는 event timestamp가 non-decreasing하도록 정렬한다.

같은 timestamp 이벤트는 time group으로 묶는다. OP 변형과 label 판정은 time group 간 순서로만 수행한다.

- group 내부 shuffle 또는 reverse만 발생하면 ARRANGED
- group 순서가 역전되면 REVERSED
- arranged·reversed가 아닌 group permutation은 SHUFFLED
- distinct time group 1개는 OP 제외
- 3-class OP에는 distinct time group 3개 이상 우선
- relation event는 anchor group과 함께 이동
- background token은 prefix에 고정

## Example과 Problem

```text
example = context + image + public interpretation
problem = context + image without hidden interpretation
```

example 정황·이미지·해석 관계와 example–problem 정황 관계를 학습하라. problem hidden interpretation은 절대 포함하지 말라.

## 출력

### Event manifest

```text
segment_id
event_kind
observation_timestamp
annotation_timestamp
timestamp_precision
same_time_group_id
event_view
spatial_scope_type
spatial_scope_id
farm_ids
zone_ids
parent_segment_ids
linked_segment_ids
relation_type
source_set
tokens
external_embedding_refs
raw_text_ref
context_prompt_template_id
context_prompt_hash
alignment_confidence
quality_flags
```

### Sequence manifest

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
contains_example
contains_problem
contains_interpretation
contains_image
maskable_feature_families
context_evidence_roles
context_coverage_score
sampling_weight
```

## 필수 검증

1. event와 segment가 1:1
2. 지역 이벤트는 정확히 하나의 farm-zone
3. 같은 시간 다른 공간은 별도 이벤트
4. aggregate event는 permutation invariant
5. 자연어 event의 frozen LLM embedding provenance 보존
6. image event의 frozen VLM embedding provenance 보존
7. relation event의 linked segment 유효
8. sequence background relation과 narrative center 일치
9. timestamp non-decreasing
10. same-time 내부 순열은 ARRANGED
11. OP target은 time-group 단위 생성
12. MLM은 token random masking
13. mask 복원에 별도 외부 context input을 주입하지 않음
14. 필요한 context는 materialized sequence 내부에 존재
15. problem hidden interpretation 미포함
16. token 수 16~5120
17. 1시간 데이터를 분 단위로 보간하지 않음
18. 공개 데이터 coverage 보고
