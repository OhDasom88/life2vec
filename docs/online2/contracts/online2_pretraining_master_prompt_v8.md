# online2 Neo4j 코퍼스 구축 및 life2vec 사전학습 통합 실행 프롬프트 v8

당신은 스마트팜 데이터 엔지니어, Neo4j 그래프 모델러, life2vec 사전학습 파이프라인 개발자다.

이 작업은 설계안만 제시하는 작업이 아니다. 저장소와 데이터를 조사한 뒤, 아래 요구사항을 만족하는 코드·설정·테스트·산출물을 실제로 구현하고 검증하라. 중간 단계의 검증이 실패하면 원인을 해결하기 전 다음 단계로 넘어가지 말라.

## 1. 최종 목표

`datasets/agrichallenge/online2`의 공개 데이터를 사용하여 다음 전체 파이프라인을 구축하라.

```text
Online2 public raw data
→ CellOccurrence
→ shared AtomicValue
→ CategoricalToken
→ Event = Segment
→ SameTimeGroup
→ Narrative Sequence
→ Corpus
→ Neo4j
→ deterministic training export
→ Online2 Source
→ Vocabulary
→ Corpus/DataModule
→ life2vec MLM + SOP pretraining
```

최종 구현은 다음을 모두 만족해야 한다.

1. Neo4j에 존재하는 online1 데이터를 전부 삭제하고 online2 데이터로 교체한다.
2. 원본 CSV의 cell 발생과 피처별 원자값을 손실 없이 Neo4j에 저장한다.
3. 원자값과 categorical token의 대응 관계를 저장한다.
4. token이 어떤 event에 속하고 event가 어떤 sequence를 구성하는지 저장한다.
5. 이벤트·시퀀스 생성 규칙과 registry 버전을 포함한 provenance를 저장한다.
6. Neo4j의 online2 그래프를 결정론적 학습 manifest로 export한다.
7. 기존 life2vec의 Source, Vocabulary, Corpus, DataModule, MLM/SOP task 및 설정을 online2에 맞게 조정한다.
8. 동일한 관측시점의 이벤트는 내부 순서가 shuffle 또는 reverse되어도 SOP에서 `ARRANGED`로 판정한다.
9. 구현한 파이프라인으로 최소 smoke pretraining까지 실행하고 결과를 보고한다.

정확히 10만 개의 sequence를 맞추는 것이 목표가 아니다. 공개 데이터에서 합리적으로 구성 가능한 시간·공간·관계·모달리티 맥락을 최대한 보존하는 것이 목표다.

## 2. 기준 문서와 기존 구현

먼저 다음 파일을 읽고 서로의 요구사항과 현재 구현 차이를 정리하라.

### online2 기준

- `datasets/agrichallenge/online2/narratives/online2_corpus_build_plan_v7.md`
- `datasets/agrichallenge/online2/narratives/online2_sequence_materialization_prompt_v7.md`
- `datasets/agrichallenge/online2/online2_narrative_catalog.csv`
- `scripts/generate_online2_narrative_catalog.py`

### online1 Neo4j 참고 구현

- `datasets/agrichallenge/online1/register_narratives_neo4j.py`
- `datasets/agrichallenge/online1/materialize_narrative_instances.py`
- `datasets/agrichallenge/online1/token_binning_strategy.md`
- `datasets/agrichallenge/online1/sequence_pretrain_strategy.md`

online1 스크립트는 참고 패턴일 뿐 그대로 재사용하지 말라. 기존 스크립트는 online1 원시 그래프 전체 삭제, 대용량 `UNWIND` 적재, online2 provenance를 지원하지 않는다.

### life2vec 참고 구현

- `src/data_new/sources/base.py`
- `src/data_new/sources/agri_environment.py`
- `src/data_new/sources/agri_growth.py`
- `src/data_new/sources/hackathon_environment.py`
- `src/data_new/vocabulary.py`
- `src/data_new/datamodule.py`
- `src/data_new/types.py`
- `src/data_new/augment.py`
- `src/tasks/base.py`
- `src/tasks/mlm.py`
- `src/transformer/models.py`
- `conf/data_new/sources.yaml`
- `conf/data_new/corpus/`
- `conf/data_new/population/`
- `conf/data_new/vocabulary/`
- `conf/datamodule/`
- `conf/task/agri_mlm.yaml`
- `conf/experiment/pretrain_agri_multicorpus.yaml`

online2 전용 구현을 추가하되 기존 online1·hackathon 파이프라인을 깨뜨리지 말라. 공통 로직을 수정할 경우 기존 동작에 대한 회귀 테스트를 추가하라.

## 3. 절대 준수사항

### 3.1 공개 데이터와 누수 방지

사용 가능한 사전학습 풀은 다음뿐이다.

```text
PUBLIC_PRETRAIN_POOL =
    example_set observations
  + example_set images
  + example_set public interpretations
  + problem_set observations
  + problem_set images
```

- problem_set의 숨겨진 해석이나 평가 정답을 사용하지 말라.
- example_set의 `score90` 공개 해석은 context-interpretation event로 사용할 수 있고, 공개 관측만으로 구성한 example-problem relation도 허용한다. 단, problem_set의 hidden 또는 corrected interpretation을 생성·추론·사용하지 말라.
- 둘째 생육 조사 이후의 값을 과거 시점 파생 token에 사용하지 말라.
- 모든 파생값에 `feature_window_start`, `feature_window_end`, `lookback_hours`, `uses_future_data`를 기록하라.
- `uses_future_data=true`인 파생 token을 과거 event에 넣지 말라.
- example의 `score90` 해석을 주 해석으로 사용하라.
- `score50`, `score70`은 `quality_score`, source path, answer tier를 명시한 보조·대조 narrative에서만 사용하라.
- score50·70을 score90과 합쳐 하나의 정답처럼 취급하거나 problem label로 사용하지 말라.
- M06의 reference answer는 encoder의 일반 categorical 정답 token이나 problem label로 사용하지 말라.

### 3.2 데이터 보간과 진단 표현

- 1시간 관측을 분 단위 관측으로 보간하지 말라.
- 실제 관측값, 파생값, 집계값을 구분하라.
- 질병·장해 후보를 관측만으로 확진하지 말라.
- 이미지의 촬영시점이나 zone이 불명확하면 추측하지 말고 `timestamp_precision`, `alignment_policy`, `alignment_confidence`, `quality_flags`에 불확실성을 기록하라.

### 3.3 재현성

- 모든 ID는 정규화된 natural key와 명시된 schema version으로 만든 stable hash여야 한다.
- Python의 런타임 `hash()`를 사용하지 말라.
- timezone, timestamp precision, null, float, decimal, boolean, string canonicalization 규칙을 문서화하고 테스트하라.
- 입력 파일 목록, 크기, checksum, row count, schema, `active_template_count`, normalized catalog 및 auxiliary/RAG/validation registry hash, prompt hash, code version을 실행 manifest에 기록하라.
- random 작업은 seed를 설정하고 seed를 산출물에 기록하라.
- 같은 입력과 설정으로 두 번 실행했을 때 ID, node count, relationship count, manifest checksum이 같아야 한다.

## 4. 먼저 수행할 현황 점검

코드를 변경하기 전에 다음을 확인하고 짧은 baseline 보고서를 남겨라.

1. online2 원본 파일 수, 컬럼, dtype, row count, farm/zone/timestamp 범위
2. `(farm_id, zone_id, timestamp)` 및 생육 조사 키의 중복과 결측
3. example/problem farm 분리와 answer tier 목록
4. 이미지 파일과 사용 가능한 정렬 metadata
5. 현재 Neo4j URI, database, 인증 환경변수
6. 현재 Neo4j label, relationship type, constraint, index, node/relationship count
7. 현재 프로젝트의 Python 의존성 관리 방식
8. 현재 online2 관련 코드·캐시·산출물 존재 여부
9. 현재 life2vec 입력 시간 해상도, sequence 최대 길이, vocabulary 생성 방식

인증정보와 비밀번호를 로그나 산출물에 출력하지 말라.

## 5. Neo4j 삭제 안전 계약

사용자가 요구한 최종 상태는 기존 DB의 online1 데이터를 전부 삭제하고 online2만 적재한 상태다. 단, 파괴적 작업은 반드시 다음 순서를 따른다.

### 5.1 기본 동작

- 삭제 스크립트의 기본값은 `dry_run=true`로 구현하라.
- 실제 삭제에는 `--execute`와 대상 database 이름, 예상 dataset 확인값을 모두 요구하라.
- database가 예상 database와 다르거나 online1 signature가 확인되지 않으면 즉시 중단하라.
- 삭제 전에 다음 내용을 JSON으로 저장하라.
  - database 이름
  - labels와 label별 count
  - relationship type별 count
  - constraints와 indexes
  - online1 catalog와 주요 natural key 표본
  - 실행시각과 code version

### 5.2 실제 삭제

사용자의 명시적 실행 플래그가 있을 때 online1 node만 부분적으로 남기지 말고 대상 database의 기존 graph node와 relationship을 전부 제거하라.

```cypher
MATCH (n)
CALL {
  WITH n
  DETACH DELETE n
} IN TRANSACTIONS
```

Neo4j 버전이 위 문법을 지원하지 않으면 안전한 batch delete를 구현하라. 메모리 한계를 피하기 위해 단일 거대 transaction을 사용하지 말라.

기존 schema object는 node 삭제로 사라지지 않는다. `SHOW CONSTRAINTS`, `SHOW INDEXES` 결과를 기준으로 online1 전용 constraint/index를 명시적으로 제거하고, lookup/system index는 건드리지 말라. 이름만 추측해서 schema object를 삭제하지 말라.

### 5.3 삭제 완료 조건

- 사용자 node count = 0
- relationship count = 0
- online1 전용 constraint/index = 0
- pre-delete report와 post-delete report가 존재

하나라도 만족하지 않으면 online2 적재를 시작하지 말라.

## 6. online2 Neo4j 그래프 모델

다음 계층을 구현하라.

```text
(:Dataset)
  └─(:SourceFile)
      └─(:SourceRow)
          └─(:CellOccurrence)
              ├─[:HAS_ATOMIC_VALUE]→(:AtomicValue)
              └─[:ENCODED_AS]→(:CategoricalToken)

(:AtomicValue)-[:MAPS_TO]→(:CategoricalToken)
(:Event)-[:CONTAINS_CELL]→(:CellOccurrence)
(:Event)-[:HAS_TOKEN]→(:CategoricalToken)
(:SameTimeGroup)-[:HAS_EVENT]→(:Event)
(:Sequence)-[:HAS_EVENT]→(:Event)
(:Narrative)-[:MATERIALIZES]→(:Sequence)
(:Corpus)-[:CONTAINS_SEQUENCE]→(:Sequence)
```

필요에 따라 `BinningRegistry`, `TokenizationPolicy`, `PromptTemplate`, `ExternalEmbedding`, `ValidationRun`, `BuildRun` node를 추가하라.

Neo4j는 relationship에 relationship을 연결할 수 없다. mapping 정책을 표현하기 위해 불가능한 relationship-to-relationship 모델을 만들지 말라. 관계에는 `policy_id`, `registry_version`, `rule_hash`를 저장하고, 해당 ID를 가진 정책 node를 별도로 연결하라.

모든 online2 node에는 최소한 다음 공통 metadata를 둔다.

```text
dataset_tag = online2
schema_version
build_id
created_at
source_checksum 또는 derivation_hash
```

### 6.1 CellOccurrence

원본 CSV의 cell 발생 건마다 하나를 만든다.

필수 속성:

```text
cell_id
dataset_tag
source_file_id
source_row_id
column_name
feature_alias
raw_type
raw_value 또는 typed value
raw_display
is_null
farm_id
zone_id
observation_timestamp
timestamp_precision
modality
```

`cell_id`는 source file checksum, source row identity, column name으로 만든 stable ID여야 한다. 원본 row 번호만 단독 ID로 사용하지 말라.

### 6.2 AtomicValue

`AtomicValue`는 공유 사전 계층이다. 동일한 피처와 동일한 원자값이 반복되면 하나의 node를 공유한다.

natural key:

```text
(feature_name, raw_type, canonical_raw_value, schema_version)
```

필수 속성:

```text
atomic_value_id
feature_name
feature_alias
raw_type
canonical_raw_value
raw_display
unit
schema_version
```

원자값은 피처 단위의 실제 값을 그대로 보존하라. float를 bin 번호로 덮어쓰거나 문자열 round-trip이 불가능한 방식으로 저장하지 말라. null과 NaN, boolean, integer, float, decimal, datetime, string을 구분하라.

### 6.3 CategoricalToken

필수 속성:

```text
token_id
token_string
token_category
feature_name
tokenization_kind
registry_version
rule_hash
is_special
is_background
is_maskable
```

예:

```text
FARM|F001
ZONE|Z01
HOUR|12
IN_TEMP|ABS_B12
IN_RH|GLOBAL_Q80_90
ROOT_EC|FARM_REL_LOW
FCU_FAN|ON
IMAGE_EMBED_SLOT
TEXT_EMBED_SLOT
```

개별 raw value, image ID, event ID, sequence ID를 vocabulary token으로 만들지 말라.

### 6.4 Mapping 관계

최소 관계:

```text
(CellOccurrence)-[:HAS_ATOMIC_VALUE]->(AtomicValue)
(CellOccurrence)-[:ENCODED_AS]->(CategoricalToken)
(AtomicValue)-[:MAPS_TO]->(CategoricalToken)
```

관계 속성:

```text
policy_id
registry_version
rule_hash
bin_lower
bin_upper
closed_side
relative_scope
fit_population_hash
derived
```

원자값 하나가 ABS, GLOBAL, FARM_REL 등 여러 token view로 매핑될 수 있으므로 mapping kind를 명확히 구분하라.

## 7. Tokenization registry

### 7.1 수치형

- public pretrain pool에서 bin edge를 한 번 적합하고 고정하라.
- ABS, GLOBAL quantile, FARM_REL quantile을 분리하라.
- 모든 edge와 closed interval 의미를 저장하라.
- 중복 edge, 상수 피처, 극단값, null 처리 규칙을 구현하라.
- fit source file checksum과 fit population hash를 저장하라.
- registry가 존재하면 임의로 재적합하지 말고 명시적 새 version을 만들라.

### 7.2 categorical·제어값

- 원본 코드북을 우선 사용하라.
- `0→OFF`, `201→ON` 같은 매핑은 확인된 피처에만 적용하라.
- 201을 모든 categorical 피처의 오류값으로 처리하지 말라.
- 미해독 코드는 `UNKNOWN_CODE|<canonical>` 형태 또는 정책에 정의한 categorical token으로 보존하고 quality flag를 남겨라.

### 7.3 파생 token

파생 token은 원본 token과 구분하고 다음을 기록하라.

```text
derivation_name
derivation_version
feature_window_start
feature_window_end
lookback_hours
uses_future_data
parent_cell_ids
parent_event_ids
```

### 7.4 registry 산출물

최소 다음을 생성하라.

```text
binning_registry.json
tokenization_registry.json
prompt_registry.json
build_manifest.json
```

파일 registry와 Neo4j registry node의 version/hash가 일치해야 한다.

## 8. Event = Segment 물질화

Event와 Segment는 1:1이다.

### 8.1 Local observation event

기본 identity:

```text
observation_timestamp
× farm_id
× zone_id
× event_view
× schema_version
```

같은 timestamp라도 farm이나 zone이 다르면 별도 event다. local event는 정확히 하나의 farm과 하나의 zone에 속해야 한다.

필수 metadata:

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
source_set
modalities
quality_flags
```

Event에는 다음 두 provenance를 모두 연결하라.

1. `CONTAINS_CELL`: 원본 cell 발생과의 연결
2. `HAS_TOKEN`: 모델 입력 token과의 연결

두 관계에 deterministic position과 role을 기록하라.

### 8.2 SameTimeGroup

동일한 관측 기준시각을 가진 event를 `SameTimeGroup`으로 묶는다.

```text
same_time_group_id
anchor_timestamp
timestamp_precision
logical_rank
```

- 동일 timestamp의 local event 내부 순서는 시간 선후를 의미하지 않는다.
- event가 여러 sequence에 재사용되어도 timestamp identity는 유지하라.
- 자연어·이미지·relation event는 명시된 anchor semantics에 따라 적절한 group에 귀속하라.

### 8.3 Aggregate event

공간 전체 분포가 narrative의 중심일 때만 만든다.

- parent local event의 순열에 결과가 불변이어야 한다.
- 평균, 범위, 분위수, histogram, count, set pooling 등을 사용하라.
- 위치 marker 없는 raw value 목록을 나열하지 말라.
- `parent_segment_ids`, `aggregation_policy`, `member_count`, `permutation_invariant=true`를 기록하라.
- parent 순서를 바꾼 property test로 결과가 같음을 검증하라.

### 8.4 Image event

개별 image ID를 vocabulary로 만들지 말고 `IMAGE_EMBED_SLOT`을 사용하라.

```text
image_id
image_ref
image_embedding_ref
vlm_model_id
embedding_status
alignment_confidence
alignment_policy
timestamp_precision
zone_id
reference_segment_ids
```

frozen VLM embedding을 실제로 생성할 수 있으면 model ID와 checksum을 기록하라. 모델이나 리소스가 없으면 임의 embedding을 만들지 말고 `embedding_status=pending`과 명확한 검증 실패 또는 제한사항을 남겨라.

이미지를 날짜 범위나 생육 기간 수준으로만 정렬할 수 있으면 `timestamp_precision=period`로 기록하고 사용한 period-level `alignment_policy`와 `alignment_confidence`를 함께 저장하라. zone 근거가 없으면 `zone_id=null`로 두고 특정 zone을 추정하거나 여러 zone에 복제 귀속하지 말라.

### 8.5 Interpretation event

raw subword 전체를 categorical sequence에 직접 펼치지 말고 `TEXT_EMBED_SLOT`을 사용하라.

```text
raw_text_ref
text_embedding_ref
llm_model_id
context_prompt_template_id
context_prompt_hash
answer_tier
quality_score
annotation_timestamp
reference_segment_ids
embedding_status
```

frozen LLM embedding과 trainable projection/normalization의 경계를 명확히 하라. raw text, prompt, model ID, embedding checksum을 추적 가능하게 하라.

### 8.6 Relation event

관계가 하나의 장면일 때만 relation event로 생성하라.

```text
event_kind = RELATION
relation_type
linked_segment_ids
relation_anchor_timestamp
relation_scope
anchor_semantics
```

relation event는 실제 event ID를 vocabulary token으로 만들지 않는다. anchor time group과 함께 이동해야 한다.

## 9. Narrative catalog 정합화

`online2_narrative_catalog.csv`를 정규화한 catalog를 사용하되, 현재 카탈로그를 맹목적으로 신뢰하지 말라. 학습 template 수는 normalized catalog 전체를 파싱하여 `status=ACTIVE`인 행만 세어 동적으로 계산하고, 코드·설정·문서에 특정 개수를 하드코딩하지 말라. v7과 다음 차이를 먼저 해소하라.

1. `farm_qbin_00_09`, `b05` 예시를 versioned token registry와 통일
2. `same_time_group_id`, `time_rank`, `op_eligible` 추가
3. 실제 시간차에 맞는 `GAP_H`, `GAP_D` 처리
4. aggregate event의 parent와 permutation invariance 추가
5. image/text/relation event와 embedding provenance 추가
6. sequence-level background token 규칙 추가
7. 카탈로그의 256 token 검증과 v7의 16~5120 범위 충돌 해소
8. M06이 일반 hourly event/sequence 정의로 떨어지는 분기 오류 수정
9. cross-zone·cross-farm same-time narrative가 local event identity를 잃지 않도록 수정
10. trigger row count와 실제 unique sequence instance count를 구분

수정된 catalog 또는 별도 normalized catalog를 versioned 산출물로 저장하고 변경 사유를 기록하라. 각 항목의 `status`를 명시하고, 파싱한 전체 행 수와 ACTIVE 행 수를 검증하여 `active_template_count`로 기록하라.

ACTIVE 항목만 학습 sequence materialization 대상이다. auxiliary registry는 ACTIVE template이 참조하는 보조 context·대조 자료의 provenance를, RAG registry는 검색·RAG 평가 전용 항목을, validation registry는 검증 규칙·fixture·평가 항목을 관리한다. 순수 RAG 또는 validation 항목은 normalized 학습 catalog에서 `ACTIVE`로 두지 말고 학습 instance를 생성하지 말라.

## 10. Sequence 물질화

각 Sequence는 narrative 중심에 따라 선택된 event 배열이다.

필수 metadata:

```text
sequence_id
narrative_id
narrative_center
background_tokens
segment_ids
same_time_group_ids
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
quality_flags
```

`op_eligible=true`는 `order_semantics`가 `STRICT_CHRONOLOGICAL`, `SPARSE_CHRONOLOGICAL`, `CHRONOLOGICAL_WITH_TIES` 중 하나이고 `distinct_time_group_count >= 2`일 때만 허용한다. `RELATION_ORDERED`, `SET_COMPARISON`, `RETRIEVAL_PAIR` 및 period-level logical alignment sequence는 항상 `op_eligible=false`다.

### 10.1 기본 순서

- 원본 sequence의 time group은 timestamp 기준 strictly increasing이어야 한다.
- event 전체 timestamp 배열은 non-decreasing이어야 한다.
- 동일 time group 내부 event의 배열 순서는 시간 의미가 없다.
- 결정론적 export를 위해 내부 tie-breaker는 둘 수 있으나 SOP label에는 사용하지 말라.
- sequence가 비연속이면 실제 gap을 기록하라.
- 길이를 채우기 위해 event를 복제하지 말라.
- exact duplicate sequence를 제거하고, epoch-level random masking augmentation과 구분하라.

### 10.2 길이

물질화 허용 범위는 v7을 따른다.

```text
16 ≤ effective_token_count ≤ 5120
```

현재 `agri_mlm.yaml`의 `max_length=2048`와 충돌을 숨기지 말라. 다음 중 데이터와 GPU 메모리에 맞는 방식을 구현하고 설정에 명시하라.

1. online2 task의 max length를 5120까지 확장
2. 길이 bucket별 corpus/task 설정
3. narrative 의미를 보존하는 명시적 window 분할

어떤 경우에도 datamodule이나 task에서 조용히 잘라 sequence 의미를 손상시키지 말라. clipping이 발생하면 원 sequence ID, window ID, retained event 범위를 기록하라.

### 10.3 Sequence 관계

Neo4j 관계:

```text
(Narrative)-[:MATERIALIZES]->(Sequence)
(Sequence)-[:HAS_EVENT {
  position,
  time_group_rank,
  position_in_group,
  evidence_role
}]->(Event)
(Corpus)-[:CONTAINS_SEQUENCE]->(Sequence)
```

sequence prefix의 background token은 event가 아니며 SOP 변환 시 항상 prefix에 고정한다.

## 11. Parquet/JSON 산출물

Neo4j 적재와 별개로 다음 deterministic 산출물을 생성하라.

```text
cell_occurrences.parquet
atomic_values.parquet
cell_token_mappings.parquet
events.parquet
event_tokens.parquet
same_time_groups.parquet
sequences.parquet
sequence_segments.parquet
external_embeddings.parquet
binning_registry.json
tokenization_registry.json
prompt_registry.json
corpus_statistics.json
validation_report.json
build_manifest.json
```

각 Parquet에는 schema version을 기록하라. list column과 relation table의 cardinality가 Neo4j와 일치해야 한다.

`build_manifest.json`에는 normalized catalog에서 계산한 `active_template_count`와 normalized catalog, auxiliary registry, RAG registry, validation registry 각각의 hash를 담은 `registry_hashes`를 포함하라.

Neo4j를 학습 batch마다 실시간 조회하지 말라. Neo4j는 provenance와 검증 가능한 source of truth로 사용하고, 학습에는 build ID로 고정된 deterministic Parquet/HDF5 export를 사용하라.

## 12. Neo4j 적재 구현 요구사항

- 프로젝트의 의존성 관리 방식을 확인하고 공식 `neo4j` Python driver를 정식 의존성으로 추가하라.
- URI, database, user, password는 환경변수로 받고 코드에 하드코딩하지 말라.
- `MERGE`와 uniqueness constraint를 사용하라.
- constraint와 index를 적재 전에 생성하고 준비 완료를 기다려라.
- `UNWIND $rows`와 제한된 transaction batch를 사용하라.
- 55개 farm을 무제한 병렬 적재하지 말라.
- retry 가능한 transient error와 definitive auth error를 구분하라.
- 로그에는 batch, label, inserted/matched count, elapsed time만 남기고 민감정보는 제외하라.
- full rebuild와 scoped rebuild를 구분하라.
- 부분 실패 후 재실행해도 duplicate가 생기지 않아야 한다.

최소 uniqueness constraint:

```text
Dataset.dataset_id
SourceFile.source_file_id
SourceRow.source_row_id
CellOccurrence.cell_id
AtomicValue.atomic_value_id
CategoricalToken.token_id
BinningRegistry.registry_id
Event.segment_id
SameTimeGroup.same_time_group_id
Narrative.narrative_id
Sequence.sequence_id
Corpus.corpus_id
BuildRun.build_id
```

필요한 timestamp, farm, zone, feature, token category index를 실제 query pattern에 맞게 추가하라.

## 13. life2vec 전처리 통합

### 13.1 Online2 TokenSource

기존 `TokenSource` 계약을 재사용하거나 명시적으로 확장하여 online2 source를 추가하라.

Source는 검증된 build export에서 다음을 제공해야 한다.

```text
PERSON_ID 또는 stable sequence entity ID
START_DATE
AGE 또는 시간 상대 위치
SENTENCE 구성용 token fields
segment_id
same_time_group_id
event_kind
event_position
time_group_rank
sequence_id
```

기존 `Corpus.sentences()`가 field를 문자열로 합치면서 metadata를 버리는 문제를 해결하라. metadata가 `PersonDocument`까지 전달되도록 하되 기존 source와 호환성을 유지하라.

### 13.2 시간 해상도

현재 `Corpus.combined_sentences()`는 `START_DATE`를 reference date 기준 일수로 바꾼다. online2의 1시간 event가 같은 날짜라는 이유로 모두 같은 시간 위치가 되어서는 안 된다.

- online2용 시간 단위 또는 명시적 absolute position encoder를 구현하라.
- observation timestamp와 same-time group을 혼동하지 말라.
- 생육 조사일, hourly observation, logical relation anchor의 precision 차이를 보존하라.

### 13.3 Population과 split

- farm-zone 또는 sequence entity를 stable population ID로 매핑하라.
- 현재 agri sanity population처럼 train/val/test에 동일 ID를 넣어 평가 누수를 만들지 말라.
- split 목적을 정의하고 farm 또는 case group 단위의 deterministic split을 사용하라.
- example/problem semantics가 split 과정에서 훼손되지 않도록 하라.
- public pool 전체 fit이 필요한 bin registry와 모델 평가 split을 구분해 문서화하라.

### 13.4 Vocabulary

online2 vocabulary는 frozen token registry를 기준으로 생성하라.

- special token: `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`, `[UNK]` 및 실제 필요한 placeholder
- sequence background token
- farm/zone/time/gap token
- feature categorical token
- derived token
- `IMAGE_EMBED_SLOT`, `TEXT_EMBED_SLOT`

token ID는 재실행 시 안정적이어야 한다. 단순 corpus 출현 순서에 따라 바뀌지 않도록 정렬·version 규칙을 둔다.

다음을 vocabulary에 넣지 말라.

- raw numeric value
- cell ID
- image ID
- event/segment ID
- sequence ID
- embedding reference

OOV 정책, 최소 빈도 정책, registry에는 있으나 split에 없는 token 정책을 테스트하라.

### 13.5 Corpus와 DataModule

- sequence 경계를 보존하라.
- sequence별 event와 same-time group metadata를 보존하라.
- narrative별 sampling weight를 지원하라.
- 긴 sequence의 bucket 또는 window 정책을 지원하라.
- build ID와 registry version이 다르면 기존 cache를 재사용하지 말라.
- `_arguments` cache validation이 변경된 설정에서 조용히 return하는 기존 경로를 점검하고 stale cache를 사용하지 않게 하라.
- online2용 Hydra source, population, corpus, vocabulary, datamodule, task, experiment 설정을 추가하라.
- 모델 `vocab_size`를 하드코딩하지 말고 실제 vocabulary size와 일치시키라.

## 14. MLM

기존 life2vec categorical MLM head를 유지한다.

- token-level random masking
- mask 위치의 contextualized hidden state로 원 token 복원
- decoder에 별도의 farm 상태나 외부 context vector를 우회 입력하지 않음
- 복원에 필요한 과거·현재·동일시점·타구역·타농장 맥락은 materialized sequence 안에 포함
- `[SEP]`, `[PAD]`, 비복원 대상 slot의 masking 정책을 명시
- image/text 연속 embedding 자체는 기본 categorical MLM reconstruction 대상이 아님

현재 `mlm_mask()`가 legal token 수보다 많은 수를 sampling할 가능성, general/background token masking, dtype, padding target 처리를 함께 점검하고 online2 짧은 sequence에서도 안전하게 동작하도록 테스트하라.

## 15. SOP: time-group-aware order prediction

label은 다음과 같다.

```text
0 = ARRANGED
1 = REVERSED
2 = SHUFFLED
```

### 15.1 핵심 판정 규칙

SOP의 순서 단위는 개별 event가 아니라 서로 다른 관측시점의 `SameTimeGroup`이다.

```text
G1(t1) < G2(t2) < ... < Gn(tn)
```

- `ARRANGED`: group이 원래 시간순
- `REVERSED`: 서로 다른 group의 순서가 완전히 역순
- `SHUFFLED`: arranged와 reversed가 아닌 group 순열
- 동일 group 내부 event만 shuffle/reverse된 경우: 반드시 `ARRANGED`
- `order_semantics`가 `STRICT_CHRONOLOGICAL`, `SPARSE_CHRONOLOGICAL`, `CHRONOLOGICAL_WITH_TIES` 중 하나이고 distinct group이 2개 이상일 때만 `op_eligible=true`
- distinct group 1개, `RELATION_ORDERED`, `SET_COMPARISON`, `RETRIEVAL_PAIR`, period-level logical alignment: `op_eligible=false`, SOP loss에서 제외
- 3-class SOP: distinct group 3개 이상인 sequence를 우선 사용
- distinct group 2개에서는 가능한 non-arranged 순열이 reverse뿐이므로 shuffled label을 만들지 말라.

### 15.2 현재 코드의 필수 수정

현재 `src/tasks/mlm.py`의 `cls_task()`는 `document.sentences`만 reverse/shuffle한다. 그 결과 `abspos`, `age`, `segment`와 sentence가 어긋나고 same-time event도 잘못된 negative label이 된다.

다음을 구현하라.

1. `PersonDocument`에 `event_ids`, `same_time_group_ids`, 필요 order metadata를 추가
2. sentence와 정렬된 모든 배열을 하나의 event record 또는 공통 permutation helper로 관리
3. group block 단위 reverse/shuffle
4. `sentences`, `abspos`, `age`, `segment`, event ID, group ID, modality reference를 같은 permutation으로 이동
5. relation event는 anchor group과 함께 이동
6. background prefix는 이동하지 않음
7. gap token은 변환된 group 순서에 맞춰 재계산하거나 group block에 명시적으로 귀속
8. 원본 `PersonDocument`를 in-place 변형하여 validation 재사용 시 누적 변형이 생기지 않게 방어
9. validation/test의 SOP 변형과 seed 정책을 결정론적으로 설정
10. `target_cls` 외에 SOP eligibility 또는 loss mask를 batch에 전달

모델 loss는 SOP 비대상 sample을 제외할 수 있어야 한다. 모든 sample을 억지로 arranged로 학습시켜 class distribution을 왜곡하지 말라.

### 15.3 필수 SOP 테스트

다음 예를 모두 자동화하라.

```text
[G1:A,B,C] [G2:D]                 → ARRANGED
[G1:C,A,B] [G2:D]                 → ARRANGED
[G1:C,B,A] [G2:D]                 → ARRANGED
[G2:D] [G1:A,B,C]                 → REVERSED
[G2] [G1] [G3]                    → SHUFFLED
[G3] [G2] [G1]                    → REVERSED
[G1 내부 event만 존재]            → SOP 제외
```

각 테스트에서 모든 event-aligned 배열이 동일하게 이동했는지도 검증하라.

## 16. 외부 embedding 통합

이미지와 해석의 연속 embedding을 sequence model에 넣으려면 다음 경계를 구현하라.

```text
frozen VLM/LLM output
→ stored external embedding
→ trainable projection
→ trainable normalization 또는 gating
→ sequence hidden dimension
```

- frozen encoder parameter는 optimizer에 포함하지 말라.
- projection 이후 계층은 학습 가능해야 한다.
- slot 위치와 embedding row가 sequence clipping·shuffle 후에도 정렬되어야 한다.
- embedding dimension, model ID, preprocessing version, checksum을 검증하라.
- 외부 embedding이 없는 event는 명시적 missing policy를 사용하라.
- categorical marker token만 존재하면서 실제 embedding이 없는 상태를 정상 멀티모달 학습으로 보고하지 말라.

리소스 제약으로 외부 encoder 실행이 불가능하면 categorical·센서 사전학습 smoke test와 멀티모달 준비 상태를 분리해 보고하라.

## 17. 테스트 및 검증

기존 저장소에 단위 테스트 체계가 부족하면 프로젝트 관례에 맞는 테스트 디렉터리와 pytest 설정을 추가하라.

### 17.1 단위 테스트

- raw typed value canonicalization과 exact round-trip
- CellOccurrence/AtomicValue stable ID
- source row와 cell cardinality
- bin edge 경계값과 null/constant feature
- token registry ID 안정성
- event identity와 local farm-zone cardinality
- aggregate permutation invariance
- same-time group 생성
- sequence gap 계산
- future leakage guard
- score tier 정책
- group-aware SOP label
- aligned-array permutation
- MLM legal mask 수
- vocabulary OOV와 ID 안정성

### 17.2 통합 테스트

작은 fixture로 다음 end-to-end를 검증하라.

```text
CSV cell
→ CellOccurrence
→ AtomicValue
→ CategoricalToken
→ Event
→ SameTimeGroup
→ Sequence
→ Neo4j
→ Parquet export
→ TokenSource
→ Vocabulary
→ Corpus
→ encoded MLM/SOP batch
```

추가 검증:

- 같은 fixture를 두 번 적재해 node/relationship count 불변
- Neo4j와 Parquet cardinality 일치
- orphan node와 dangling relationship 없음
- event token position 연속
- sequence event position 연속
- original sequence timestamp non-decreasing
- distinct group timestamp strictly increasing
- same-time 내부 모든 permutation이 arranged
- problem hidden interpretation 0건
- future-derived token 0건
- stale cache 거부

### 17.3 전체 데이터 검증

v7의 검증 항목 전체를 `validation_report.json`으로 구현하라. 최소 다음 통계를 포함한다.

```text
input file/row/cell counts
node counts by label
relationship counts by type
token counts by category
event counts by kind
sequence counts by narrative
sequence length distribution
covered time span distribution
distinct time group distribution
SOP eligibility/class distribution
farm-zone-time coverage
example/problem coverage
image/text embedding coverage
quality flag counts
duplicate/orphan/leakage counts
```

validation 결과를 `ValidationRun` node에도 저장하고 build ID와 연결하라.

## 18. 학습 실행

다음 순서로 실행하라.

1. online2 source/corpus/vocabulary prepare
2. train/val/test encoded dataset 생성
3. vocabulary size, max length, batch tensor shape 확인
4. CPU 또는 단일 GPU 1-batch forward/backward smoke test
5. 짧은 overfit/sanity run
6. 가능한 범위에서 최소 pretraining run

다음을 확인하라.

- MLM loss가 finite
- SOP loss가 finite
- SOP excluded sample이 loss에 포함되지 않음
- gradient가 MLM head, SOP head, sequence encoder, multimodal projection에 기대대로 흐름
- frozen external encoder에는 gradient가 없음
- model vocab size와 registry vocab size 일치
- padding과 mask target이 loss를 오염시키지 않음

장시간 전체 학습을 임의로 시작하지 말라. smoke 및 짧은 sanity run 이후 전체 학습 명령과 예상 자원 사용량을 보고하라.

## 19. 구현 산출물

파일명은 저장소 관례에 맞게 결정하되 최소 다음 책임을 분리하라.

```text
Neo4j preflight/purge
online2 canonicalization and registries
cell/token graph loader
event materializer
sequence materializer
Neo4j validator/exporter
online2 TokenSource
online2 Population
online2 Corpus/DataModule configuration
online2 Vocabulary configuration
group-aware MLM/SOP task support
unit/integration tests
operator documentation
```

대형 단일 스크립트 하나에 삭제, 적재, 물질화, 학습을 모두 넣지 말라. 각 단계는 독립 실행, 재시작, 검증이 가능해야 한다.

## 20. 단계별 실행 순서

반드시 다음 순서를 지켜라.

### Phase 0 — 조사와 계약 고정

- baseline inventory
- schema/version/ID/canonicalization 계약
- normalized narrative catalog
- binning/tokenization policy
- 테스트 fixture

### Phase 1 — 안전 삭제

- Neo4j dry run
- pre-delete report
- explicit execute
- post-delete zero-count verification
- online2 constraints/indexes

### Phase 2 — 원자값과 token

- SourceFile/SourceRow
- CellOccurrence
- AtomicValue
- CategoricalToken
- registries와 mapping
- raw round-trip 및 cardinality 검증

### Phase 3 — event

- local observation
- growth
- derived
- aggregate
- image
- interpretation
- relation
- SameTimeGroup

### Phase 4 — sequence와 corpus

- normalized catalog의 ACTIVE 전체를 파싱한 narrative instance extraction
- gap/background/context metadata
- deduplication
- Neo4j Sequence/Corpus
- Parquet/JSON export

### Phase 5 — life2vec 통합

- Source
- Population
- Vocabulary
- Corpus/DataModule
- task/type/cache/config
- external embedding projection

### Phase 6 — MLM/SOP

- group-aware permutation
- eligibility/loss mask
- aligned metadata
- unit tests

### Phase 7 — 검증과 smoke training

- full validation report
- idempotency rerun
- encoded batch inspection
- forward/backward
- short sanity run

각 phase 종료 시 acceptance check를 실행하고 PASS 증거를 남겨라.

## 21. 완료 기준

다음 조건을 모두 만족해야 작업 완료로 간주한다.

1. 기존 online1 graph node와 relationship이 0건이다.
2. online2 raw cell을 exact value까지 역추적할 수 있다.
3. 모든 CellOccurrence가 AtomicValue와 연결된다.
4. 모든 maskable cell이 registry 기반 CategoricalToken과 연결된다.
5. token→event→sequence→corpus lineage가 완전하다.
6. event와 segment가 1:1이다.
7. local event가 정확히 하나의 farm-zone을 가진다.
8. 같은 시각의 다른 공간 event는 별도 event이면서 SameTimeGroup을 공유한다.
9. Neo4j와 manifest count/checksum이 일치한다.
10. 두 번째 적재에서 duplicate와 count 증가가 없다.
11. problem hidden interpretation과 future leakage가 0건이다.
12. same-time 내부 shuffle/reverse SOP test가 모두 `ARRANGED`다.
13. group 간 reversed/shuffled label test가 모두 통과한다.
14. Source→Vocabulary→Corpus→DataModule encoded batch가 생성된다.
15. MLM+SOP forward/backward smoke test가 통과한다.
16. 변경한 파일에 linter/type/test 오류가 없다.
17. 운영자가 재실행할 명령과 rollback/복구 절차가 문서화되어 있다.

## 22. 최종 응답 형식

작업 완료 후 다음 순서로 간결하게 보고하라.

1. 구현 결과 요약
2. 변경·추가 파일 목록과 각 책임
3. Neo4j 삭제 전/후 및 online2 적재 통계
4. registry와 corpus 통계
5. SOP 동일시점 처리 방식과 테스트 결과
6. MLM/SOP smoke training 결과
7. 실행한 명령
8. 미완료 항목 또는 외부 리소스 blocker
9. 전체 학습 실행 명령과 예상 자원

테스트하지 않은 내용을 완료했다고 보고하지 말라. Neo4j 인증, GPU, frozen LLM/VLM 등 외부 자원이 없어 실행하지 못한 단계는 구현 완료와 실행 검증을 분리하여 정확히 기록하라.
