# Online2 코퍼스 운영 가이드

## 범위

`src/online2`는 공개 online2 파일을 오프라인에서 inventory하고, 원자값·토큰·이벤트·시퀀스 Parquet을 만든 뒤 필요할 때 Neo4j에 적재한다. 빌드는 Neo4j에 접속하지 않는다.

normalized catalog의 모든 ACTIVE 행을 `materialization_key` registry로 실행한다. registry는 일별/hourly, 생육·혼합 생육, cross-zone, threshold·positive·change·jump·reset·run·quantile·joint·and·diff, 설비 불일치, CO2/EC/root 진단 후보, noon, 이미지 맥락, score90 해석, 공개 관측 similarity를 지원한다. 같은 key를 공유해도 각 행의 `data_sources/event_views`, required columns, min/max events, timestamp precision, order semantics와 OP 정책을 별도로 적용한다.

각 narrative의 matcher 실행 여부, trigger row 수, exact duplicate 제거 후 unique sequence 수, 길이 탈락과 catalog 추정치 차이는 `validation_report.json`에 따로 기록된다. trigger가 0개인 것은 `NO_TRIGGER_IN_PUBLIC_POOL`이며 미지원으로 위장하지 않는다. 등록되지 않은 key는 `FAIL_UNSUPPORTED_SEMANTICS`로 전체 상태를 `LIMITED`로 만든다. 이미지와 텍스트 embedding은 생성하지 않으며 `pending`으로 저장한다.

## 재현성 계약

- ID는 `online2-v1`, namespace, canonical natural key를 SHA-256으로 계산한다.
- null, bool, int, IEEE-754 float(hex), Decimal의 tuple, UTC datetime과 precision, date, UTF-8 string을 구분한다.
- naive datetime은 허용하지 않는다. CSV의 naive 시각은 `Asia/Seoul`로 해석해 UTC로 저장한다.
- source row ID는 file checksum과 전체 row natural key를 포함한다.
- `SOURCE_DATE_EPOCH` 기본값은 0이다. 따라서 같은 입력의 manifest가 실행 시각 때문에 달라지지 않는다.
- numeric registry는 public observation 전체에서 한 번 적합하며 ABS, GLOBAL quantile, FARM_REL quantile edge를 분리한다. 중복 edge는 제거하고, 상수와 null은 별도 token으로 보존한다.
- trigger matcher는 현재 시각과 과거 context만 sequence에 넣는다. population quantile threshold는 공개 pool에서 한 번 적합한 정책값이며 개별 event의 미래 window가 아니다.
- sequence는 exact event-ID 배열로 중복 제거하고 `16 <= effective_token_count <= 5120`을 강제한다. event timestamp는 non-decreasing, 서로 다른 time group timestamp는 strictly increasing이어야 한다.
- period/relation/set/retrieval semantics는 catalog 값과 무관하게 OP 비대상으로 내린다. cross-zone set metadata는 정렬된 parent ID를 사용해 순열 불변이다.

## 오프라인 실행

```bash
scripts/online2 inventory \
  --output outputs/online2/baseline_inventory.json

scripts/online2 normalize-catalog \
  --output outputs/online2/normalized_catalog.csv

scripts/online2 build \
  --output outputs/online2/build-v1 \
  --batch-size 10000

scripts/online2 export-training \
  --build-dir outputs/online2/build-v1
```

Parquet writer와 CSV reader는 batch/streaming으로 동작한다. `build_manifest.json`, `validation_report.json`, registry JSON과 모든 lineage Parquet을 확인한 뒤에만 DB 작업을 진행한다. 학습용 `training_events.parquet`와 `life2vec_token_registry.json`은 Neo4j 없이 export한다.

## Neo4j 환경과 preflight

다음 값을 환경변수로만 전달한다. 비밀번호는 report에 기록하지 않는다.

```bash
# URI 직접 지정 또는 HOST/BOLT 조합
export NEO4J_URI='bolt://host:7687'
# 또는
export NEO4J_HOST='host'
export NEO4J_BOLT='7687'

export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='...'
export NEO4J_DATABASE='agrichallenge'   # .env의 NEO4j_DB 오타도 허용

scripts/online2 neo4j preflight
scripts/online2 neo4j report --output outputs/online2/neo4j-before.json
```

## 삭제 안전 절차

기본 purge는 report만 만드는 dry-run이다.

```bash
scripts/online2 neo4j purge \
  --report outputs/online2/purge.json
```

실제 삭제는 아래 세 값이 모두 있어야 한다.

```bash
scripts/online2 neo4j purge \
  --report outputs/online2/purge.json \
  --execute \
  --database agrichallenge \
  --expected-dataset online1
```

설정 database와 CLI database가 다르거나 online1 legacy signature(`Dataset.dataset_tag=online1`, `NarrativeCatalog.name=online1_v1`, 또는 Cell/EnvSample/SoilSample 라벨)가 없으면 중단한다. 삭제 전후 report를 남기며 batch `DETACH DELETE` 후 node/relationship 0건을 검증한다. pre-delete `SHOW CONSTRAINTS`/`SHOW INDEXES`에 나타난 사용자 schema object를 제거하고 lookup/system index는 유지한다.

삭제는 되돌릴 수 없다. rollback은 삭제 전 Neo4j native backup 또는 dump를 별도로 확보하고, DB를 중지한 뒤 해당 backup으로 복구하는 방식으로만 수행한다. backup이 없으면 `--execute`를 사용하지 않는다.

## Schema, 적재, 검증

```bash
scripts/online2 neo4j ensure-schema
scripts/online2 neo4j load \
  --build-dir outputs/online2/build-v1 \
  --batch-size 2000
scripts/online2 neo4j validate \
  --expected-counts outputs/online2/expected_node_counts_r3.json
```

적재는 uniqueness constraint와 `UNWIND` batch `MERGE`를 사용하므로 같은 build를 재실행해도 duplicate node나 relationship을 만들지 않는다. 전체 build가 `LIMITED`이면 그 제한을 해소하지 않은 상태를 완료로 간주하지 않는다.

## life2vec smoke

```bash
# smoke subset은 full export에서 샘플링한 뒤
ONLINE2_SMOKE_DIR=outputs/online2/smoke_train \
  python scripts/online2_smoke_pretrain.py
```

전체 사전학습 예:

```bash
export ONLINE2_PARQUET_PATH=outputs/online2/build-v8-active80-r3/training_events.parquet
export ONLINE2_BUILD_ID=build_85c4aba3a72e0e321cfcf6af46eecd20a19a2c0a85fab2b70d0af677dd85840a
export ONLINE2_REGISTRY_VERSION=1
export ONLINE2_TOKEN_REGISTRY_PATH=outputs/online2/build-v8-active80-r3/life2vec_token_registry.json
export ONLINE2_REFERENCE_DATE=2024-01-01
export ONLINE2_THRESHOLD=2099-01-01
python -m src.train experiment=pretrain_online2
```

예상 자원: vocab≈994, sequences≈967k, max_length=2560, GPU 24GB에서는 batch_size 1–4와 gradient accumulation을 권장한다. 이미지/텍스트 frozen embedding은 `pending`이므로 기본 smoke는 categorical+sensor MLM/SOP만 검증한다.

세부 산정·명령·reconnect 검증 결과는 `outputs/online2/full_pretrain_plan.json`과 `outputs/online2/acceptance_v8.md`를 본다. RTX 3090 / batch_size=4 기준 1 epoch는 대략 8–34시간, 설정상 50 epoch는 수백~천 시간대이므로 먼저 `trainer.max_epochs=1`로 전체 데이터 1 epoch만 돌리는 것을 권장한다.
