# 01. 시퀀스 생성

## 핵심 코드

| 단계 | 파일 | 함수 |
|------|------|------|
| CSV → cell/event | `src/online2/builder.py` | `CorpusBuilder.process_csv`, `_emit_group`, `_modality` |
| V2 재토큰화 | `scripts/online2_v2/build_v2.py` | `tokenize_events`, `materialize_sequences` |
| event-grain export | `scripts/online2_v2/export_training_events_v2_event_grain.py` | (sequences ⨝ events) |

## raw row → event

1. CSV 경로는 `_modality`로 view 결정  
   `E_environment` / `A_actuator` / `R_rootzone` / `G_growth` / `I_images`.
2. 각 행의 timestamp는 KST naive → UTC `Z`로 변환 (`_timestamp`, timezone policy 문서 참고).
3. `(timestamp, farm, zone, view)` 단위로 `event_id`를 만들고 cell을 event에 연결.
4. `cell_occurrences`에 `raw_value` / `raw_display` 저장.

주요 산출물:

- `outputs/online2/build-v8-active80-r3/cell_occurrences.parquet`
- `outputs/online2/build-v8-active80-r3/events.parquet`
- `outputs/online2/build-v8-active80-r3/same_time_groups.parquet`

## 동일 시점 이벤트 묶음

`_emit_group`이 `same_time_group_id`를 발급합니다.

- **같은 timestamp**면 같은 STG에 묶임.
- 그러나 **farm / zone / view가 다르면 별도 event**로 유지 (한 STG 안에 여러 event 가능).
- `build_v2.materialize_sequences`는 event 사이에  
  `DELTA_T|*`, `DAY_FROM_START|*`, `LOCAL_HOUR|*`, `[GROUP_SEP]`를 삽입.

## zone / channel 처리

| 개념 | 표현 |
|------|------|
| zone | builder: `zone_id` / `zone_ids`; finetune: `zone_emb` (정수 ID) |
| channel/view | CSV modality → event_view → V2 `VIEW\|ENVIRONMENT` 등 → `view_id` |

V2 토큰 단계에서 view 토큰이 SENTENCE 앞에 붙고, Stage A cache/PAD에서는 `view_id`, `zone_id`, `local_hour`, `case_age_hours` 메타로 인코딩됩니다.

## 구동기(actuator) 이벤트

- 원천: `datasets/agrichallenge/online2/data/A_actuator/F*_z*.csv`
- modality: `A_actuator` → `VIEW|ACTUATOR`
- 값 타입(`ordinal_actuator` / `boolean` / `flow`)은 대개 **연속 bin이 아니라 literal state**  
  예: `OBSERVED_VALUE|ZERO` / `POSITIVE`, `STATE_SEMANTICS|UNRESOLVED`  
  (`src/online2/v2/tokenizer.py` `tokenize_value`)

## V2 학습용 시퀀스 산출물

- `outputs/online2/v2_build/events_tokenized_v2.parquet` — event당 SENTENCE / token_ids
- `outputs/online2/v2_build/sequences_v2.parquet`
- `outputs/online2/v2_build/training_events_v2.parquet` — seq×event grain
