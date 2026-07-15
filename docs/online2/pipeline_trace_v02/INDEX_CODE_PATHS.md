# 코드 경로 색인

## Sequence

- `src/online2/builder.py` — `_modality`, `_timestamp`, `_emit_group`, `process_csv`
- `scripts/online2_v2/build_v2.py` — `tokenize_events`, `materialize_sequences`, `build_registries`
- `scripts/online2_v2/export_training_events_v2_event_grain.py`

## Tokenizer / registry

- `src/online2/v2/tokenizer.py` — `TokenizerV2.tokenize_value`
- `src/online2/v2/vocab.py` — `VocabV2`
- `src/online2/v2/binning.py` — `BinRuleV2`, `BinningRegistryV2`
- `src/online2/v2/feature_schema.py`

## Model

- `scripts/online2_v2/cache_stage_a_event_embeddings.py` — `encode_batch_pool`
- `src/online2/v2/event_pooling_finetune.py` — `EventMetadataEncoder`, `PooledAttentionEventDecoder`, `DiagnosisHead`, `EventPoolingDiagnosisModel`
- `src/online2/v2/finetune_v02/model.py` — `EventPoolingDiagnosisModelV02`
- `scripts/online2_v2/v02/run_diagnosis_finetune_v02.py`
- `scripts/online2_v2/v02/run_evaluation_pipeline_v02.py`

## Saliency

- `src/online2/v2/finetune_v02/saliency.py` — `event_input_x_gradient`
- `src/online2/v2/finetune_v02/consensus.py` — `consensus_table`, `select_report_events`
- `src/online2/v2/v01_layer_saliency.py` — token IxG (optional)
- `scripts/online2_v2/analyze_v01_layer_saliency.py`

## Evaluation → evidence

- `scripts/online2_v2/v02/run_evaluation_pipeline_v02.py` — ensemble, `run_case_saliency`, `build_structured_evidence`
- `src/online2/v2/finetune_v02/evidence.py` — structured evidence JSON

## Report

- `scripts/online2_v2/generate_gemma_reports.py` — `load_token_saliency`, `build_rich_payload`, markdown writer
- `src/online2/v2/finetune_v02/gemma_report.py`

## IMAGE / DINO (optional Stage A)

- `scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py`
- `src/online2/v2/finetune_v02/{image_adapter,stage_a_image_encode,image_index}.py`

## Labels / split

- `src/online2/v2/diagnosis_dataset.py` — `make_stratified_farm_folds`
- `outputs/online2/v2_finetune/label_map.json`
- `outputs/online2/v2_finetune/labels_example_score90.csv`
