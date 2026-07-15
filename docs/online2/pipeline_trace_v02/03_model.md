# 03. 모델 코드

## 단계 구조

```text
Stage A (frozen encoder)
  sequences → TransformerEncoder → event span mean/max cache
Stage B/C (finetune)
  EventMetadataEncoder → PooledAttentionEventDecoder → DiagnosisHead
```

## Stage A: encoder forward · event pooling (캐시)

- 스크립트: `scripts/online2_v2/cache_stage_a_event_embeddings.py`
- `encode_batch_pool`: target event 토큰 span에 대해  
  **masked mean + elementwise max** → `event_mean`, `event_max` (각 384-d, life2vec hidden).
- 산출: `outputs/online2/v2_finetune_v02/event_embeddings/{case_id}.parquet`

일상 finetune/inference는 **원 토큰 시퀀스를 다시 넣지 않고** 이 cache만 사용합니다.

## Meta → z (event 표현)

`src/online2/v2/event_pooling_finetune.py` `EventMetadataEncoder.forward`

- 입력: `event_mean ‖ event_max` + age/view/zone/hour embedding
- 출력: `z` `[B, T, hidden]` (padding mask 적용)

## Event pooling (case 표현)

`PooledAttentionEventDecoder.forward`

- learned query가 event `z`에 cross-attention
- 출력: `h_case`, `attn` weights

## Classification head

`DiagnosisHead.forward`: LayerNorm → Dropout → Linear(`num_classes=10`) → logits

## Hidden state 반환 여부

| 텐서 | 반환? | 용도 |
|------|-------|------|
| `z` (event) | ✅ | saliency IxG 입력 |
| `h_case` | ✅ | 분류 / state·cause align (v0.2) |
| `attn` | ✅ | (선택) 가중치 로그 |
| 토큰급 hidden | ❌ (일상 경로) | Stage A 재실행 + layer saliency 스크립트에서만 |

v0.2 래퍼: `src/online2/v2/finetune_v02/model.py` `EventPoolingDiagnosisModelV02`  
→ `{logits, z, h_case, attn, z_state, z_cause, ...}`

## Fold별 inference

1. 학습: `scripts/online2_v2/v02/run_diagnosis_finetune_v02.py`  
   - `make_stratified_farm_folds` → fold마다 **새 모델·optimizer**  
   - checkpoint: `runs/cv_*/fold{k}_best.pt`
2. 평가: `scripts/online2_v2/v02/run_evaluation_pipeline_v02.py`  
   - `load_fold_models` → 케이스별 5모델 softmax  
   - ensemble 평균 확률 + agreement(가장 높은 클래스에 투표한 fold 수)
