# SOURCE_EXCERPTS — 실제 구현 스냅샷

경로 설명만이 아니라 **패치 설계용 소스 복사·발췌**입니다.  
원본이 바뀌면 여기 스냅샷은 stale 할 수 있으니, 최종 패치는 항상 repo 원본을 기준으로 하세요.

원본 루트: `/home/dasom/life2vec`

## 우선순위 맵

| # | 목적 | 이 디렉터리 | repo 원본 |
|---|------|-------------|-----------|
| 1 | DINO load/project/fuse · gate · 정규화 | [`01_dino_fuse/`](01_dino_fuse/) | 아래 참고 |
| 2 | `EventMetadataEncoder` + padding | [`02_pad_meta/event_pooling_finetune.py`](02_pad_meta/event_pooling_finetune.py) | `src/online2/v2/event_pooling_finetune.py` |
| 3 | v0.2 모델 전체 (mean/max/DINO) | [`03_v02_model/`](03_v02_model/) | `src/online2/v2/finetune_v02/model.py` |
| 4 | Encoder forward + MLM head | [`04_pretrain_encoder_mlm/`](04_pretrain_encoder_mlm/) | `src/transformer/{transformer,models}.py` |
| 5 | Masking (토큰 MLM / grouped) | [`05_masking/`](05_masking/) | `src/tasks/mlm.py`, `src/online2/v2/masking.py` |
| 6 | Actuator ZERO/POSITIVE | [`06_tokenizer_actuator/tokenizer.py`](06_tokenizer_actuator/tokenizer.py) | `src/online2/v2/tokenizer.py` |
| 7 | Event→token saliency 연결 | [`07_saliency/`](07_saliency/) | `finetune_v02/saliency.py`, `v01_layer_saliency.py` |
| 8 | IMAGE 이벤트·slot·ID 소실 | [`08_image_events/`](08_image_events/) | `builder.py`, `build_v2.py`, export |

월드모델 재사용 판정 요약: [`WORLD_MODEL_REUSE.md`](WORLD_MODEL_REUSE.md)

---

## 1. DINO fuse (중요)

`cache_stage_a_event_embeddings.py` **안에는 DINO project/fuse가 없습니다.**  
기본 Stage A는 IMAGE를 **제외**하고, span mean/max만 냅니다.

실제 fuse 수식은 여기:

| 파일 | 역할 |
|------|------|
| `image_adapter.py` | `u = A(v/‖v‖); residual = g * u` |
| `stage_a_image_encode.py` | `[IMAGE_SLOT]` emb에 residual 더한 뒤 `forward_finetuning_with_embeddings` → mean/max |
| `cache_stage_a_image_slots_v02.py` | attach `dino_vec` / optional `--inject-slots` bake |
| `image_index.py` | DINO parquet → `event_id → vec` |
| `cache_stage_a_event_embeddings_IMAGE_and_pool.py` | IMAGE exclude + (비-DINO) pool |

**Fold 학습 시 기본 경로 (v0.2 model):**  
`event_mean/max ← event_mean/max + SharedImageAdapter(dino) * dino_mask`  
(`03_v02_model/model.py` `fuse_image_into_events`)

---

## 2–3. Downstream 입력

- PAD는 **`event_mean` ‖ `event_max` (+ age/view/zone/hour)** 만 받음.  
  `event_embedding` 단일 필드는 cache에서 `None`으로 두는 경우가 많음.
- padding: `z *= padding_mask`; MHA에는 `key_padding_mask = ~padding_mask`.
- DINO는 meta 전에 mean/max에 잔차로 섞임 → **fused mean/max가 `EventMetadataEncoder`에 전달**.

---

## 4–5. 사전학습 ↔ 월드모델

- Encoder: `Transformer.forward_finetuning` / `_with_embeddings` — bidirectional stack, **causal mask 없음**.
- MLM head: `MaskedLanguageModel` — `target_pos` gather → vocab softmax.
- Masking:
  - life2vec 학습: `src/tasks/mlm.py` `mlm_mask` — **토큰 단위 랜덤** (시간 span 아님).
  - online2 V2 grouped: `GroupedMLMMasker` — measurement group 단위, 역시 **시간 방향 future mask 아님**.
- SOP는 same-time **블록 순서** 셔플/역순 — 미래 예측 목표가 아님.

→ **가중치(encoder) 재사용은 가능**, 그러나 현재 목표는 masked reconstruction + SOP이지  
**masked future world model이 아님.** 시간 span / causal 확장은 masking·attn mask를 새로 짜야 함.

---

## 6. Actuator

`literal_state`: `value==0 → ZERO`, `>0 → POSITIVE`, `<0 → NEGATIVE`.  
연속 magnitude/bin 없음 → **세기 손실**이 구조적으로 발생.

패치 후보: `ordinal_actuator`/`flow`에 ABS bin 병행, 또는 multi-level ordinal 토큰.

---

## 7. Saliency 연결

- Event: IxG on `z` after meta (`saliency.py`).
- Token: `v01_layer_saliency.token_ixg_for_event`가  
  Stage A 재인코드 → mean/max를 case batch에 **교체** → fold head logit IxG on token emb.  
- v0.2 eval은 token 경로를 기본 돌리지 않음 → evidence에서 token 공백.

---

## 8. IMAGE ID·slot이 사라지는 지점

```text
I_images CSV
 → builder process_external: event_kind=IMAGE, token=IMAGE_EMBED_SLOT, embedding_status=pending
 → build_v2 tokenize: SENTENCE="[IMAGE_SLOT] IMAGE_ROLE|… VIEW|IMAGE …"
 → Stage A default: IMAGE event_kind 제외 (--include-image 없으면)
 → PAD cache: 센서 이벤트만; dino는 v02 attach/inject 또는 fold fuse
 → evidence/report: event_id·view·saliency 중심, image file ID / slot index 미전파
```

slot index는 토큰 시퀀스 위치(`find_image_slot_positions`)에만 존재하고,  
event parquet/evidence JSON 스키마에는 **보존되지 않음**.
