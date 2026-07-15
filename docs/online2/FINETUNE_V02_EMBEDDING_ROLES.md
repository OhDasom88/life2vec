# Finetune v0.2 — Embedding roles

## Two different embedding uses (do not mix)

| Embedding | Source | Used as | Not used as |
|-----------|--------|---------|---------------|
| **Text** (해석문) | score90/70/50 상태진단·원인분석 | **Semantic alignment targets** (`z_state`/`z_cause` cosine) | Stage A 입력 / 분류 CE 입력 |
| **Image** (DINO) | `external_embeddings/image_*.npy` keyed by IMAGE `event_id` | **IMAGE_SLOT / IMAGE event** residual via fold-trainable adapter | 해석문·text bank |

```text
관측 시퀀스 (센서 토큰 + IMAGE_SLOT)
        │
        ▼
Stage A cache  (+ dino_vec on IMAGE rows)
        │
        ▼
Stage B PAD  ←── SharedImageAdapter(DINO) @ IMAGE events
        │
   ┌────┴────┐
   ▼         ▼
 CE head   state/cause heads ──align──► text bank (H/M/L)
```

## Build / train

```bash
# text bank (alignment targets)
python scripts/online2_v2/v02/build_interpretation_banks.py

# IMAGE-enabled Stage A caches (writes dino_vec column; v0.1 script untouched)
python scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py --device cuda --skip-existing

# CV (CE + alignment + IMAGE adapter)
python scripts/online2_v2/v02/run_diagnosis_finetune_v02.py \
  --mode cv --device cuda --monitor val_macro_f1 \
  --emb-dir outputs/online2/v2_finetune_v02/event_embeddings
```

Recommended IMAGE path: **attach raw DINO + fold-trainable adapter** (default).  
`--inject-slots` bakes residuals into frozen Stage A (ablation only).

Optional later: replace TF-IDF text bank with Qwen (same parquet schema).
