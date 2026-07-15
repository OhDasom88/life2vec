# Finetune v0.2 scripts

## Roles (important)

| Signal | Used as | Not used as |
|--------|---------|-------------|
| **Text embeddings** (Qwen / bank) | state/cause **alignment targets** | Stage A / PAD **input** |
| **Image embeddings** (DINOv2) | residual at **IMAGE event / IMAGE_SLOT** | analysis/report input |

## Checkpoints

```text
outputs/online2/v2_finetune_v02/runs/<run>/fold{0..4}_best.pt
```

## Pipeline

```bash
# 1) Build H/M/L × state/cause text bank (structure)
python scripts/online2_v2/v02/build_interpretation_banks.py

# 2) Replace embeddings with Qwen (alignment targets)
python scripts/online2_v2/v02/embed_interpretation_bank_qwen.py

# 3) Stage A with IMAGE + attach DINO (v0.1 script called with --include-image)
python scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py --device cuda
# optional token-slot bake: add --inject-slots

# 4) Train (CE + semantic align + fold-trainable image adapter)
python scripts/online2_v2/v02/run_diagnosis_finetune_v02.py \
  --mode cv --epochs 150 --device cuda --monitor val_macro_f1 \
  --emb-dir outputs/online2/v2_finetune_v02/event_embeddings

# 5) Eval (auto-detects v02)
python scripts/online2_v2/run_evaluation_pipeline.py \
  --run-dir outputs/online2/v2_finetune_v02/runs/<cv_stamp> \
  --emb-dir outputs/online2/v2_finetune_v02/event_embeddings \
  --model-version auto --saliency-cases all

# Preferred for IMAGE/DINO-aware reports (writes only under v2_finetune_v02/):
python scripts/online2_v2/v02/run_evaluation_pipeline_v02.py \
  --run-dir outputs/online2/v2_finetune_v02/runs/<cv_stamp> \
  --labels outputs/online2/v2_finetune_v02/labels_example35_problem20.csv \
  --saliency-cases all \
  --out-dir outputs/online2/v2_finetune_v02/evaluation/<stamp>

# Live Gemma markdown (point --out-dir at v02; do not touch v0.1 evaluation/)
python scripts/online2_v2/generate_gemma_reports.py \
  --eval-dir outputs/online2/v2_finetune_v02/evaluation/<stamp> \
  --out-dir outputs/online2/v2_finetune_v02/evaluation/<stamp>/reports/gemma4_local_rich \
  --api-mode openai --api-base http://127.0.0.1:8080/v1 --prompt-style rich
```

Do **not** edit v0.1 `cache_stage_a_event_embeddings.py` / `run_diagnosis_event_pooling_finetune.py`.

Next version (design): `docs/online2/DIAGNOSIS_FINETUNE_V03.md`, scaffold `scripts/online2_v2/v03/`.
Do **not** morph v0.2 entrypoints into v0.3 in place.
