# 원본 절대 경로

스냅샷 파일 → repo 원본

```
01_dino_fuse/image_adapter.py
  → src/online2/v2/finetune_v02/image_adapter.py
01_dino_fuse/stage_a_image_encode.py
  → src/online2/v2/finetune_v02/stage_a_image_encode.py
01_dino_fuse/cache_stage_a_image_slots_v02.py
  → scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py
01_dino_fuse/image_index.py
  → src/online2/v2/finetune_v02/image_index.py
01_dino_fuse/cache_stage_a_event_embeddings_IMAGE_and_pool.py
  → scripts/online2_v2/cache_stage_a_event_embeddings.py (excerpt)

02_pad_meta/event_pooling_finetune.py
  → src/online2/v2/event_pooling_finetune.py

03_v02_model/model.py
  → src/online2/v2/finetune_v02/model.py
03_v02_model/dataset.py
  → src/online2/v2/finetune_v02/dataset.py

04_pretrain_encoder_mlm/*
  → src/transformer/transformer.py, models.py (excerpts)

05_masking/masking.py
  → src/online2/v2/masking.py
05_masking/tasks_mlm.py
  → src/tasks/mlm.py

06_tokenizer_actuator/tokenizer.py
  → src/online2/v2/tokenizer.py

07_saliency/saliency.py
  → src/online2/v2/finetune_v02/saliency.py
07_saliency/v01_layer_saliency.py
  → src/online2/v2/v01_layer_saliency.py
07_saliency/consensus.py
  → src/online2/v2/finetune_v02/consensus.py

08_image_events/*
  → src/online2/builder.py, scripts/online2_v2/build_v2.py,
     scripts/online2_v2/export_training_events_v2_event_grain.py (excerpts)
```
