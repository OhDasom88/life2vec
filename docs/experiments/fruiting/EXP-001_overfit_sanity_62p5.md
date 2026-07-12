# EXP-001: 1차 확인 — Overfit sanity (plain CE)

| 항목 | 값 |
|------|-----|
| **Exp ID** | FRT-001 |
| **날짜** | 2026-07-02 |
| **상태** | 완료 |
| **단계** | Overfit sanity check |
| **Hydra** | `experiment=finetune_agri_fruiting_overfit` |
| **W&B** | offline: `logs/agri/wandb/offline-run-20260702_151024-up7kbrf0` |
| **Notion** | [EXP-001](https://app.notion.com/p/39175b02ea8181549035c45e4ccbcf14) |
| **GitHub** | [eee0b4e](https://github.com/OhDasom88/life2vec/commit/eee0b4ebe9cd4bc2488a5a820ba08d3d3652e517) *(분류 코드 미커밋)* |
| **Label Studio** | `berry2vec-fruiting-EXP-001` · [import](../../data/labelstudio/imports/EXP-001_predictions.json) |
| **체크포인트** | `checkpoints/agri/fruiting/l2v/overfit/overfit-epoch=19.ckpt` |
| **Git** | `eee0b4e` (학습 시점; 분류 모델 코드는 미커밋) |

## 한줄 결론

> **파이프라인 동작 1차 확인**: 전체 40건으로 train/val 동일 평가 시 버킷 Acc **62.5%**, MAE Count **0.625**. 완전 과적합은 실패했으나 다수 클래스 상수 예측(40%)보다 개선.

## 가설 / 목적

- 분류 모델(`Transformer_CLS` + 6버킷) 전환 후 **학습 파이프라인이 정상 동작하는지** 확인
- weighted sampling / class weight **없이** 순수 memorization 가능 여부 검증

## 변경 사항

### 데이터
- `AgriFruitingOverfitPopulation`: train/val/test **전부 40건**
- `weighted_sampling: false` (일반 shuffle)

### 모델
- `Transformer_AgriFruiting` ← `Transformer_CLS` 기반
- pooled `AttentionDecoder`, 출력 6클래스
- `fruiting_count_to_bucket()` 로 TARGET 매핑

### 파라미터
| 항목 | 값 |
|------|-----|
| loss | `entropy` (plain CE, class weight 없음) |
| lr | 1e-3 |
| dropout | 0 (att/fw/dc/emb) |
| freeze_embeddings | false |
| redraw_projections | false |
| max_epochs | 300 |
| batch_size | 8 |

## 결과

| 지표 | val (40건) |
|------|------------|
| **버킷 Acc** | **62.5%** (25/40) |
| **MAE Count** | **0.625** |
| Loss | ~1.18 (epoch 19 이후 정체) |

### 실제 vs 예측 버킷 (best ckpt)

| 버킷 | 실제 | 예측 |
|------|------|------|
| 0 (≤3) | 2 | 0 |
| 1 (4) | 6 | 4 |
| 2 (5) | 16 | 25 |
| 3 (6) | 10 | 11 |
| 4 (7) | 4 | 0 |
| 5 (≥8) | 2 | 0 |

→ 버킷 2·3(착과수 5·6)에 쏠림. 희소 버킷 미학습.

## 해석

- forward → loss → backward → checkpoint **정상**
- 300 epoch에도 **acc 62.5%에서 정체** → 표현력 또는 optimizer/구조 이슈 가능
- 본학습 전 **decoder-only probe** 또는 **MLP baseline** 권장

## 실행 명령

### 학습
```bash
python -m src.train experiment=finetune_agri_fruiting_overfit trainer.devices=[0]
```

### 예측 (EXP-001 전용)
```bash
python scripts/predict_fruiting_exp001.py
python scripts/predict_fruiting_exp001.py --show-errors
```

> `predict_fruiting_exp001.py`는 학습 당시 vocabulary(`agri_growth_set`, 86 tokens)와
> 체크포인트 hparams를 맞춰 62.5%를 재현합니다. 범용 `predict_fruiting.py`와 설정이 다릅니다.

## W&B 동기화

```bash
wandb sync logs/agri/wandb/offline-run-20260702_151024-up7kbrf0
```

## Label Studio 검수

| 항목 | 값 |
|------|-----|
| 프로젝트 | `berry2vec-fruiting-EXP-001` |
| Import | `data/labelstudio/imports/EXP-001_predictions.json` (6 tasks, test split) |
| Config | `labelstudio/projects/fruiting/config.xml` |
| 상태 | import 준비 완료, 검수 대기 |

```bash
python scripts/export_labelstudio_fruiting.py \
  --predictions outputs/berry2vec_fruiting_predictions/predictions_all.csv \
  --exp-id EXP-001 \
  --output data/labelstudio/imports/EXP-001_predictions.json
```

## 다음 단계

- [ ] EXP-003: train/val split 본학습
- [ ] decoder-only overfit 실험
- [ ] W&B run cloud sync 후 Notion URL 연결
