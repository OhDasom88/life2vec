# EXP-002: Overfit — weighted sampling + asymmetric loss

| 항목 | 값 |
|------|-----|
| **Exp ID** | FRT-002 |
| **날짜** | 2026-07-02 |
| **상태** | 완료 (실패) |
| **단계** | Overfit sanity check |
| **Hydra** | `experiment=finetune_agri_fruiting_overfit` (당시 weighted+asymmetric 설정) |
| **W&B** | offline: `logs/agri/wandb/offline-run-20260702_144838-yzfwn8cp` |
| **Notion** | [EXP-002](https://app.notion.com/p/39175b02ea8181c7918bfab190405802) |
| **GitHub** | [eee0b4e](https://github.com/OhDasom88/life2vec/commit/eee0b4ebe9cd4bc2488a5a820ba08d3d3652e517) |
| **Label Studio** | — (오답 검수 불필요) |
| **체크포인트** | `checkpoints/agri/fruiting/l2v/overfit/overfit-epoch=07.ckpt` |

## 한줄 결론

> **이중 불균형 보정 실패**: weighted sampling + asymmetric loss + class weight 동시 적용 시 val MAE **2.4**, Acc **10%** — 극단 버킷(≤3, ≥8)으로 쏠림.

## 변경 사항 (EXP-001 대비)

### Sampling
- `WeightedRandomSampler` ON — 버킷 역빈도 기반 upsample

### Loss
- `AsymmetricMulticlassCrossEntropyLoss`
- `asym_beta=1.5` (과소예측 페널티), `asym_alpha=1.0`
- `on_fit_start`에서 class weight (역빈도) 주입

## 결과

| 지표 | val |
|------|-----|
| 버킷 Acc | 10% (4/40) |
| MAE Count | 2.400 |

### 예측 분포
- 버킷 5 (≥8): **38건** 예측
- 버킷 0 (≤3): **2건** 예측

## 해석

- sampling 가중치 + loss class weight **이중 보정**이 희소 클래스를 과도하게 끌어올림
- overfit sanity에는 **plain CE + no sampling**이 적합 (→ EXP-001)

## W&B 동기화

```bash
wandb sync logs/agri/wandb/offline-run-20260702_144838-yzfwn8cp
```
