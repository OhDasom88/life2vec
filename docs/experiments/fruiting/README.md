# 착과수 (Fruiting) 실험 인덱스

## 연동 허브

| 도구 | 링크 |
|------|------|
| **Notion** | [실험 허브](https://app.notion.com/p/39175b02ea8181b89a81fd7422f447d3) · [실험 DB](https://app.notion.com/p/dab8a00f3dad4488bfe5607c6c5365c0) |
| **GitHub** | [OhDasom88/life2vec](https://github.com/OhDasom88/life2vec) · [가이드](../GITHUB.md) |
| **W&B** | [Berry2Vec](https://wandb.ai/dasom-oh/Berry2Vec) |
| **Label Studio** | [가이드](../LABELSTUDIO.md) · config: `labelstudio/projects/fruiting/config.xml` |
| **Registry** | [`registry.yaml`](registry.yaml) |

## 진행 요약 (2026-07-02)

| Exp ID | 제목 | Acc | MAE | GitHub | Label Studio | 상태 |
|--------|------|-----|-----|--------|--------------|------|
| [EXP-001](EXP-001_overfit_sanity_62p5.md) | **1차 확인** plain CE overfit | **62.5%** | **0.625** | [eee0b4e](https://github.com/OhDasom88/life2vec/commit/eee0b4e) | import 준비 | 완료 |
| [EXP-002](EXP-002_weighted_asymmetric_overfit.md) | weighted + asymmetric | 10% | 2.400 | [eee0b4e](https://github.com/OhDasom88/life2vec/commit/eee0b4e) | — | 실패 |
| EXP-003 | 본학습 split | — | — | `exp/EXP-003-finetune` | 계획 | 계획 |

## 버킷 정의

| 버킷 | 착과수 | 대표값 |
|------|--------|--------|
| 0 | ≤3 | 3 |
| 1 | 4 | 4 |
| 2 | 5 | 5 |
| 3 | 6 | 6 |
| 4 | 7 | 7 |
| 5 | ≥8 | 8 |

## 데이터 규모

- 전체 cohort: **40건**
- 기본 split: train 28 / val 6 / test 6
- Overfit split: train/val/test 각 40건
