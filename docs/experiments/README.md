# Berry2Vec / Life2Vec 실험 기록

실험 기록은 **5곳**에 연동됩니다.

| 계층 | 도구 | 역할 |
|------|------|------|
| **대시보드** | **Notion** | 전체 진행, 실험 비교, 한줄 결론 |
| **수치·곡선** | **Weights & Biases** | loss, acc, hparams, run 비교 |
| **코드·재현** | **GitHub** | 커밋, PR, 브랜치, 코드 diff |
| **라벨·검수** | **Label Studio** | 오답 검수, 버킷 라벨 확인, 예측 QA |
| **상세 문서** | **로컬 (`docs/experiments/`)** | 설정, 변경 이력, 체크포인트 경로 |

```
Notion (허브)
  ├── W&B run ──────────── 메트릭·hparams
  ├── GitHub commit/PR ─── 코드 스냅샷
  ├── Label Studio ─────── 오답·라벨 검수
  └── docs/experiments/ ── 상세 재현 정보
```

## ID 규칙

| ID | 형식 | 예 |
|----|------|-----|
| 실험 | `EXP-NNN` | EXP-001 |
| Notion | `FRT-NNN` | FRT-001 |
| W&B tag | `exp-NNN`, `fruiting` | |
| Git branch | `exp/EXP-NNN-<slug>` | `exp/EXP-001-overfit-sanity` |
| Git commit | `exp(fruiting): EXP-NNN ...` | Conventional Commits |
| Label Studio project | `berry2vec-fruiting-EXP-NNN` | |

## 새 실험 추가 절차

1. `docs/experiments/fruiting/EXP-NNN_<slug>.md` 작성 ([`_template.md`](_template.md))
2. `docs/experiments/fruiting/registry.yaml` 업데이트
3. Git: `exp/EXP-NNN-*` 브랜치 → PR (본문에 Notion·W&B 링크)
4. 학습 실행 → W&B tag `exp-NNN`
5. (선택) Label Studio에 오답 샘플 import → 검수
6. Notion DB 행 추가/업데이트

## 상세 가이드

- [GitHub 연동](GITHUB.md)
- [Label Studio 연동](LABELSTUDIO.md)

## W&B 오프라인 sync

```bash
wandb sync logs/agri/wandb/offline-run-<run_id>
```

## 태스크별 인덱스

- [착과수 (fruiting)](fruiting/README.md)

## Notion

- [Berry2Vec 착과수 실험 기록](https://app.notion.com/p/39175b02ea8181b89a81fd7422f447d3)
- [착과수 실험 DB](https://app.notion.com/p/dab8a00f3dad4488bfe5607c6c5365c0)

## GitHub

- [OhDasom88/life2vec](https://github.com/OhDasom88/life2vec)
