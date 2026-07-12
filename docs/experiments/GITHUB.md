# GitHub 실험 연동

**Repository**: [OhDasom88/life2vec](https://github.com/OhDasom88/life2vec)

> 현재 repo에서 **Issues가 비활성화**되어 있습니다. 실험 추적은 **커밋 + PR + Notion** 조합을 사용합니다. Issues를 켜면 `experiment` 라벨로 EXP별 issue를 추가할 수 있습니다.

## 브랜치

```
exp/EXP-NNN-<short-slug>
```

예: `exp/EXP-001-overfit-sanity`

## 커밋 메시지 (Conventional Commits)

```
exp(fruiting): EXP-001 overfit sanity — acc 62.5%

- Transformer_CLS 6-bucket classifier
- Notion: https://app.notion.com/p/39175b02ea8181549035c45e4ccbcf14
- W&B: offline-run-20260702_151024-up7kbrf0
```

## PR 템플릿 (본문)

```markdown
## Exp ID
EXP-NNN

## Notion
<링크>

## W&B
<run URL 또는 sync 명령>

## Label Studio
<프로젝트 URL 또는 N/A>

## 결과
- 버킷 Acc:
- MAE Count:

## 변경 요약
- 데이터:
- 모델:
- 파라미터:
```

## 커밋 URL 형식

```
https://github.com/OhDasom88/life2vec/commit/<sha>
```

## PR URL 형식

```
https://github.com/OhDasom88/life2vec/pull/<number>
```

## Issues 활성화 시 (선택)

Settings → Features → Issues ON 후:

```bash
gh label create "experiment" --color "5319E7" -R OhDasom88/life2vec
gh label create "fruiting" --color "0E8A16" -R OhDasom88/life2vec

gh issue create -R OhDasom88/life2vec \
  --title "EXP-001: 착과수 overfit sanity" \
  --label "experiment,fruiting"
```

## 실험별 GitHub 링크 (현재)

| Exp | Commit | PR | 비고 |
|-----|--------|-----|------|
| EXP-001 | [eee0b4e](https://github.com/OhDasom88/life2vec/commit/eee0b4ebe9cd4bc2488a5a820ba08d3d3652e517) | — | 분류 모델 코드 **미커밋** (working tree) |
| EXP-002 | [eee0b4e](https://github.com/OhDasom88/life2vec/commit/eee0b4ebe9cd4bc2488a5a820ba08d3d3652e517) | — | 동일 베이스 |
| EXP-003 | — | — | 계획 |

## Notion / registry 연동 필드

- `GitHub Commit`: short SHA + full URL
- `GitHub Issue`: issue URL (활성화 시) 또는 PR URL
- `Git`: registry.yaml `git_commit` 과 동일
