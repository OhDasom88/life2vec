# 05. 보고서 생성 · 프롬프트

## 핵심 코드

| 역할 | 파일 |
|------|------|
| Gemma 호출·마크다운 기록 | `scripts/online2_v2/generate_gemma_reports.py` |
| evidence JSON 스캐폴드 | `src/online2/v2/finetune_v02/gemma_report.py` |

API 모드:

- `openai`: local `llama-server` `/v1/chat/completions`
- `generate`: remote `/generate`

## LLM에 전달되는 내용

### rich 모드 (`build_rich_payload` → `build_rich_user_prompt`)

대략 포함:

- 예측 진단명, ensemble 확률, 5-fold agreement
- top-5 class probabilities
- view×zone strata (`sum_abs_saliency`, agree 비율)
- 시간 버킷 strata
- **top events** (saliency·합의 순 선정 후 시간순 정렬): event_id, view, zone, timestamp, median_saliency, agree, (있으면) top_tokens
- disagree / alternative 진단 힌트

### compact / default

`evidence/{case_id}.json`을 압축해 user 메시지로 전달.

시스템 프롬프트는 모드별(`rich` / `cause` / `narrative` 등)로 섹션·분량 지시를 담습니다.  
실제 retained prompt는 `reports/.../prompts.jsonl`에 남을 수 있으나, 재생성 시 비어 있는 경우도 있음 → 그때는 evidence에서 payload를 재구성.

샘플 LLM 입력(evidence):  
`07_case_artifacts/F420458_…/05_evidence_llm_input.json`

## top event 선정

1. 평가 파이프라인: `select_report_events` → primary 등 → evidence `diagnosis_support`
2. 보고서 rich: consensus/primary 테이블을  
   `median_saliency`, `positive_agreement_count` 내림차순 → `head(top_events)` (기본 12)  
   → 서술용으로 timestamp 오름차순 재정렬

## 인용 토큰이 0이 되는 이유

`generate_gemma_reports.py`:

```python
def load_token_saliency(eval_dir):
    path = eval_dir / "layer_analysis" / "token_saliency_all.parquet"
    if not path.exists():
        return None
```

v0.2 평가 (`cv_20260714_052556_…`)에는 **`layer_analysis/`가 없음**.  
→ `top_tokens`가 전부 빈 리스트 → 마크다운 헤더 `인용 토큰: 0`.

토큰 인용을 쓰려면 `analyze_v01_layer_saliency.py` 등으로 token IxG를 돌려  
`layer_analysis/token_saliency_all.parquet`를 만들어야 합니다.  
(v0.1 eval `7czhwn2y_…`에는 유사 case 파일이 존재.)

부가 효과: 프롬프트에 **원시 ℃/%/dS/m가 거의 안 들어가고** saliency 메타만 강조됨.

## Markdown 렌더링

같은 스크립트 후반부:

- 출력: `reports/{tag}/markdown/{case_id}.md`
- 헤더: 진단, 합의, 근거 이벤트 수, 인용 토큰 수
- 본문: LLM completion 텍스트 그대로 기록

샘플: `07_case_artifacts/F420458_…/06_gemma_report.md`
