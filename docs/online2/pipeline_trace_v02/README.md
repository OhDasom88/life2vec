# Online2 V2 파이프라인 추적 (v0.2 기준)

진단 분류·saliency·Gemma 보고서까지 **코드 경로와 중간 산출물**을 한곳에 정리한 디렉토리입니다.

기준 평가 런: `outputs/online2/v2_finetune_v02/evaluation/cv_20260714_052556_20260714_124217`  
기준 사례: `F420458_2025-02-16_2025-03-01` (예시 35건, 정상_운영)

## 디렉토리 구조

| 경로 | 내용 |
|------|------|
| [END_TO_END.md](END_TO_END.md) | **위치·CLI·eval→보고서 진입점 점검 요약** |
| [01_sequence.md](01_sequence.md) | raw → event, same-time, zone/channel, 구동기 |
| [02_tokenizer.md](02_tokenizer.md) | vocab, bin 범위, raw 보존 위치 |
| [03_model.md](03_model.md) | Stage A pooling, PAD, head, fold inference |
| [04_saliency.md](04_saliency.md) | IxG, abs 사용처, 5-fold consensus, 저장 형식 |
| [05_report.md](05_report.md) | LLM 입력, top event, 인용 토큰 0, markdown |
| [06_labels.md](06_labels.md) | 35건 label map, 정상 정의, farm CV split |
| [07_case_artifacts/](07_case_artifacts/) | 사례 1건 중간 산출물 샘플 |
| [INDEX_CODE_PATHS.md](INDEX_CODE_PATHS.md) | 파일·함수 빠른 색인 |
| [SOURCE_EXCERPTS/](SOURCE_EXCERPTS/) | **실제 구현 복사·발췌** (패치 설계용) |
| [SOURCE_EXCERPTS/WORLD_MODEL_REUSE.md](SOURCE_EXCERPTS/WORLD_MODEL_REUSE.md) | 1–5번 기준 월드모델 재사용 판정 |

관련:

- **`docs/online2/DIAGNOSIS_FINETUNE_V03.md`** — v0.3 멀티태스크·open-set 설계 (이 trace는 v0.2 구현 기준)
- `docs/online2/FINETUNE_VERSIONS.md` — 버전 맵
- `docs/online2/DIAGNOSIS_FINETUNE_V02.md` — v0.2 계획
- `outputs/online2/v2_finetune/EVALUATION_PROCESS.md` — **v0.1** 평가 절차
- `outputs/online2/v2_finetune/EVENT_POOLING_FINETUNE_ARCH.md` — PAD 아키텍처

## 한 줄 흐름

```text
CSV
 → builder (events / same_time_groups / cell_occurrences)
 → build_v2 (TokenizerV2 → events_tokenized_v2 → sequences_v2)
 → Stage A cache (event_mean / event_max, 384-d)
 → PAD + DiagnosisHead (5-fold)
 → event IxG saliency → consensus
 → evidence JSON → Gemma rich prompt → markdown
```

## 중요 제한 (추적 결과)

1. **닫힌 10-class**: 예시 35건 진단명만 출력 가능.
2. **v0.2 평가 기본 경로에 token saliency 없음** → 보고서 `인용 토큰: 0`.
3. **raw float는 tokenized event parquet에 없음** → cell_occurrences / builder 단계에서만 보존.
4. **농가 단위 Group K-fold** → 같은 farm이 train/val에 동시에 들어가지 않음.
