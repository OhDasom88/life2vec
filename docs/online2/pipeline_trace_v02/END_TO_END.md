# End-to-end: 데이터 → 보고서 추출

이 문서는 `pipeline_trace_v02`에서 **빠져 있던 실행 진입점**을 보완합니다.  
단계 설명은 `01`–`06`, 구현 스냅샷은 `SOURCE_EXCERPTS/`를 보세요.

## 문서 위치 (정본)

| 용도 | 경로 |
|------|------|
| **전체 추적 정본** | `docs/online2/pipeline_trace_v02/` |
| 단계 문서 | `01_sequence.md` … `06_labels.md` |
| 사례 중간 산출물 | `07_case_artifacts/F420458_…/` |
| 소스 스냅샷 | `SOURCE_EXCERPTS/` |
| v0.1 평가 프로세스(별도) | `outputs/online2/v2_finetune/EVALUATION_PROCESS.md` |
| v0.2 계획/버전 | `docs/online2/DIAGNOSIS_FINETUNE_V02.md`, `FINETUNE_VERSIONS.md` |

## 파이프라인 → 산출물

```text
[빌드] CSV → builder → build_v2 tokenize
   → outputs/online2/v2_build/{events_tokenized_v2,sequences_v2,…}

[Stage A] cache_stage_a_event_embeddings.py
   → outputs/online2/v2_finetune_v02/event_embeddings/{case}.parquet
   (IMAGE/DINO: cache_stage_a_image_slots_v02.py 별도)

[학습] v02/run_diagnosis_finetune_v02.py
   → outputs/online2/v2_finetune_v02/runs/cv_*/fold{k}_best.pt

[평가] v02/run_evaluation_pipeline_v02.py
   → …/evaluation/<stamp>/{ensemble,saliency,evidence}/

[보고서] generate_gemma_reports.py --eval-dir <stamp>
   → …/reports/<tag>/markdown/{case}.md
```

## 기준 스탬프 (문서에 쓰인 것)

- run: `outputs/online2/v2_finetune_v02/runs/cv_20260714_052556`
- eval: `outputs/online2/v2_finetune_v02/evaluation/cv_20260714_052556_20260714_124217`
- reports: `…/reports/gemma4_local_rich/markdown/` (55건)

## 재실행 CLI (요약)

```bash
# 1) 평가 + event saliency + evidence JSON
python scripts/online2_v2/v02/run_evaluation_pipeline_v02.py \
  --run-dir outputs/online2/v2_finetune_v02/runs/cv_20260714_052556 \
  --saliency-cases all

# 2) Gemma markdown (local llama-server 가정)
python scripts/online2_v2/generate_gemma_reports.py \
  --eval-dir outputs/online2/v2_finetune_v02/evaluation/<stamp> \
  --out-dir  outputs/online2/v2_finetune_v02/evaluation/<stamp>/reports/gemma4_local_rich \
  --api-mode openai \
  --api-base http://127.0.0.1:8080/v1 \
  --prompt-style rich \
  --top-events 12
```

평가 파이프라인은 **Markdown을 직접 쓰지 않고** evidence까지입니다.  
스크립트 note: `Live Gemma markdown via generate_gemma_reports.py --eval-dir <this> …`.

## 점검에서 확인된 내용 (2026-07-15)

| 항목 | 상태 |
|------|------|
| 01–06 단계 서술 → 보고서까지 논리 연결 | OK |
| `07` 사례 sample (evidence·md·saliency head) | OK, 존재 |
| 인용 토큰 0 원인 (`layer_analysis` 부재) | OK, `05_report`·샘플 md에 반영 |
| `SOURCE_EXCERPTS` 1–8 우선순위 소스 | OK |
| eval→report CLI가 본 README에 명시 | **보완: 이 파일** |
| `EVALUATION_PROCESS.md` | v0.1 중심 — v0.2와 혼동 주의 |
| 샘플 md 모델명 `gemma-4-12B-it` | CLI 기본값; 로컬은 31B Q4일 수 있음 |

## 알려진 문서 공백 (의도/잔여)

1. 빌드(CSV→v2_build) full CLI 레시피는 여기보다 `scripts/online2_v2/build_v2.py` / runbook 쪽이 상세.
2. Token saliency를 v0.2 eval에 붙이는 운영 절차는 `analyze_v01_layer_saliency.py` 별도.
3. `SOURCE_EXCERPTS`는 스냅샷 — 패치 시 repo 원본 기준.
