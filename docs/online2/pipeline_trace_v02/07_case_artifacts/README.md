# 07. 사례 중간 산출물 — `F420458_2025-02-16_2025-03-01`

- 라벨: **정상_운영** (예시 35)
- 평가: `cv_20260714_052556_20260714_124217`
- 이벤트 캐시: **3929** events, mean/max dim **384**

이 폴더 파일은 **전체 복제가 아니라 head/샘플**입니다. 원본 경로는 아래 표.

## 파일 목록

| 파일 | 의미 |
|------|------|
| `00_label_row.csv` | labels CSV 해당 행 (있으면) |
| `00_fold_info.json` | fold membership hit |
| `01_cell_occurrences_sample30.csv` | 원시 cell (`raw_value`/`raw_display`) |
| `01_legacy_events_sample25.csv` | builder events |
| `02_events_tokenized_sample20.csv` | V2 SENTENCE / token_ids |
| `02_one_event_tokenized.json` | 이벤트 1건 전체 필드 예시 |
| `02_binning_registry_feature_sample.json` | ABS bin edges 샘플 |
| `03_model_input_embedding_meta.json` | cache 컬럼·건수 |
| `03_model_input_meta_sample15.csv` | 벡터 제외 메타 샘플 |
| `03_vector_dims.json` | mean/max 차원 |
| `04_*` | fold0 점수 · consensus/primary/strong/disputed |
| `05_evidence_overview.json` | evidence 키 요약 |
| `05_evidence_llm_input.json` | LLM/evidence 전체 JSON |
| `06_gemma_report.md` | rich 보고서 |

## 원본 전체 경로

| 단계 | 경로 |
|------|------|
| raw CSV | `datasets/agrichallenge/online2/data/{E_environment,A_actuator}/F420458_z*.csv` |
| score90 | `datasets/agrichallenge/online2/answers/reference_answers/score90/F420458_answer.txt` |
| cells/events | `outputs/online2/build-v8-active80-r3/{cell_occurrences,events}.parquet` |
| tokenized | `outputs/online2/v2_build/events_tokenized_v2.parquet` |
| model input | `outputs/online2/v2_finetune_v02/event_embeddings/F420458_2025-02-16_2025-03-01.parquet` |
| event saliency | `…/saliency/per_model/fold*/…` , `…/saliency/consensus/F420458_…/` |
| token attribution | **없음** (`layer_analysis/` 미생성) |
| LLM 입력 | `…/evidence/F420458_….json` |
| markdown | `…/reports/gemma4_local_rich/markdown/F420458_….md` |

## token attribution

이 사례(및 해당 v0.2 eval)에서는 **없음** → 보고서 `인용 토큰: 0`.  
필요 시 v0.1 쪽:

`outputs/online2/v2_finetune/evaluation/7czhwn2y_20260713_212502/layer_analysis/cases/F420458_…/`
