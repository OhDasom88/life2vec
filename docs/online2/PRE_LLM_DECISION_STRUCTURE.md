# Pre-LLM 의사결정 산출 구조 (Structure)

상태: 스키마·디렉터리 계약  
대상: `outputs/m1/decision_pre_llm/`

---

## 1. 디렉터리 트리

```text
outputs/m1/decision_pre_llm/
├── diagnosis_model_manifest.json      # 동결 ckpt hash
├── feature_semantics_registry.yaml
├── feature_semantics_audit.csv
├── actuator_semantics_audit.csv
├── cohort_summary.csv                 # 55건 요약표
├── cohort_summary.json
├── batch_run.log
└── cases/
    └── {case_id}/
        ├── decision_package.json      # 1급 산출 (LLM 입력 후보)
        ├── diagnosis.json
        ├── event_ixg.parquet
        ├── mg_attribution.parquet
        ├── event_attribution.parquet
        ├── actuator_spans.parquet
        ├── span_alignment.json
        ├── loci_path_a.jsonl
        ├── loci_path_b.jsonl
        ├── path_a_candidates.jsonl
        └── path_b_candidates.jsonl
```

---

## 2. `decision_package.json` 스키마 (v1)

```json
{
  "schema_version": "pre_llm_decision.v1",
  "created_at": "ISO-8601",
  "case_id": "F…_YYYY-MM-DD_YYYY-MM-DD",
  "llm_boundary": "STOPPED_BEFORE_LLM",
  "stages_completed": ["pretrain_stage_a_cache", "diagnosis_v03_ensemble", "..."],
  "stages_not_run": ["open_set_retrieval_llm", "gemma_report_generation", "..."],
  "diagnosis": {
    "set": "example_set|problem_set",
    "ground_truth_diagnosis": "… or null for problem",
    "p_abnormal_mean": 0.0,
    "fine_pred_label": "…",
    "fine_probs": {"진단명": 0.0},
    "route": "NORMAL|KNOWN|UNKNOWN_ABNORMAL|REJECT",
    "route_reasons": [],
    "fold_agreement": 0.0,
    "llm_stage": "NOT_RUN"
  },
  "counts": {
    "path_a_loci": 0,
    "path_b_loci": 0,
    "path_a_candidates": 0,
    "path_b_candidates": 0,
    "spans": 0,
    "span_aligned_rate": 0.0
  },
  "top_path_a": [],
  "top_path_b": [],
  "recommendation_gate": {
    "analysis_report": true,
    "management_suggestion": false,
    "llm_narrative": false
  }
}
```

---

## 3. 모듈 ↔ 산출 매핑

| 단계 | 코드 | 산출 |
|------|------|------|
| Manifest | `counterfactual/manifest.py` | `diagnosis_model_manifest.json` |
| Semantics | `semantics_registry.py` | `feature_semantics_registry.yaml` |
| Diagnosis | `run_pre_llm_decision_batch_v03.diagnose_case` | `diagnosis.json` |
| Attribution | `attribution/token_attribution.py` | `*_attribution.parquet` |
| Temporal | `temporal/span_builder.py` + `span_event_align.py` | `actuator_spans.parquet` |
| Locus | `locus/selector.py` | `loci_path_*.jsonl` |
| Path A | `candidates/path_a_generator.py` + gates | `path_a_candidates.jsonl` |
| Path B B0 | `candidates/path_b_b0_generator.py` | `path_b_candidates.jsonl` |
| Batch CLI | `scripts/.../run_pre_llm_decision_batch_v03.py` | `cohort_summary.*` |

---

## 4. 라우팅 구조

```text
p_abnormal, fine_probs, fold_agree_count, energy
        ↓
   route_case()
        ├── NORMAL
        ├── KNOWN          → Path A/B 추천 의미 있음
        ├── UNKNOWN_ABNORMAL → 분석 가능, LLM 후속 후보 (본 배치에서는 LLM 미호출)
        └── REJECT         → 낮은 신뢰, 개입 제안 억제 권고
```

Threshold는 `RoutingThresholds` (binary/fine τ, fold 합의).  
3-fold run에서는 `min_fold_agreement`를 fold 수에 맞게 클램프한다.

---

## 5. Path A / Path B 출력 의미

| Path | 객체 | 의미 | 적격성 |
|------|------|------|--------|
| A | `grounded` target interval | 모델상 목표 **관측 상태** | EXPLANATORY_ONLY ~ EXPERT_REVIEW |
| B | `edit_scope` operation | actuator **구간 편집** (용량 unknown) | EXPERT_REVIEW_REQUIRED |
| — | `token_edits` | Gate4 후 파생 로그 | 직접 제어 명령 아님 |

`OPERATIONAL_CANDIDATE`는 본 구조에서 **생성하지 않는다**.

---

## 6. 데이터 의존성

| 입력 | 경로 |
|------|------|
| 55 labels | `outputs/online2/v2_finetune_v02/labels_example35_problem20.csv` |
| Embeddings | `outputs/online2/v2_finetune_v02/event_embeddings/{case_id}.parquet` |
| Cells | `build-v8-active80-r3/cell_occurrences.parquet` |
| Binning | `v2_build/binning_registry_v2_transductive.json` |
| Feature audit | `v2_audit/feature_semantics_and_units.csv` |

---

## 7. LLM 연결 포인트 (후속)

`decision_package.json` + `loci_*.jsonl` + `path_*_candidates.jsonl`을  
근거 패키지로 묶어 P3 LLM에 전달하면 된다.  
본 구조는 그 입력을 **고정 스키마로 동결**하는 계층이다.
