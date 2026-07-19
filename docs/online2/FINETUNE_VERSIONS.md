# Online2 Diagnosis Finetune — Version Map

| Version | Status | Scope |
|---------|--------|-------|
| **v0.1** | shipped / active baseline | Classification only (Stage A cache → PAD → 10-way head) |
| **v0.2** | active (sweep / eval / Gemma) | + semantic align (state/cause) + IMAGE adapter + consensus saliency + Gemma |
| **v0.3** | **active (Sweep1 + follow-up build)** | Multi-task heads (binary + 10-class + projection), Known/Unknown open-set, P0 evidence recovery → open diagnosis; CF as later track. See [`DIAGNOSIS_FINETUNE_V03_CV_REPORT.md`](./DIAGNOSIS_FINETUNE_V03_CV_REPORT.md), canonical plan [`DIAGNOSIS_FINETUNE_V03_PLAN.md`](./DIAGNOSIS_FINETUNE_V03_PLAN.md), follow-up [`DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md`](./DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md), initial sketch [`DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md`](./DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md) |

## Isolation rules

1. **Never edit older training entrypoints to “upgrade in place”.**
2. v0.2: `src/online2/v2/finetune_v02/`, `scripts/online2_v2/v02/`, `outputs/online2/v2_finetune_v02/`.
3. v0.3: `src/online2/v2/finetune_v03/`, `scripts/online2_v2/v03/`, `outputs/online2/v2_finetune_v03/`.
4. Newer versions may **import** older PAD/dataset helpers but must not mutate them.
5. Artifacts stay in **separate roots** so caches/ckpts cannot overwrite each other.

## Path map

| Role | v0.1 | v0.2 | v0.3 |
|------|------|------|------|
| Plan | `docs/online2/DIAGNOSIS_FINETUNE_PLAN.md` | `docs/online2/DIAGNOSIS_FINETUNE_V02.md` | **`docs/online2/DIAGNOSIS_FINETUNE_V03.md`** (+ [`DIAGNOSIS_FINETUNE_V03_PLAN.md`](./DIAGNOSIS_FINETUNE_V03_PLAN.md), [`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md), [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md)) |
| Pipeline trace | — | `docs/online2/pipeline_trace_v02/` | (reuse v02 trace; v03 deltas in V03 plan) |
| Model | `event_pooling_finetune.py` | `finetune_v02/model.py` | `finetune_v03/` (planned) |
| Train / CV | `run_diagnosis_event_pooling_finetune.py` | `v02/run_diagnosis_finetune_v02.py` | `v03/…` (planned) |
| Ckpt naming | `fold{i}_best.pt` | same + `finetune_version=0.2.0` | same + `0.3.0` |
| Eval | `run_evaluation_pipeline.py` | `v02/run_evaluation_pipeline_v02.py` | `v03/…` (planned) |
| Output root | `outputs/online2/v2_finetune/` | `…/v2_finetune_v02/` | **`…/v2_finetune_v03/`** |

## v0.3 vs v0.2 (short)

| | v0.2 | v0.3 |
|--|------|------|
| Primary decision | 10-class softmax | binary → Known/Unknown → fine or open |
| Checkpoints | 5 multi-task (cls+semantic) | **5 multi-task** (binary+fine+proj); split to 10 only if negative transfer |
| Open-set | none (forced map) | prototype / energy / agreement + LODO calibration |
| Report gate | rich saliency | **P0** raw+token before open LLM |
| Counterfactual | — | P4 optional; not current MLM-as-WM |

## Freeze checklists

### Do not change when iterating v0.2

- `src/online2/v2/event_pooling_finetune.py`
- `scripts/online2_v2/cache_stage_a_event_embeddings.py`
- `scripts/online2_v2/run_diagnosis_event_pooling_finetune.py`
- `outputs/online2/v2_finetune/{event_embeddings,runs}/` (read-only OK)

### Do not change when iterating v0.3

- All of the above, plus v0.2 entrypoints under `scripts/online2_v2/v02/` and `src/online2/v2/finetune_v02/`
- `outputs/online2/v2_finetune_v02/` (read Stage A / ckpt OK)

v0.3 may **read** v0.1/v0.2 Stage A caches; writes only under `v2_finetune_v03/`.
