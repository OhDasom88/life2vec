# CF M1 구현 메모 (사전 조사 + 완료 상태)

작성: 2026-07-17  
정본: `CF_M1_ATTRIBUTION_LOCUS_ARCHITECTURE_REVIEW.md`, `CURSOR_PROMPT_CF_M1_ATTRIBUTION_LOCUS_IMPLEMENTATION.md`

## 실제 경로 vs 설계

| 항목 | 설계 | 실제 |
|------|------|------|
| CF 패키지 | `finetune_v03/counterfactual/` | `src/online2/v2/finetune_v03/counterfactual/` |
| Risk / event IxG | fuse-aware | `risk_saliency.event_ixg_abnormal_margin` |
| Critic ΔR | Stage A + 동결 진단 | `evaluation/score_candidates.py` + `diagnosis_critic.ensemble_delta_r` |
| Stage A 재인코딩 | token→encoder | **cache perturbation** (진단 inference와 동일 fuse 경로); mode 필드 기록 |
| Raw join | evidence | `evaluation/raw_join.py` → `raw_join_metrics.json` |
| Span↔event | edit_scope | `temporal/span_event_align.py` zone+timestamp |
| Binning decode | 필요 | `grounding/raw_target.decode_interval` |
| Checkpoint folds | 5 | 이 run은 **3 folds** |

## 완료된 M1 갭 (후속 작업)

- [x] Path A `delta_r` fold critic
- [x] Path B `model_space_delta_r` 실측 (placeholder 제거)
- [x] NO_OP sanity (ΔR≈0)
- [x] raw join success rate
- [x] span–event alignment (~99.6%)
- [x] wandb metrics summary JSON
- [x] Gate4 retokenize + eligibility 가드

## 의도적 M1 제한

- B1 capacity / response model / OPERATIONAL_CANDIDATE 금지
- Path B는 물리 반응이 아닌 classifier-oriented `model_space_delta_r`
- Attribution 부호 → 물리 증감 직접 매핑 금지
