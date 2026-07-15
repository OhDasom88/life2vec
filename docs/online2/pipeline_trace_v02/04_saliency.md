# 04. Saliency 생성

## event saliency 계산

파일: `src/online2/v2/finetune_v02/saliency.py`  
함수: `event_input_x_gradient`

1. `z = meta(event_mean, event_max, …)` 후 `z.requires_grad_(True)`
2. `h_case = pad(z)` → `logits = head(h_case)` (또는 state/cause 목적)
3. 선택 class logit에 대해 `backward`
4. 점수: `(z.grad * z).sum(dim=-1)` — **Input×Gradient, signed**
5. padding mask 곱

기본 타깃: classification (앙상블/예측 클래스).

## token saliency 계산 여부

| 경로 | 상태 |
|------|------|
| v0.2 evaluation pipeline | **deferred** — event IxG만 |
| 별도 스크립트 | `scripts/online2_v2/analyze_v01_layer_saliency.py` + `src/online2/v2/v01_layer_saliency.py` (Stage A 재인코딩) |

기준 eval 디렉터리에는 `layer_analysis/`가 없어 **token attribution 산출물이 없음**.

## 절댓값(abs)을 쓰는 위치

| 위치 | abs? | 설명 |
|------|------|------|
| `event_input_x_gradient` 점수 | ❌ | signed IxG |
| consensus `normalize_fold_scores` | ❌ | percentile rank (또는 MAD z) |
| `positive_agreement_count` | raw `> 0` 카운트 | 부호 사용 |
| report `_view_zone_strata` / `_time_strata` | ✅ `abs(median_saliency)` | 층위 합산용 `sum_abs` |
| token 경로 | ✅ `abs_saliency` (있을 때) | top-token 정렬 |

## 5-fold consensus

파일: `src/online2/v2/finetune_v02/consensus.py`

1. fold별 점수를 percentile로 정규화
2. event마다 `median_saliency`, `mean`, `IQR`
3. `positive_agreement_count` = signed raw > 0인 fold 수
4. `top_rank_agreement_count` = top-frac 안에 들어간 fold 수
5. `select_report_events`:
   - **primary**: agree≥min & top-rank≥min (기본 ≥4 / ≥3 부류)
   - **strong / disputed / oof_only** 분할

## 저장 형식

평가 루트 예:  
`…/evaluation/cv_20260714_052556_20260714_124217/saliency/`

```text
per_model/fold{k}/{case_id}_event_scores.parquet
  columns: event_id, ixg_score, attn_weight, …

consensus/{case_id}/
  event_consensus.parquet
  event_primary.parquet
  event_strong.parquet
  event_disputed.parquet
  event_oof_only.parquet   # 있으면
  meta.json
```

evidence로 합쳐짐: `…/evidence/{case_id}.json`
