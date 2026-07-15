# Online2 진단 Finetune v0.2 구조안

상태: **preparation scaffold**  
버전: `0.2.0`  
작성일: 2026-07-14  

v0.1과의 관계·경로 분리: [`FINETUNE_VERSIONS.md`](./FINETUNE_VERSIONS.md)

> **Isolation:** v0.1 코드·스크립트·`outputs/online2/v2_finetune/` 산출물을 수정하지 않는다.  
> v0.2는 `src/online2/v2/finetune_v02/`, `scripts/online2_v2/v02/`, `outputs/online2/v2_finetune_v02/` 에만 쓴다.

---

## 1. 최종 목표

1. **진단명 분류** — 10클래스; fold별 train/val F1=1.0 후보만 채택; OOF 35 Macro-F1=1.0
2. **상태진단·원인분석 의미 정렬** — H/M/L 해석문 임베딩에 품질 가중 cosine alignment
3. **근거 추출 + Gemma 보고서** — 5모델 consensus saliency → Structured Evidence → 소형 멀티모달 Gemma

원칙: semantic head를 classification logits에 **fusion하지 않음**.

---

## 2. 파이프라인 (요약)

```text
Case observations
  → Stage A contextual event encoder (≤1024-token windows; IMAGE slots ← DINO+adapter)
  → event summary sequence [e1…eT] + metadata
  → Stage B PAD → h_case
      ├─ C-1 Diagnosis Head → 10 logits (CE)
      └─ C-2 Semantic Heads → z_state, z_cause (quality-weighted align)
  → 5-fold models → probability ensemble
  → consensus saliency → Structured Evidence JSON
  → Gemma multimodal report
```

---

## 3. Loss (초기)

```yaml
loss:
  classification_ce: 1.0
  state_alignment: 0.10
  cause_alignment: 0.15
  quality_ranking: 0.00   # 분류 안정 후 활성화
quality_weights: { high: 1.0, medium: 0.5, low: 0.1 }
```

원인 alignment를 상태보다 약간 높게 — classification head가 이미 상태 라벨을 직접 학습.

원인 텍스트에서는 진단명 문자열을 `[DIAGNOSIS_MASK]`로 치환.

---

## 4. Fold 채택 조건

| 단계 | 지표 | 기준 |
|------|------|------|
| Fold train/val | present-label Macro-F1 | = 1.0 |
| OOF 35 | accuracy + 10-class Macro-F1 | = 1.0 |

Problem 추론은 5개 fold 확률 평균 앙상블.

---

## 5. Saliency

| 대상 | 위치 |
|------|------|
| Event | `z_i` (Stage A summary + metadata LN 후, PAD 직전) |
| Token | 합의 event에 한해 Stage A target hidden `H_target` |
| Image | projected DINO residual / slot mask / leave-one-image-out |

세 축: `diagnosis_support` / `state_evidence` / `cause_evidence`  
합의: median of fold-normalized scores + positive agreement ≥4/5

Example case OOF 설명은 **해당 case를 val로 둔 fold 1개만** 사용.

---

## 6. 산출물 루트

```text
outputs/online2/v2_finetune_v02/
  folds/fold_{0..4}/model.ckpt
  ensemble/
  saliency/{per_model,consensus}/
  reports/
  interpretation_banks/
  event_embeddings/          # IMAGE-included Stage A (optional)
```

---

## 7. 코드 진입점

| 단계 | 스크립트 |
|------|----------|
| 해석 임베딩 bank | `scripts/online2_v2/v02/build_interpretation_banks.py` |
| Stage A + IMAGE slots | `scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py` |
| Train / 5-fold CV | `scripts/online2_v2/v02/run_diagnosis_finetune_v02.py` |
| Ensemble infer | `scripts/online2_v2/v02/run_ensemble_infer_v02.py` |
| Saliency consensus | `scripts/online2_v2/v02/run_saliency_consensus_v02.py` |
| Gemma inputs/reports | `scripts/online2_v2/v02/generate_gemma_reports_v02.py` |

패키지: `src.online2.v2.finetune_v02`  
PAD 블록은 v0.1 `event_pooling_finetune`에서 **import만** 한다.

---

## 8. 구현 상태 (B-track)

| 구성요소 | 상태 |
|----------|------|
| Text emb → state/cause **alignment only** | ready (Qwen via `embed_interpretation_bank_qwen.py`) |
| Image emb → IMAGE event / slot residual | ready (DINO attach + fold-trainable adapter) |
| Stage A IMAGE include | ready (`cache_stage_a_image_slots_v02.py`) |
| Ckpt `fold{i}_best.pt` + eval auto-detect | ready |
| Token-slot bake (`--inject-slots`) | optional experiment |

### 학습 시작

```bash
python scripts/online2_v2/v02/build_interpretation_banks.py
python scripts/online2_v2/v02/embed_interpretation_bank_qwen.py
python scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py --device cuda

python scripts/online2_v2/v02/run_diagnosis_finetune_v02.py \
  --mode cv --epochs 150 --device cuda --monitor val_macro_f1 \
  --emb-dir outputs/online2/v2_finetune_v02/event_embeddings \
  --interpretation-bank outputs/online2/v2_finetune_v02/interpretation_banks/state_cause_bank.parquet
```
