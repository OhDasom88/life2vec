# Finetune v0.3 개선 계획 (Canonical)

| 항목 | 값 |
|------|-----|
| 버전 | `0.3.0` |
| 기준 run | `cv_20260715_104037_wandb` |
| Wandb | [dqrr69z6](https://wandb.ai/dasom-oh/Berry2Vec/runs/dqrr69z6) |
| CV 상세 | [`DIAGNOSIS_FINETUNE_V03_CV_REPORT.md`](./DIAGNOSIS_FINETUNE_V03_CV_REPORT.md) |
| 로드맵 | [`ROADMAP_V03.md`](./ROADMAP_V03.md) |
| Isolation | `finetune_v03/` · `scripts/.../v03/` · `outputs/.../v2_finetune_v03/` only |
| 작성일 | 2026-07-16 |
| 상태 | **Canonical improvement plan** (사용자 확정안) |

> **Callout — 문서 역할**  
> 본 문서(`DIAGNOSIS_FINETUNE_V03_PLAN.md`)가 v0.3 개선의 **정본(canonical) 상세 계획**입니다.  
> [`DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md`](./DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md)는 초기 CV 분석·튜닝 스케치로 유지합니다.

---

## 사용자 확정 개선 방향 (요약)

- **정확한 해석·라우팅:** 예측 확률이 낮을 때 무리하게 확정 진단(Known)을 내리지 않고 판정을 보류(Reject)하거나 미지(Open-set)로 넘기는 라우팅 전략 적용. Sweep을 통해 파라미터를 조정하되 **적용하지 않는 선택지**도 Sweep에 포함.
- **Binary 분석:** 필요한 로그 전부 추가. threshold 기본값 0.5, 튜닝 시 0.25 단위. `pos_weight_scale` 대신 **pos_weight 절대값** 적용. default는 자동 balanced weights(Sweep hyperparameter에도 동일).
- **Fine–Binary 불일치:** 측정용 로그 전부 적용 + Consistency Loss 적용.
- **SupCon:** prototype/center loss 적용 + 관련 로그 적용.
- **CV 설계:** repeated group, **3-fold**로 하이퍼파라미터 선택 → 추론은 **5-fold**.
- **Early stopping:** `val_loss` 사용.
- **튜닝범위:** 3단계 Sweep, 동일 W&B **Group**으로 한 화면에서 비교.
- **구조 변경:** 설계 변경 우선순위 1·2·4 적용. 4번(Attention token pooling)은 Attention 이용.
- **Stage A cache DINO 포함** = DINO vector가 cache에 별도 저장.
- **Projection/Open-set:** 학습·CV default와 추론 default를 분리(누수 방지 규칙은 학습에만).
- **이미지:** adapter 튜닝은 하지 않음(이미지는 계속 사용 — 병명 판별에 필수).
- **Open-set 개선 방향** 적용.

위 개선 방향을 바탕으로 작성한 아래 개선안을 문서와 코드에 반영한다.

### 코드 반영 상태 (2026-07-16)

| 영역 | 상태 | 위치 |
|------|------|------|
| pos_weight absolute / auto | 반영 | `pos_weight.py`, train CLI `--pos-weight` |
| consistency + prototype + SupCon(λ=0 가능) | 반영 | `losses.py` |
| Binary/thr/consistency 로그 | 반영 | `metrics.py`, `*_oof_*.csv` |
| Reject / Known / UNKNOWN 라우팅 | 반영 | `open_set.py` |
| Repeated 3-fold / final 5-fold modes | 반영 | `cv.py`, `--mode cv_repeated\|cv_final5` |
| Early stop `val_loss` default | 반영 | train `--monitor val_loss` |
| Task-specific query + attn residual | 반영 | `model.py` |
| Token→event pool (cache 있을 때 stub) | 부분 | `--token-attention-pool` (token cache 없으면 no-op에 가깝음) |
| Image adapter 고정 ON | 반영 | sweep에서 true 고정 |
| 3-stage sweep + 동일 group | 반영 | `conf/sweep/finetune_v03_s{1,2,3}_*.yaml` |
| Token hidden Stage A 재캐시 | 미완 | P3 후속 |
| LODO / risk–coverage 전체 | 부분 | 라우팅·로그 기반; 전용 eval 스크립트 후속 |

---

# Finetune v0.3 개선안

## 정상·비정상, Open-set, Attention Pooling 및 Counterfactual 추론 기반

---

## 1. 개선 목표

현재 `finetune_v03`의 Fine 10-class, Binary 정상·비정상, Projection 멀티태스크 구조는 유지하되 다음 문제를 해결한다.

1. 낮은 예측 확률에서도 기존 10개 진단 중 하나를 강제로 출력하는 문제
2. Binary head의 threshold와 불균형 보정이 고정된 문제
3. Fine과 Binary head가 서로 다른 정상·비정상 판정을 내리는 문제
4. SupCon이 소규모 batch에서 충분한 positive pair를 확보하지 못하는 문제
5. fold별 validation 표본이 너무 적어 모델 선택과 튜닝이 불안정한 문제
6. Stage A의 고정 mean/max pooling으로 이벤트 내부 token 정보가 손실되는 문제
7. fold별 Projection 공간을 직접 합치는 경우 발생하는 좌표계 불일치 문제
8. 신규 이상을 기존 진단으로 강제 매핑하는 문제
9. Counterfactual 편집 결과를 평가할 안정적인 정상성 위험 함수가 필요한 문제

최종 목표 추론 구조는 다음과 같다.

```text
입력 시퀀스
  ↓
Binary 정상·비정상
  ├─ 정상
  └─ 비정상
       ↓
Fine confidence + prototype + energy + model disagreement
  ├─ Known abnormal
  ├─ UNKNOWN_ABNORMAL
  └─ REJECT / REVIEW
       ↓
근거 이벤트·token·원시 값 복원
       ↓
검색 기반 열린 진단
       ↓
Counterfactual 시퀀스 편집
       ↓
정상 위험 감소량 기반 관리 개입 제안
```

---

## 2. 개선 모델 구조

### 2.1 전체 구조

```text
Stage A token hidden states
  ├─ 기존 event_mean / event_max
  └─ AttentionEventPooler
           ↓
  Gated Event Representation
           ↓
  DINO vector → SharedImageAdapter → IMAGE event residual
           ↓
  EventMetadataEncoder → z [B,T,H]
           ↓
  Shared MHA key/value
     ├─ Fine task query       → h_fine
     ├─ Binary task query     → h_binary
     └─ Projection task query → h_proj
           ↓
     ├─ Fine 10-class head
     ├─ Binary abnormal head
     └─ Projection embedding
```

### 주요 변경사항

* Fine, Binary, Projection이 하나의 `h_case`만 공유하지 않고 **task-specific learned query**를 사용한다.
* MHA의 key/value projection은 공유해 파라미터 증가를 제한한다.
* Stage A의 mean/max pooling을 제거하지 않고 **Attention token pooling을 residual로 추가**한다.
* 이미지 adapter는 항상 활성화한다.
* DINO vector는 Stage A cache 안에 저장된 별도 vector이며, Stage B에서 IMAGE 이벤트 표현에 융합한다.
* 이미지 사용 여부는 Sweep 대상에서 제외한다.

---

### 2.2 Attention 기반 token→event pooling

현재 구조:

```text
token hidden states
→ mean/max
→ event representation
```

개선 구조:

$$
h_{\text{attn},e} = \sum_{j \in e}\alpha_{e,j}h_{e,j}
$$


$$
\alpha_{e,j} = \operatorname{softmax} \left( q_e^\top W h_{e,j} \right)
$$


최종 event representation:

$$
h_e = \operatorname{LayerNorm} \left( W_{mm} [\operatorname{mean}(H_e);\operatorname{max}(H_e)] + g_e h_{\text{attn},e} \right)
$$


여기서 $g_e$는 학습 가능한 gate이다.

### 설계 원칙

* 기존 mean/max 정보를 유지한다.
* Attention pooling을 residual로 추가해 회귀 위험을 줄인다.
* Attention weight는 설명의 보조 지표로만 사용한다.
* 주요 근거는 signed IxG와 token perturbation으로 유지한다.

### 필요한 추가 cache

```text
event_token_hidden
event_token_mask
event_token_role
measurement_group_id
event token span
```

사전학습을 다시 수행하지 않고 기존 Encoder로 token hidden state만 재생성한다.

---

### 2.3 Task-specific query

```text
z [B,T,H]
  ├─ q_fine   → Fine attention → h_fine
  ├─ q_binary → Binary attention → h_binary
  └─ q_proj   → Projection attention → h_proj
```

Fine과 Binary가 서로 다른 이벤트에 집중할 수 있도록 query를 분리한다.

* Fine: 병명 구분에 필요한 이미지·생육·환경 이벤트
* Binary: 정상 범위 이탈을 나타내는 환경·근권·구동기 이벤트
* Projection: 진단 사례 간 거리와 prototype 형성에 유리한 이벤트

완전한 독립 decoder를 만드는 대신 key/value와 MHA weights는 공유한다.

---

## 3. Loss 개선

### 3.1 최종 Loss

$$
L = \lambda_f L_{\text{fine}} + \lambda_b L_{\text{binary}} + \lambda_s L_{\text{supcon}} + \lambda_p L_{\text{prototype}} + \lambda_c L_{\text{consistency}}
$$


각 loss는 `λ=0`을 허용해 Sweep에서 “적용하지 않음”을 선택할 수 있게 한다.

---

### 3.2 Fine loss

기존 10-class CE를 유지한다.

$$
L_{\text{fine}} = CE(y_{\text{fine}}, z_{\text{fine}})
$$


초기에는 class weight와 label smoothing을 사용하지 않는다. 필요할 경우 3차 Sweep 이후 별도 실험으로 분리한다.

---

### 3.3 Binary loss

$$
L_{\text{binary}} = BCEWithLogits \left( y_{\text{abnormal}}, z_{\text{binary}}, pos_weight \right)
$$


### `pos_weight` 변경

기존 `pos_weight_scale` 방식은 사용하지 않는다.

CLI:

```text
--pos-weight auto
--pos-weight 0.25
--pos-weight 0.5
--pos-weight 1.0
--pos-weight 2.0
```

기본값:

```text
pos_weight = auto
```

`auto`일 경우:

$$
pos_weight = \frac{N_{\text{normal}}}{N_{\text{abnormal}}}
$$


각 CV train split에서 자동 계산한다.

Sweep에서는 다음을 비교한다.

```text
pos_weight ∈ {
  auto,
  0.25,
  0.5,
  1.0,
  2.0
}
```

---

### 3.4 Fine–Binary Consistency loss

Fine head에서 유도한 비정상 확률:

$$
p_{\text{abnormal}}^{fine} = 1-p_{\text{normal class}}^{fine}
$$


Binary head 확률:

$$
p_{\text{abnormal}}^{bin} = \sigma(z_{\text{binary}})
$$


두 출력을 정렬한다.

$$
L_{\text{consistency}} = JS \left( p_{\text{abnormal}}^{fine}, p_{\text{abnormal}}^{bin} \right)
$$


또는 안정적인 첫 구현에서는 stop-gradient BCE를 사용한다.

$$
L_{\text{consistency}} = BCE \left( p_{\text{abnormal}}^{bin}, \operatorname{stopgrad} \left( p_{\text{abnormal}}^{fine} \right) \right)
$$


Sweep:

```text
lambda_consistency ∈ {
  0,
  0.05,
  0.1,
  0.2
}
```

`0`은 consistency를 적용하지 않는 대조군이다.

---

### 3.5 SupCon + Prototype/Center loss

SupCon은 유지하되 유효 positive pair가 부족한 문제를 보완하기 위해 prototype loss를 함께 구현한다.

### SupCon

$$
L_{\text{supcon}} = -\sum_i \log \frac{ \sum_{p \in P(i)} \exp(z_i^\top z_p/\tau) }{ \sum_{a \ne i} \exp(z_i^\top z_a/\tau) }
$$


Sweep:

```text
lambda_supcon ∈ {0, 0.05, 0.1}
temperature ∈ {0.05, 0.1, 0.2}
```

### Prototype/Center loss

$$
L_{\text{prototype}} = \frac{1}{N} \sum_i \left| z_i-\mu_{y_i} \right|^2
$$


Sweep:

```text
lambda_prototype ∈ {0, 0.05, 0.1}
```

### 주의

Binary 라벨을 기준으로 모든 비정상 사례를 하나로 모으지 않는다.

Prototype과 SupCon의 positive label은 기본적으로 Fine diagnosis label을 사용한다.

```text
염분·응애·탄저·동해 등 모든 비정상
→ 하나의 비정상 cluster로 collapse시키지 않음
```

---

## 4. Binary 분석 로그

Binary 결과 해석에 필요한 로그를 train, fold validation, OOF aggregation 단계에 모두 추가한다.

### 4.1 기본 성능 로그

각 fold 및 OOF에서 다음을 기록한다.

```text
binary_loss
binary_pos_weight_effective
binary_logit_mean/std
binary_probability_mean/std
```

지표:

* AUROC
* abnormal AUPRC
* normal-as-positive AUPRC
* balanced accuracy
* accuracy
* abnormal recall
* normal recall
* abnormal precision
* normal precision
* specificity
* sensitivity
* F1 abnormal
* F1 normal
* MCC
* Brier score
* ECE
* NLL

---

### 4.2 Threshold별 로그

기본 threshold:

```text
0.5
```

튜닝 후보:

```text
threshold ∈ {0.25, 0.50, 0.75}
```

각 threshold에서 다음을 저장한다.

```text
TP / FP / TN / FN
abnormal recall
normal recall
balanced accuracy
precision
NPV
F1
coverage
```

최종 threshold는 OOF 또는 repeated CV 결과로 결정한 뒤 freeze한다.

---

### 4.3 Score distribution

다음 분포를 별도로 기록한다.

* 실제 정상의 `p_abnormal`
* 실제 비정상의 `p_abnormal`
* Fine 정답 사례의 `p_abnormal`
* Fine 오답 사례의 `p_abnormal`
* Known 후보의 `p_abnormal`
* Reject 후보의 `p_abnormal`

W&B histogram:

```text
binary/prob_normal_cases
binary/prob_abnormal_cases
binary/logit_normal_cases
binary/logit_abnormal_cases
```

---

## 5. Fine–Binary 불일치 로그

### 5.1 기본 비교

Fine-derived abnormal probability:

$$
p_{\text{abnormal}}^{fine} = 1-p_{\text{fine-normal}}
$$


Binary probability:

$$
p_{\text{abnormal}}^{bin} = \sigma(z_{\text{binary}})
$$


다음을 기록한다.

```text
pearson_corr
spearman_corr
mean_absolute_gap
max_gap
```

---

### 5.2 불일치 유형

| Fine | Binary | 유형                   |
| ---- | ------ | -------------------- |
| 정상   | 정상     | agreement-normal     |
| 비정상  | 비정상    | agreement-abnormal   |
| 정상   | 비정상    | binary-only abnormal |
| 비정상  | 정상     | fine-only abnormal   |

각 fold 및 OOF에서 다음을 기록한다.

```text
head_agreement_rate
head_contradiction_rate
fine_normal_binary_abnormal_count
fine_abnormal_binary_normal_count
```

케이스 단위 CSV:

```text
case_id
true_label
fine_pred
fine_normal_probability
binary_probability
binary_pred
probability_gap
contradiction_type
fold
```

---

### 5.3 Fine–Binary 결합 점수

추론 단계에서 다음 score fusion을 평가한다.

$$
R(x) = \alpha p_{\text{abnormal}}^{bin} + (1-\alpha) p_{\text{abnormal}}^{fine}
$$


후보:

```text
alpha ∈ {
  0,
  0.25,
  0.5,
  0.75,
  1.0
}
```

`α=0`과 `α=1`은 각 head 단독 사용에 해당한다.

---

## 6. SupCon 및 Prototype 로그

### 6.1 SupCon 유효성 로그

매 epoch 또는 일정 step마다 기록한다.

```text
valid_supcon_anchor_count
valid_supcon_anchor_ratio
positive_pair_count
mean_positive_pairs_per_anchor
batch_without_positive_ratio
supcon_loss_zero_ratio
class별 positive_pair_count
```

유효 anchor 비율이 지나치게 낮으면 SupCon을 제거하거나 prototype loss 중심으로 전환한다.

---

### 6.2 Prototype 로그

fold별·class별로 기록한다.

```text
prototype_norm
intra_class_distance_mean/std
inter_class_distance_mean/std
nearest_other_class_distance
intra_inter_ratio
prototype_classification_accuracy
```

retrieval:

* Precision@1
* Precision@3
* cross-farm Precision@k
* same-diagnosis cross-farm retrieval rate

---

## 7. CV 설계 개선

### 7.1 하이퍼파라미터 선택

튜닝과 모델 비교에는 다음을 사용한다.

```text
Repeated group-stratified 3-fold CV
```

기본:

```text
n_folds = 3
n_repeats = 3
seeds = [2023, 2024, 2025]
group = farm_id
```

총 9개의 OOF prediction을 각 config에 생성한다.

### split 생성 목적

다음을 동시에 최소화한다.

$$
J_{\text{split}} = \lambda_1D_{\text{class}} + \lambda_2D_{\text{normal}} + \lambda_3D_{\text{size}}
$$


* fold별 Fine class 분포 차이
* fold별 정상 사례 수 차이
* fold 크기 차이

같은 farm 또는 case의 window가 train과 validation에 동시에 들어가지 않도록 한다.

---

### 7.2 최종 추론 모델

하이퍼파라미터가 확정된 뒤:

```text
farm-stratified 5-fold
```

로 최종 모델 5개를 학습한다.

```text
튜닝·설계 선택:
Repeated 3-fold

최종 unseen 추론:
5-fold soft ensemble
```

5-fold 모델의 output은 다음을 ensemble한다.

* Binary probability
* Fine probability
* energy
* fold별 prototype distance
* Known/Unknown score
* saliency agreement

---

## 8. Early stopping

Early stopping monitor는 다음으로 고정한다.

```text
monitor = val_loss
```

기본값:

```text
max_epochs = 150
min_epochs = 30
patience = 20
mode = min
```

Fold macro-F1은 저장하지만 checkpoint selection과 early stopping에 사용하지 않는다.

각 run 종료 후 모델 선택은 repeated OOF 집계 지표로 수행한다.

---

## 9. Reject 및 Open-set 라우팅

### 9.1 세 가지 출력 상태

### Known diagnosis

기존 10개 class 중 하나로 충분히 설명 가능한 경우

### UNKNOWN_ABNORMAL

비정상 가능성은 높지만 기존 class 공간과 거리가 먼 경우

### REJECT / REVIEW

모델 confidence가 낮거나 head·fold 간 판단이 불안정해 판정을 보류해야 하는 경우

`UNKNOWN_ABNORMAL`과 `REJECT`를 구분한다.

```text
UNKNOWN:
비정상은 확실하지만 기존 진단으로 설명하기 어려움

REJECT:
정상/비정상 또는 진단 자체의 확신이 부족함
```

---

### 9.2 정상 판정

$$
p_{\text{abnormal}} < \tau_{\text{binary}}
$$


이고 다음 조건을 만족할 때 정상 후보로 판정한다.

* Fine normal probability가 최소 기준 이상
* Fine–Binary contradiction 없음
* fold disagreement가 낮음

---

### 9.3 Known 판정

다음을 모두 만족할 때만 Known으로 판정한다.

```python
known = (
    p_abnormal >= tau_binary
    and max_fine_probability >= tau_fine
    and normalized_prototype_distance <= tau_proto
    and energy <= tau_energy
    and fold_agreement >= min_fold_agreement
)
```

---

### 9.4 Unknown 판정

```python
unknown_abnormal = (
    p_abnormal >= tau_binary
    and not known
    and confidence_is_sufficient
)
```

비정상은 확실하지만 기존 진단 confidence, prototype, energy 조건을 통과하지 못하면 열린 진단으로 넘긴다.

---

### 9.5 Reject 판정

다음 중 하나가 발생하면 REJECT로 보낸다.

* Binary 확률이 threshold 주변에 위치
* Fine max probability가 낮음
* Binary와 Fine이 크게 모순
* fold별 정상·비정상 판정이 크게 불일치
* Projection distance와 energy 판단이 충돌
* 이미지와 시계열 근거가 상충
* 모든 검색 후보의 유사도가 낮음

---

### 9.6 Reject/Open-set Sweep

적용하지 않는 선택지를 포함한다.

```text
use_reject ∈ {false, true}
use_open_set ∈ {false, true}
```

후보 threshold:

```text
tau_binary ∈ {0.25, 0.50, 0.75}
tau_fine ∈ {0.0, 0.50, 0.75}
min_fold_agreement ∈ {3, 4, 5}
prototype_percentile ∈ {90, 95, 99}
```

`tau_fine=0.0`은 Fine confidence reject를 적용하지 않는 대조군이다.

---

### 9.7 평가

* coverage
* retained accuracy
* selective risk
* risk–coverage curve
* AURC
* error rejection rate
* correct rejection rate
* Known→Unknown 비율
* 오답 중 Reject/Unknown으로 빠진 비율
* 정상 사례 reject 비율

---

## 10. Projection/Open-set 구현의 fold 주의사항

### 10.1 학습·CV 기본값

학습 및 CV에서는 다음 보호 규칙을 기본으로 활성화한다.

```text
fold_local_prototype = true
oof_only_calibration = true
merge_fold_embeddings = false
crossfit_disagreement = true
```

### 의미

* prototype은 해당 fold의 train 사례만으로 계산
* validation 사례는 prototype 계산에 포함하지 않음
* 서로 다른 fold의 `z_proj` vector를 직접 평균하거나 합치지 않음
* distance와 percentile score만 fold 간 집계
* open-set threshold는 OOF 또는 LODO 결과로만 보정

---

### 10.2 최종 추론 기본값

최종 5-fold inference에서는 다음과 같이 적용한다.

```text
oof_only_calibration = false
crossfit_disagreement = false
```

단, 각 fold의 Projection 좌표계를 직접 합치지 않는 원칙은 유지한다.

```text
fold 0 prototype distance
fold 1 prototype distance
...
→ 각 fold 내부에서 percentile 또는 z-score 변환
→ 정규화된 score만 ensemble
```

즉, 추론에서 open-set을 끄는 것이 아니라 **학습 누수 방지를 위한 OOF 제한만 해제**한다.

---

## 11. 이미지 처리

이미지는 병명 판별에 핵심적인 입력이므로 모든 학습과 추론에서 사용한다.

고정값:

```text
use_image_adapter = true
```

다음은 Sweep에서 제외한다.

```text
use_image_adapter true/false
gate_init
DINO on/off
```

`Stage A cache DINO 포함`의 정확한 의미는:

```text
DINO vector가 Stage A cache에 별도로 저장됨
```

이며, `event_mean/event_max`에 이미 fuse된 것은 아니다.

따라서 Stage B에서:

```text
dino_vec
→ SharedImageAdapter
→ IMAGE event residual
```

경로를 항상 사용한다.

### 필수 로그

튜닝 대상은 아니지만 정상 동작 확인을 위해 다음을 기록한다.

* 이미지 보유 사례 수
* fold별 이미지 사례 수
* `dino_mask` 활성 비율
* adapter gate mean/std
* IMAGE event attention
* DINO residual norm
* 이미지 누락 사례의 zero contribution 확인
* DINO ablation 위험도 차이 — 진단용 분석만 수행

---

## 12. 3단계 Sweep 설계

세 Sweep의 모든 run은 동일한 W&B Group을 사용한다.

```text
group = finetune_v03_improvement_<date_or_revision>
```

예:

```text
group = finetune_v03_improvement_r1
```

Sweep별 구분:

```text
job_type = sweep_1_loss_binary
job_type = sweep_2_metric_openset
job_type = sweep_3_architecture
```

공통 tags:

```text
v03
multitask
repeated3fold
open-set
attention-pooling
```

W&B에서는 `group=finetune_v03_improvement_r1`로 필터링해 세 Sweep 결과를 한 화면에서 확인한다.

---

### 12.1 Sweep 1 — Binary 및 Loss 정합

목적:

* Binary imbalance 조정
* Fine–Binary consistency
* 기존 Fine 성능 유지
* threshold operating point 선정

고정:

```text
lambda_supcon = 0
lambda_prototype = 0
task_specific_query = false
token_attention_pool = false
```

후보:

```text
lr ∈ {1e-4, 3e-4}
lambda_binary ∈ {0.5, 1.0, 2.0}
pos_weight ∈ {auto, 0.25, 0.5, 1.0}
lambda_consistency ∈ {0, 0.05, 0.1}
binary_threshold ∈ {0.25, 0.50, 0.75}
```

`binary_threshold`는 재학습 parameter가 아니라 동일 OOF prediction을 대상으로 평가 단계에서 계산한다.

선정 지표:

1. Fine macro-F1
2. Binary AUROC
3. threshold별 balanced accuracy
4. Fine–Binary contradiction rate
5. Brier score

---

### 12.2 Sweep 2 — Metric learning 및 Open-set

Sweep 1에서 선정한 loss·optimizer 설정을 고정한다.

후보:

```text
lambda_supcon ∈ {0, 0.05, 0.1}
lambda_prototype ∈ {0, 0.05, 0.1}
supcon_temperature ∈ {0.05, 0.1, 0.2}

use_reject ∈ {false, true}
use_open_set ∈ {false, true}
tau_fine ∈ {0.0, 0.5, 0.75}
prototype_percentile ∈ {90, 95, 99}
min_fold_agreement ∈ {3, 4, 5}
```

선정 지표:

* Fine macro-F1
* prototype classification accuracy
* cross-farm Precision@k
* LODO open-set AUROC
* AURC
* coverage 80% 기준 retained accuracy
* 정상 사례 reject rate
* 오류 포착률

---

### 12.3 Sweep 3 — Architecture

Sweep 1과 2의 최적 loss·open-set 설정을 고정한다.

후보:

```text
task_specific_query ∈ {false, true}
token_attention_pool ∈ {false, true}
attention_residual ∈ {false, true}
dropout ∈ {0.1, 0.2, 0.3}
weight_decay ∈ {1e-3, 1e-2}
```

제약:

```text
token_attention_pool=false
→ attention_residual=false 고정
```

이미지 adapter는 항상 활성화한다.

선정 지표:

* 반복 CV Fine macro-F1 평균
* 반복 CV Binary AUROC 평균
* 표준편차
* Fine–Binary contradiction rate
* open-set AURC
* token attention deletion fidelity
* 학습시간 및 cache 비용

---

## 13. Run 저장 규칙

각 run은 다음 정보를 반드시 저장한다.

```text
group_id
sweep_stage
config_hash
split_manifest
repeat_id
fold_id
train_case_ids
val_case_ids
best_epoch
best_val_loss
OOF predictions
OOF embeddings
fold-local prototypes
threshold table
reject/open-set result
```

파일 예:

```text
run_dir/
├── config.json
├── split_manifest.json
├── fold_metrics.json
├── oof_predictions.csv
├── oof_binary_thresholds.csv
├── oof_head_consistency.csv
├── oof_embeddings.parquet
├── fold_prototypes/
├── open_set_metrics.json
├── selective_risk.csv
└── report.md
```

---

## 14. 최종 모델 선택 기준

단일 최고 점수가 아니라 repeated CV 평균과 안정성을 기준으로 선택한다.

### 최소 회귀 방지 기준

현재 baseline:

```text
Fine OOF macro-F1 = 0.776
Binary AUROC = 0.911
```

후보 모델은 다음을 만족해야 한다.

```text
Fine macro-F1 평균이 baseline 대비 -0.02 이내
Binary AUROC 평균이 baseline 이상
Fine–Binary contradiction rate 감소
Repeated split 표준편차 감소
```

### Open-set 기준

* Reject 사용 시 retained accuracy 증가
* 오류가 정답보다 높은 비율로 reject됨
* coverage가 지나치게 낮지 않음
* LODO pseudo-unknown 탐지 성능 개선
* 정상 사례가 과도하게 unknown 또는 reject되지 않음

---

## 15. 최종 5-fold 학습 및 추론

3단계 Sweep이 끝나면 최종 config를 고정한다.

```text
최종 학습
→ farm-stratified 5-fold
→ checkpoint 5개
```

추론:

```text
각 fold 모델
  ├─ Binary probability
  ├─ Fine probability
  ├─ energy
  ├─ fold-local prototype distance
  ├─ DINO/Image contribution
  └─ saliency
       ↓
score normalization
       ↓
5-fold ensemble
       ↓
Normal / Known / Unknown / Reject
```

---

## 16. 열린 진단

`UNKNOWN_ABNORMAL`로 라우팅된 경우 기존 Fine class를 최종 진단명으로 사용하지 않는다.

입력 근거:

* Binary abnormal score
* Fine top-k 후보
* fold agreement
* prototype distance
* energy
* event saliency
* token signed attribution
* raw value
* 정상 대응 구간
* DINO 이미지 유사 사례
* 기존 35건 유사 시퀀스
* 반증 근거

출력:

```text
판정: 비정상
기존 10개 진단 범위: 낮은 적합도
1차 열린 진단 후보
감별 후보
기존 예시와의 유사점
기존 예시와의 차이
필요한 현장 확인
확신도
판정 보류 가능성
```

---

## 17. Counterfactual 의사결정 연계

Binary와 Fine-derived abnormal score를 이용해 연속 위험도를 구성한다.

$$
R(x) = \alpha p_{\text{abnormal}}^{bin} + (1-\alpha) \left( 1-p_{\text{normal}}^{fine} \right)
$$


행동:

```text
현재 시퀀스 x
→ 이벤트 또는 measurement group 편집 a
→ 편집 시퀀스 x'
```

보상:

$$
Reward = R(x)-R(x') -\lambda_e C_{\text{edit}} -\lambda_o C_{\text{OOD}} -\lambda_u C_{\text{uncertainty}} -\lambda_m C_{\text{MLM}}
$$


개입 시점은 비정상 위험에 양의 영향을 주는 이벤트 saliency로 탐색한다.

Attention은 다음 용도로 사용한다.

```text
Attention:
모델이 집중한 이벤트·token 후보

Signed IxG:
비정상 위험을 높이거나 낮춘 방향

Perturbation:
실제 위험도 변화 검증
```

최종 개입안은 최소 편집 집합으로 축소한다.

---

## 18. 구현 우선순위

### P0 — 평가 및 로그

* Binary 전체 로그
* Fine–Binary 불일치 로그
* SupCon positive pair 로그
* Prototype 거리 로그
* threshold table
* risk–coverage curve
* repeated 3-fold split

### P1 — Loss

* absolute `pos_weight`
* Consistency loss
* Prototype/center loss
* loss weight `0` 허용

### P2 — Open-set

* Reject
* fold-local prototype
* energy
* LODO calibration
* selective prediction

### P3 — 구조 변경

* task-specific query
* Attention token→event pooling
* mean/max + attention residual

### P4 — 최종 5-fold

* best config 재학습
* 5-fold unseen ensemble
* open-set inference

### P5 — 열린 진단·Counterfactual

* raw/token evidence
* retrieval
* 관리 개입 탐색

---

## 19. 최종 기본값

```yaml
cv:
  tuning_folds: 3
  repeats: 3
  final_inference_folds: 5
  group_key: farm_id

training:
  monitor: val_loss
  pos_weight: auto
  binary_threshold: 0.5

loss:
  lambda_fine: 1.0
  lambda_binary: 1.0
  lambda_consistency: 0.05
  lambda_supcon: 0.0
  lambda_prototype: 0.05

architecture:
  task_specific_query: true
  token_attention_pool: true
  attention_residual: true
  use_image_adapter: true

open_set_training:
  fold_local_prototype: true
  oof_only_calibration: true
  merge_fold_embeddings: false
  crossfit_disagreement: true

open_set_inference:
  oof_only_calibration: false
  crossfit_disagreement: false
  merge_fold_embeddings: false

routing:
  use_reject: true
  use_open_set: true
  tau_binary: 0.5
  tau_fine: 0.5
  prototype_percentile: 95
  min_fold_agreement: 4
```

이 기본값은 Sweep 결과에 따라 변경하되, 이미지 사용과 fold별 Projection 공간 분리는 고정한다.