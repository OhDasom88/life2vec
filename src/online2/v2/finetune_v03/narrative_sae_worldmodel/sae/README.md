# sae

**근거**: 계획서 §9 (SAE 기반 기계적 해석)
**Phase**: Phase 4 · **범위**: 현재 필수 (§3.1, "선택 layer 대상 SAE pilot"만) · **상태**: 핵심 구현 완료(상태 머신·모델·평가·dead feature resampling) + 실제 캐시 activation 전체(216,040개)로 5가지 설정 재보정 실행, 34개 테스트 통과. **재보정 후에도 §18 중단 조건이 계속 발동 — 아래 참조**

## 목적

사전학습/미세조정 encoder의 dense activation에서 희소 특징을 추출하고, 의미적 순도·기능적 개입 효과가 검증된 특징만 해석 단위로 승격한다. "SAE latent = monosemantic feature"를 자동으로 확정하지 않는다.

## 상태 머신 (§9.1)

```
SAE_LATENT → CANDIDATE_SEMANTIC_FEATURE → EVALUATED_SEMANTIC_FEATURE
           → INTERVENTION_SUPPORTED_FEATURE → CIRCUIT_SUPPORTED_FEATURE
```

CF1S의 `classify_case_disposition()`과 동일한 설계 원칙을 따른다: 상태는 저장된 라벨이 아니라 매번 재계산한다(자기선언 금지).

## ⚠ 실제 pilot 결과 — 전체 데이터 재보정 완료, §18 중단 조건은 여전히 발동

이번 세션에서 실제로 학습된 SAE를 여러 설정으로 돌렸다(`scripts/online2_v2/pilot_sae_stage_a_activations.py`). 입력은 새로 계산한 게 아니라 `../representation_tracking/`가 감싸는 것과 같은 소스 — `cache_stage_a_event_embeddings.py`가 이미 캐시해 둔 실제 Stage-A pooled activation(`event_mean`, 384차원, `outputs/online2/v2_finetune_v02/event_embeddings/`, 전체 216,040개 벡터).

첫 pilot(샘플 14,960개, dict_size=3072)에서 `dead_feature_ratio=0.928`이 나와 §18 중단 조건에 걸린 뒤, "전체 데이터로 재보정" 요청에 따라 데이터 규모와 dictionary 크기·sparsity 모드를 바꿔가며 4가지 설정을 **실제로 전부 돌려봤다**(전체 216,040개 벡터, 15 epoch, held-out 21,604개 기준):

| 설정 | dict_size | sparsity | mean_l0 | explained_variance | dead_feature_ratio |
|---|---|---|---|---|---|
| 원래 pilot(샘플 15K) | 3072 (8x) | Top-K(k=16) | 16.0 | 0.950 | **0.928** |
| 전체 데이터, 동일 dict | 3072 (8x) | Top-K(k=16) | 16.0 | 0.965 | 0.742 |
| 전체 데이터, dict 축소 | 1536 (4x) | Top-K(k=16) | 16.0 | **0.970** | 0.641 |
| 전체 데이터, dict 더 축소 | 768 (2x) | Top-K(k=16) | 16.0 | 0.965 | 0.577 |
| 전체 데이터, L1 모드 | 768 (2x) | L1 | 161.1 | 0.950 | **0.405**(최저) |

**결론**: 전체 데이터 사용은 확실히 도움이 됐다(0.928 → 0.742, 같은 dict_size에서). dict_size를 줄이면 더 개선된다(0.742 → 0.577). L1 모드가 dead ratio는 가장 낮지만(0.405) 그 대가로 sparsity 자체가 무너진다(mean_l0=161/768 ≈ 21% 활성 — SAE의 핵심 목적인 "희소함"이 사실상 없어짐). **어떤 설정도 "낮은 dead ratio"와 "진짜 희소함"을 동시에 만족하지 못했다** — 이건 dictionary 크기나 학습 데이터양을 조정하는 것만으로는 안 풀리는, 이 activation 공간(384차원 Stage-A pooled output, 55농장 규모) 자체의 독립적인 변화 방향이 수백 개의 개별 feature를 정당화할 만큼 많지 않다는 신호로 보인다.

**따라서 §18 "SAE feature 대부분이 dead 또는 불안정" 중단 조건은 재보정 후에도 여전히 발동한 상태다** — 다음 Phase(concept_governance, world_model)로 자동 진행하지 않는다. 이건 실패 은폐가 아니라 §18이 명시한 대로 readiness 결과로 기록하는 것이다. `pilot_sae_stage_a_activations.py`의 기본값은 이번 실측을 반영해 `--max-rows 220000`(전체 데이터), `--dict-expansion-factor 4`(실측 중 explained_variance가 가장 높았던 설정)로 바꿔뒀다 — 그래도 dead ratio 자체는 여전히 기준을 넘는다는 걸 기본 실행 결과로 바로 보게 했다.

**다음으로 시도해볼 것(아직 안 해본 것)**: (1) dict_size를 2x보다 더 줄이기(예: 1x=384, 입력 차원과 동일 — "확장"이 아예 없는 지점까지), (2) resampling을 epoch당이 아니라 더 자주/드물게 하는 스케줄 비교, (3) 여러 layer/pooling(event_max, 또는 `../representation_explorer/`의 3D projection으로 먼저 활성화 공간의 실제 유효 차원을 추정) 확인, (4) L1 계수를 sparsity와 dead ratio 사이 다른 지점으로 sweep. 지금 갖고 있는 도구(`model.py`의 `sparsity_mode`/`top_k`, `pilot_sae_stage_a_activations.py`의 CLI 인자)로 전부 시도 가능하다 — 시간 제약으로 이번 세션에서는 여기까지만 했다.

## 구현한 것

- [`schemas.py`](schemas.py) — `FeatureState`(5단계 상태 머신, 한 단계씩만 승격 가능하도록 `assert_valid_promotion`이 강제) + `CausalGrade`(E0~E5) + `assert_causal_claim_allowed`(**E3 미만이면 실행 시점에 예외** — "인과적"이라는 주장을 코드 레벨에서 막는다, 조용한 문서 문구가 아니다) + `SAEFeatureRef`(`INTERVENTION_SUPPORTED_FEATURE` 상태인데 `causal_grade < E3`이면 생성 자체가 막힘).
- [`model.py`](model.py) — `SparseAutoencoder`(표준 SAE 구성: pre-encoder bias 차감, encoder+decoder, L1 또는 Top-K sparsity, tied/untied weight 선택 가능) + `sae_loss`. 실제 torch forward/backward로 검증(합성 데이터에서 30 step 학습 시 reconstruction loss가 실제로 줄어드는 것까지 테스트).
- [`evaluation.py`](evaluation.py) — §9.4 6영역 중 라벨 없이 계산 가능한 두 영역만: `reconstruction_metrics`(MSE, explained variance), `sparsity_metrics`(평균 L0, activation frequency, dead feature ratio). `downstream_fidelity`는 `../representation_tracking/attribution.py`의 modality ablation과 같은 패턴 — 이미 계산된 두 출력(원본 activation 기반 vs SAE 재구성 기반)의 차이만 계산, 실제 downstream 모델 순전파는 호출자 책임.
- [`training.py`](training.py) — `resample_dead_features`: activation frequency가 threshold 이하인 feature의 encoder/decoder 가중치만 골라 재초기화(다른 feature는 안 건드림, 테스트로 확인). 표준 기법 그대로 씀 — "재구성 오차가 큰 방향으로 재초기화"하는 더 정교한 변형은 안 함(아래 "아직 없는 것").
- [`scripts/online2_v2/pilot_sae_stage_a_activations.py`](../../../../../../scripts/online2_v2/pilot_sae_stage_a_activations.py) — 위 결과를 만든 실제 실행 스크립트. 재현 가능.

## 아직 없는 것

- **의미(semantic)/분해(decomposition)/안정성(stability)/기능(functional) 4개 평가 영역**: 사람 라벨(`../narrative_grounding/`의 검토 결과)이나 실제 개입 실험(ablation/steering/patching)이 있어야 계산할 수 있다 — 지금은 복원·희소성만 있다. 계획서 §9.4 자체가 "SAE reconstruction과 sparsity가 좋아도 의미적 순도가 확보됐다고 판정하지 않는다"고 못 박아 두므로, 지금 나온 explained_variance=0.95라는 숫자만으로 이 SAE가 "좋다"고 말할 수 없다.
- **BatchTopK sparsity**: §9.3이 언급하는 세 번째 옵션(배치 전체에서 top-k를 뽑는 변형). 지금은 샘플별 Top-K만 구현.
- **4개 SAE version(`SAE_PRETRAIN`/`SAE_FINETUNE`/`SAE_WORLD`/`SAE_POLICY`) 구분 실행**: `SparseAutoencoder`는 버전과 무관한 범용 아키텍처다. 버전별로 어느 activation 소스에 붙일지는 아직 안 정했다 — `SAE_WORLD`/`SAE_POLICY`는 `../world_model/`/`../edit_policy_rl/` 자체가 미착수라 원천적으로 이름.
- **`concept_mapping.py`(SAE feature ↔ ontology N:M)**, **`feature_alignment.py`(사전학습↔미세조정 특징 정렬)**: 위 dead feature 문제부터 해결한 뒤에 의미가 있는 작업이라 미룸.
- **sweep_config.py/wandb 로깅**: `../multimodal_pretrain/`과 마찬가지로 실제 학습 루프에 연결되기 전까지 보류.

## 의존성

- 기존: `../representation_tracking/`(activation 소스, `capture.py`가 감싸는 것과 동일한 캐시 parquet)
- 신규: 없음(activation 캡처는 이미 있음, 재사용)

## Acceptance 연결

D1(복원), D2(dead feature ratio 일부) 계산 가능. D3–D6(개념 매핑, 인과 주장, feature alignment)은 미착수(계획서 §17-D)

## 중단 조건 연결

§18 "SAE feature 대부분이 dead 또는 불안정" — **실측으로 발동, 전체 데이터 재보정(216,040개 벡터, 5개 설정 비교) 이후에도 해소 안 됨**(위 pilot 결과 표 참조). 최선 설정(dict 2x + L1)도 dead ratio 0.405는 확보했지만 그 대가로 sparsity 자체를 잃었다(mean_l0 21%). concept_governance/world_model 등 이 SAE의 feature를 입력으로 쓰는 다음 단계로 넘어가기 전에, 위 "다음으로 시도해볼 것" 목록을 마저 시도하거나, 이 activation 공간의 유효 차원이 SAE pilot에 적합한 규모인지부터 재검토할 것.
