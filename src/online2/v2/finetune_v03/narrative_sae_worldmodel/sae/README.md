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

## ⚠ 실제 pilot 결과 — 재보정 완료, 근본 원인 특정, §18 중단 조건은 여전히 발동

이번 세션에서 실제로 학습된 SAE를 여러 설정으로 돌렸다(`scripts/online2_v2/pilot_sae_stage_a_activations.py`). 입력은 새로 계산한 게 아니라 `../representation_tracking/`가 감싸는 것과 같은 소스 — `cache_stage_a_event_embeddings.py`가 이미 캐시해 둔 실제 Stage-A pooled activation(`event_mean`, 384차원, `outputs/online2/v2_finetune_v02/event_embeddings/`, 전체 216,040개 벡터).

### 왜 이런 일이 생기는지 — 근본 원인을 실측으로 특정함

"시퀀스를 만들 때 쓰는 서사(narrative) 카탈로그가 작아서 그런가?"라는 질문을 계기로 인코더 학습 계보를 다시 추적했다. **결론: 부분적으로만 맞다 — 정확한 병목은 다른 데 있었다.**

- Stage-A 인코더(이 SAE가 분해하는 activation을 만든 모델)는 §5 `narrative_grounding`이 쓰는 80개 서사 템플릿 코퍼스(v8 build, `sequences.parquet`)로 학습된 게 **아니다**. `outputs/online2/v2_build/build_manifest_v2.json`과 `run_manifest_v2.json`을 직접 확인한 결과, 별도의 훨씬 큰 코퍼스(`v2_build`, 원시 관측 264만 건, vocab 728개, rule 1196개, `training_mode: transductive_public_pretraining`)로 18,600 step 학습됐다. 인코더 자체의 학습 다양성은 narrative 카탈로그 크기에 갇혀 있지 않다.
- 진짜 병목은 **SAE pilot이 어떤 activation을 봤는가**다. `cache_stage_a_event_embeddings.py`는 인코더가 배울 수 있는 전체 분포가 아니라, CF1S 진단용 55개 case(Dev3 3 + Primary32 32 + Validation20 20)의 activation만 캐시해 둔 것이다 — 이 세션 내내 반복된 "27MB·55케이스 규모" 리스크가 SAE에도 그대로 나타난 것이다.
- **직접 측정**: 55개 case를 섞은 실제 샘플(16,500개, case당 300개)에 PCA를 돌리면 **384차원 중 단 29개 주성분이 분산의 99%를 설명한다**(50%는 1개, 90%는 7개, 95%는 13개). 한 case 안에서만 봐도 연속된 이벤트의 코사인 유사도가 평균 0.89(무작위 쌍도 0.63)로, 센서값이 천천히 변해 "이벤트"라 부르는 것들이 사실상 서로 거의 같은 값이다. 즉 384차원짜리 벡터를 쓰고 있지만 이 55-case 표본의 **실제 유효 차원은 약 29**다 — dictionary를 768~3072개로 만든 게 데이터가 가진 변화량보다 25~100배 큰 것이었다.

### 그래도 dead ratio는 다 안 없어졌다 — 별도의 학습 역학 문제

유효 차원(~29)에 맞춰 dict_size를 직접 낮춰 봤다(`--dict-size` 옵션 추가):

| 설정 | dict_size | sparsity | mean_l0 | explained_variance | dead_feature_ratio |
|---|---|---|---|---|---|
| 원래 pilot(샘플 15K) | 3072 (8x) | Top-K(k=16) | 16.0 | 0.950 | 0.928 |
| 전체 데이터, 동일 dict | 3072 (8x) | Top-K(k=16) | 16.0 | 0.965 | 0.742 |
| 전체 데이터, dict 축소 | 1536 (4x) | Top-K(k=16) | 16.0 | 0.970 | 0.641 |
| 전체 데이터, dict 더 축소 | 768 (2x) | Top-K(k=16) | 16.0 | 0.965 | 0.577 |
| 전체 데이터, L1 모드 | 768 (2x) | L1 | 161.1 | 0.950 | 0.405 |
| 전체 데이터, 1x(입력과 동일) | 384 | Top-K(k=8) | 8.0 | 0.952 | 0.536 |
| 전체 데이터, 유효차원 근접 | 64 | Top-K(k=8) | 8.0 | 0.915 | 0.422 |
| 전체 데이터, 유효차원 근접 | 32 | Top-K(k=4) | 4.0 | 0.913 | **0.406**(dict 32에서) |

dict_size를 PCA 유효 차원(29)에 거의 맞춘 32~64에서도 dead ratio가 **0.40~0.42 선에서 더 안 내려간다** — L1+dict768 조합(0.405)과 거의 같은 바닥이다. `mean_activation_frequency`(예: dict=32에서 0.125 = 4/32)는 평균적으로 각 feature가 21,604개 검증 샘플 중 12.5%는 활성화될 만큼 충분한데도 40%가 정확히 0번 활성화됐다는 건, 사용량이 **극단적으로 쏠려 있다**는 뜻이다 — 소수의 "만능" feature가 Top-K 선택을 거의 독점하고 나머지는 구조적으로 밀려난다(TopK SAE의 전형적인 "승자독식" 현상으로 보이며, 매 epoch 사후에 완전히 죽은 feature만 재초기화하는 지금의 resample 정책으로는 이 쏠림 자체를 못 막는다).

**결론**: dead feature 문제는 두 겹이다 — (1) 55-case 표본의 낮은 유효 차원(~29, **원인 특정 완료**), (2) dict_size를 그 차원에 맞춰도 남는 TopK 학습 역학 문제(**미해결**). (1)은 이번에 명확히 설명됐지만 (2)는 이번 세션에서 풀지 못했다.

**따라서 §18 "SAE feature 대부분이 dead 또는 불안정" 중단 조건은 재보정 후에도 여전히 발동한 상태다** — 다음 Phase(concept_governance, world_model)로 자동 진행하지 않는다. 이건 실패 은폐가 아니라 §18이 명시한 대로 readiness 결과로 기록하는 것이다. `pilot_sae_stage_a_activations.py`의 기본값은 이번 실측을 반영해 `--max-rows 220000`(전체 데이터), `--dict-expansion-factor 4`로 뒀다 — 그래도 dead ratio 자체는 여전히 기준을 넘는다는 걸 기본 실행 결과로 바로 보게 했다. `--dict-size`(신규 옵션)로 expansion factor 배수가 아닌 임의의 dict_size를 직접 지정할 수 있다.

**다음으로 시도해볼 것(아직 안 해본 것, 원인 (2) "TopK 승자독식"을 겨냥)**: (1) 더 공격적인 resampling(지금은 epoch당 1회, threshold=0.0 — 배치 단위로 더 자주 하거나 threshold를 0보다 높여서 "거의 안 쓰이는" feature까지 선제적으로 재초기화), (2) auxiliary loss로 저사용 feature에 보너스를 주는 방식(예: OpenAI TopK SAE 논문의 "AuxK" 손실 — 죽은/저사용 feature가 reconstruction residual을 추가로 설명하도록 강제), (3) 55-case 대신 `v2_build`의 넓은 transductive pretraining 모집단에서 직접 activation을 새로 뽑아 표본 다양성 자체를 키우기(지금은 그런 캐시가 없어 frozen encoder를 새로 돌려야 함 — 원인 (1)에 대한 근본 해결), (4) 초기화 방식(직교 초기화 등)을 바꿔 초반 승자독식을 완화. 지금 갖고 있는 도구(`model.py`의 `sparsity_mode`/`top_k`, `training.py`의 `resample_dead_features`, `pilot_sae_stage_a_activations.py`의 `--dict-size`)로 (1)(2)(4)는 바로 시도 가능하다 — 시간 제약으로 이번 세션에서는 여기까지만 했다.

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

§18 "SAE feature 대부분이 dead 또는 불안정" — **실측으로 발동, 전체 데이터 재보정(216,040개 벡터, 8개 설정 비교) 이후에도 해소 안 됨**(위 pilot 결과 표 참조). 근본 원인 두 가지를 특정했다: (1) 55-case 표본의 낮은 유효 차원(PCA로 384차원 중 29개가 99% 분산 설명 — **원인 특정 완료**), (2) dict_size를 그 차원(29)에 맞춰도 남는 TopK "승자독식" 학습 역학(dict=32~64에서도 dead ratio가 0.40~0.42 바닥 — **미해결**). concept_governance/world_model 등 이 SAE의 feature를 입력으로 쓰는 다음 단계로 넘어가기 전에, 위 "다음으로 시도해볼 것" 목록(AuxK loss, 더 공격적인 resampling, 또는 55-case를 넘어선 넓은 표본 확보)을 마저 시도할 것.
