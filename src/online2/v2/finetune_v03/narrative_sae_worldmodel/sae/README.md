# sae

**근거**: 계획서 §9 (SAE 기반 기계적 해석)
**Phase**: Phase 4 · **범위**: 현재 필수 (§3.1, "선택 layer 대상 SAE pilot"만) · **상태**: 핵심 구현 완료(상태 머신·모델·평가·dead feature resampling) + dense/narrative-selected 두 표본 구성으로 총 9가지 설정 실측 비교, 34개 테스트 통과. **원인 진단 완료(데이터의 유효 차원이 근본적으로 낮음, 표본 구성 방식과 무관), §18 중단 조건은 계속 발동 — 아래 참조**

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

"시퀀스를 만들 때 쓰는 서사(narrative) 카탈로그가 작아서 그런가?"라는 질문을 계기로 인코더 학습 계보를 다시 추적했다.

- **정정**: 처음엔 "Stage-A 인코더가 §5 `narrative_grounding`의 80개 서사 템플릿 코퍼스와 무관한 별도 코퍼스로 학습됐다"고 결론냈으나, 이건 `build_manifest_v2.json`의 요약 통계만 보고 낸 성급한 결론이었다 — 실제 학습 데이터(`outputs/online2/v2_build/training_events_v2.parquet`, 1,500만 행)를 직접 열어보니 `narrative_id` 컬럼이 있고 값이 정확히 그 80개 템플릿(A05, D12, D13, C01...)과 같았다. 사전학습 코퍼스도 **같은 80개 규칙으로 선별된 것**이다(추정 시퀀스 수 약 94만 개, v8의 967,012개와 같은 자릿수). 다만 실제 모델 입력 토큰(`BACKGROUND_TOKENS`)에는 `FARM|...`만 들어가고 `NARRATIVE|...`는 안 들어간다 — 모델이 "이건 C01이다"라고 직접 보고 배우는 건 아니지만, 애초에 어떤 원시 구간이 사전학습 데이터로 뽑히는지 자체는 이 80개 규칙이 정한다.
- **더 근본적인 발견**: `events_tokenized_v2.parquet`(모든 코퍼스가 파생되는 원시 이벤트 원천)를 확인한 결과 **55개 농장 전부가 정확히 같은 길이(13.96일)의 창을 가지며, 원시 이벤트는 총 222,309건뿐이다.** 사전학습용 94만 "시퀀스"는 새 원시 데이터가 아니라 이 222,309건을 80개 템플릿으로 평균 16번씩 겹쳐 재윈도잉한 결과다. CF1S 55-case 캐시도 각 case가 자기 농장의 14일 전체를 조밀하게 담아, 사실상 이 222,309건을 이미 거의 다 커버한다 — **"더 넓은 코퍼스에서 새로 뽑기"는 실제로는 다양성을 더해주지 않는다**(직접 시도해서 확인함, 아래 참조).
- **PCA 직접 측정**: 55개 case를 섞은 실제 샘플(16,500개, case당 300개)에서 **384차원 중 단 29개 주성분이 분산의 99%를 설명한다**(50%는 1개, 90%는 7개, 95%는 13개). 한 case 안에서만 봐도 연속된 이벤트의 코사인 유사도가 평균 0.89(무작위 쌍도 0.63)다 — 센서값이 천천히 변해 "이벤트"라 부르는 것들이 사실상 서로 거의 같은 값이다.

### "서사가 선별한 순간만 뽑으면 다양성이 늘지 않을까?" — 직접 검증, 결과는 기각

CF1S의 밀집 캡처(농장의 14일을 이상치든 평온한 구간이든 구분 없이 매시간 전부 포함) 대신, **80개 narrative 템플릿이 실제로 매칭한 "타깃" 이벤트만** 골라 activation을 새로 뽑아 비교했다(`scripts/online2_v2/pilot_sae_narrative_selected_activations.py`, 같은 frozen 인코더·같은 222,309건 원시 풀, 표본 구성 방식만 다름). `training_events_v2.parquet`에서 시퀀스별 마지막(최대 `event_position`) 이벤트를 그 시퀀스의 "타깃"으로 보고, 80개 템플릿당 최대 250개씩 균등 추출했다(80/80 템플릿 커버, 14,544건).

같은 크기(14,544 vs 14,520)로 맞춰 두 표본을 직접 비교하면:

| 표본 | 50% | 80% | 90% | 95% | 99% |
|---|---|---|---|---|---|
| Dense(무작위, CF1S 방식) | 1 | 3 | 7 | 13 | 29 |
| Narrative-selected(80템플릿 타깃) | 1 | 3 | 7 | 12 | 29 |

**유효 차원이 사실상 동일하다.** 같은 설정(dict=64, top_k=8)으로 SAE를 돌려도 dead_feature_ratio가 narrative-selected 쪽이 0.500, dense 쪽이 0.422로 **오히려 narrative 선별이 더 낫지 않았다**(정확히 같은 학습 스텝 수 기준 비교, 차이는 잡음 수준으로 보임).

**결론: 애초 가설("서사 선별이 다양성을 늘려줄 것")은 실측으로 기각된다.** narrative 템플릿으로 "흥미로운 순간"만 골라도, 그냥 매시간 다 뽑아도 근본적으로 같은 ~29차원 구조가 나온다 — 이건 표본 구성 방식의 문제가 아니라, 이 데이터(온도·습도·CO2·EC 등이 서로 강하게 얽혀 움직이는 물리계, 55농장×14일)가 원래 가진 선형 자유도가 그 정도라는 뜻으로 보인다. 사전학습 코퍼스가 80개 규칙으로 선별됐다는 사실(위 "정정" 참조)과 무관하게, 그 선별이 SAE가 보는 activation의 다양성 자체를 늘려주지는 못한다.

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

**시도했지만 효과 없었던 것**: 55-case 대신 더 넓은 표본(narrative-template 선별, 또는 v2_build 전체)에서 activation을 새로 뽑는 것 — 위 "직접 검증" 참조. 효과 없음이 명확히 확인됐으므로 더 이상 이 방향은 시도할 필요 없다.

**다음으로 시도해볼 것(아직 안 해본 것, 원인 (2) "TopK 승자독식"을 겨냥 — 데이터가 아니라 SAE 학습 알고리즘 쪽 문제이므로 이쪽이 남은 유일한 레버다)**: (1) 더 공격적인 resampling(지금은 epoch당 1회, threshold=0.0 — 배치 단위로 더 자주 하거나 threshold를 0보다 높여서 "거의 안 쓰이는" feature까지 선제적으로 재초기화), (2) auxiliary loss로 저사용 feature에 보너스를 주는 방식(예: OpenAI TopK SAE 논문의 "AuxK" 손실 — 죽은/저사용 feature가 reconstruction residual을 추가로 설명하도록 강제), (3) 초기화 방식(직교 초기화 등)을 바꿔 초반 승자독식을 완화, (4) 더 이른/다른 transformer layer의 activation(지금은 최종 pooled output만 썼다 — PCA는 선형 구조만 보므로, pooling 이전의 토큰별 hidden state나 중간 layer에는 비선형적으로 더 많은 구조가 남아있을 가능성이 있다). 지금 갖고 있는 도구(`model.py`의 `sparsity_mode`/`top_k`, `training.py`의 `resample_dead_features`, `pilot_sae_stage_a_activations.py`의 `--dict-size`)로 (1)(2)(3)은 바로 시도 가능하다 — 시간 제약으로 이번 세션에서는 여기까지만 했다.

## 구현한 것

- [`schemas.py`](schemas.py) — `FeatureState`(5단계 상태 머신, 한 단계씩만 승격 가능하도록 `assert_valid_promotion`이 강제) + `CausalGrade`(E0~E5) + `assert_causal_claim_allowed`(**E3 미만이면 실행 시점에 예외** — "인과적"이라는 주장을 코드 레벨에서 막는다, 조용한 문서 문구가 아니다) + `SAEFeatureRef`(`INTERVENTION_SUPPORTED_FEATURE` 상태인데 `causal_grade < E3`이면 생성 자체가 막힘).
- [`model.py`](model.py) — `SparseAutoencoder`(표준 SAE 구성: pre-encoder bias 차감, encoder+decoder, L1 또는 Top-K sparsity, tied/untied weight 선택 가능) + `sae_loss`. 실제 torch forward/backward로 검증(합성 데이터에서 30 step 학습 시 reconstruction loss가 실제로 줄어드는 것까지 테스트).
- [`evaluation.py`](evaluation.py) — §9.4 6영역 중 라벨 없이 계산 가능한 두 영역만: `reconstruction_metrics`(MSE, explained variance), `sparsity_metrics`(평균 L0, activation frequency, dead feature ratio). `downstream_fidelity`는 `../representation_tracking/attribution.py`의 modality ablation과 같은 패턴 — 이미 계산된 두 출력(원본 activation 기반 vs SAE 재구성 기반)의 차이만 계산, 실제 downstream 모델 순전파는 호출자 책임.
- [`training.py`](training.py) — `resample_dead_features`: activation frequency가 threshold 이하인 feature의 encoder/decoder 가중치만 골라 재초기화(다른 feature는 안 건드림, 테스트로 확인). 표준 기법 그대로 씀 — "재구성 오차가 큰 방향으로 재초기화"하는 더 정교한 변형은 안 함(아래 "아직 없는 것").
- [`scripts/online2_v2/pilot_sae_stage_a_activations.py`](../../../../../../scripts/online2_v2/pilot_sae_stage_a_activations.py) — dense(CF1S) 표본 pilot. 재현 가능.
- [`scripts/online2_v2/pilot_sae_narrative_selected_activations.py`](../../../../../../scripts/online2_v2/pilot_sae_narrative_selected_activations.py) — narrative-template 타깃 표본 pilot. `training_events_v2.parquet`(1,500만 행)에서 시퀀스별 타깃 이벤트를 스트리밍 집계(전체를 메모리에 안 올림, 청크 단위로 처리), 80개 템플릿당 균등 샘플링한 뒤 `cache_stage_a_event_embeddings.py`의 인코딩 함수를 그대로 import해 재사용한다(활성화 계산 로직 중복 없음).

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

§18 "SAE feature 대부분이 dead 또는 불안정" — **실측으로 발동, dense/narrative-selected 두 표본 구성 방식(총 9개 설정) 비교 이후에도 해소 안 됨**(위 pilot 결과 표 참조). 근본 원인을 특정했다: (1) 이 데이터셋(55농장×14일, 원시 이벤트 222,309건) 자체의 낮은 유효 차원(PCA로 384차원 중 29개가 99% 분산 설명 — **원인 특정 완료, 표본을 dense/narrative-selected 어느 쪽으로 구성해도 동일함을 직접 검증**), (2) dict_size를 그 차원(29)에 맞춰도 남는 TopK "승자독식" 학습 역학(dict=32~64에서도 dead ratio가 0.40~0.50 바닥 — **미해결**). (1)은 "더 넓거나 더 잘 고른 표본"으로 해결되는 문제가 아님이 확인됐으므로, concept_governance/world_model 등 다음 단계로 넘어가려면 (2)를 겨냥한 남은 방향(AuxK loss, 더 공격적인 resampling, 다른 layer의 activation)을 시도하거나, 이 데이터셋 규모에서 SAE pilot 자체의 기대 수준을 재설정해야 한다.
