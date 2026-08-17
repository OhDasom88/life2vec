# sae

**근거**: 계획서 §9 (SAE 기반 기계적 해석)
**Phase**: Phase 4 · **범위**: 현재 필수 (§3.1, "선택 layer 대상 SAE pilot"만) · **상태**: 핵심 구현 완료(상태 머신·모델·평가·dead feature resampling) + 표본 구성 9종 + 6개 encoder layer 비교 + 랜덤 초기화·MLM/SOP 디코더 헤드 비교 + **SOP 재조정·서사 다양성 6-arm 실제 재학습 비교** 실측, 34개 테스트 통과. **원인 진단 완료(데이터의 유효 차원이 근본적으로 낮음, 표본 구성과 무관) + 압축이 학습으로 생긴 것임을 랜덤 초기화 대조로 확인 + 개선 방향 두 개 발견(최종 layer 대신 layer 2/4, SOP를 더 균형 있게), §18 중단 조건은 계속 발동 — 아래 참조**

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

### 효과가 있었던 방향 — 최종 layer 대신 중간 layer

앞서 "SAE 학습 알고리즘 쪽만 남은 레버"라고 썼는데, 이건 **틀렸다** — activation을 어느 layer에서 뽑느냐도 실제로 유의미한 차이를 만든다. `scripts/online2_v2/pilot_sae_layer_sweep.py`로 6개 encoder layer 전부의 pooled activation을 (`Transformer.forward_finetuning`을 고치지 않고 같은 루프를 복제해) 한 번의 순전파로 동시에 캡처해 비교했다(같은 14,544개 narrative-selected 타깃, dict=64, top_k=8, 15 epoch):

| layer | 유효차원(rank99) | explained_variance | dead_feature_ratio |
|---|---|---|---|
| 0 | 18 | 0.946 | 0.469 |
| 1 | 20 | 0.951 | 0.438 |
| **2** | 29 | 0.941 | **0.391**(6개 layer 중 최저) |
| 3 | 30 | 0.927 | 0.438 |
| 4 | **35**(6개 layer 중 최고) | 0.889 | 0.406 |
| 5(최종 layer — **지금까지 모든 pilot이 이걸 썼다**) | 29 | 0.923 | **0.500**(6개 layer 중 최악) |

**지금까지의 모든 실험(위 표 전부)이 6개 layer 중 dead_feature_ratio가 가장 나쁜 최종 layer만 썼다.** MLM/SOP 예측 헤드 바로 앞이라 과제에 맞춰 표현이 "눌린" 것으로 보인다 — layer 4는 유효 차원이 가장 높고(35), layer 2는 dead ratio가 가장 낮다(0.391, 지금까지 나온 모든 설정 중 최저값). 다만 이건 seed 1개짜리 결과이므로 재현성 확인 전까지는 "유력한 다음 후보"로만 취급한다.

**다음으로 시도해볼 것**: (1) layer 2/4를 기본으로 놓고 dict_size·sparsity를 다시 sweep(지금까지의 dict/L1 실험은 전부 최종 layer 기준이었다 — 최적 조합이 layer마다 다를 수 있음), (2) 여러 seed로 layer 2/4 결과 재현성 확인, (3) TopK 승자독식 자체를 겨냥한 것들(AuxK류 loss, 더 공격적인 resampling, 다른 초기화) — 여전히 유효한 방향이지만 이번엔 layer 2/4 위에서 시도해야 함.

### 이 압축이 "학습으로 생긴 것"인지 "구조 자체의 특성"인지 — 랜덤 초기화와 직접 비교

"사전학습이 빈칸 맞추기·순서 맞추기라는 과제를 익히면서 표현의 다양성을 없앤다고 봐도 되는가?"라는 질문을 검증하려면, 같은 구조를 **학습 안 시킨 채로** 같은 실험을 돌려 비교해야 한다. `scripts/online2_v2/pilot_sae_layer_decoder_random_comparison.py`로 (1) 체크포인트와 동일한 hparams로 architecture만 만들고 `load_state_dict`를 생략한 랜덤 초기화 인코더, (2) 그 최종 layer 출력을 실제 `MaskedLanguageModel`/`CLS_Decoder`의 예측 직전 변환(`V`+`tanh`+`l2_norm`, `in_layer`+`swish`+`ScaleNorm`)에 통과시킨 표현까지 — 같은 14,544개 narrative-selected 타깃, 같은 dict=64/top_k=8/15 epoch로 학습된 모델과 나란히 측정했다.

| position | 학습된 인코더 rank99 | 학습된 dead | 랜덤 초기화 rank99 | 랜덤 초기화 dead |
|---|---|---|---|---|
| layer0 | 18 | 0.469 | 118 | 0.406 |
| layer1 | 20 | 0.438 | 118 | 0.406 |
| layer2 | 29 | 0.391 | 118 | 0.406 |
| layer3 | 30 | 0.438 | 118 | 0.406 |
| layer4 | 35 | 0.406 | 118 | 0.406 |
| layer5(최종) | 29 | 0.500 | 118 | 0.406 |
| mlm_transform | 52 | 0.516 | 103 | 0.875 |
| sop_transform | 15 | 0.547 | 1 | 0.875 |

**랜덤 초기화 6개 layer가 전부 bit-for-bit 동일하다(rank99=118, EV=0.434528 소수 6자리까지 일치) — 이건 버그가 아니라 이 체크포인트의 `norm_type="rezero"` 설정 때문에 생기는 정확한 항등식이다.** `ReZero.forward(x, y) = x + y * self.weights`이고 `self.weights`는 `torch.zeros(1)`로 초기화된다(`src/transformer/transformer_utils.py:275`) — 즉 학습 전에는 **모든 encoder layer가 수학적으로 정확히 항등함수**다(residual 게이트가 정확히 0이라 sublayer 출력이 전부 지워짐). 그래서 layer 0~5 사이에서 관측되는 "상승 후 하강" 패턴은 전부 학습이 만든 것이다 — 학습을 안 시키면 깊이가 몇이든 표현이 전혀 안 바뀐다.

**결론 1 — 사용자 가설이 실측으로 뒷받침된다.** 랜덤 초기화 상태의 유효 차원(6개 layer 전부 118)은 학습된 모델의 어느 layer보다도 훨씬 높다(학습된 쪽은 15~35). MLM/SOP라는 좁은 과제로 학습을 시키자 표현의 선형 자유도가 384차원 중 100차원대에서 15~35차원대로, **3~7배 줄어들었다.** 이건 architecture/depth 자체의 부산물이 아니라(랜덤 초기화에서는 깊이가 전혀 영향을 안 준다는 걸 위에서 확인했다), 학습이 직접 만든 압축이다.

**결론 2 — 다만 "디코더 헤드로 갈수록 계속 줄어든다"는 아니다(단순하지 않음, 정직하게 기록).** 학습된 모델 기준으로 SOP 변환(rank99=15)은 layer5(29)보다 더 압축됐지만, **MLM 변환(rank99=52)은 오히려 layer5보다 유효 차원이 더 높다** — PCA 기준으로는 "더 다양해졌다." 반면 SAE dead_feature_ratio는 두 디코더 변환(0.516, 0.547) 모두 layer5(0.500)보다 나쁘다 — PCA 유효 차원과 SAE dead ratio가 항상 같은 방향으로 움직이지는 않는다는 뜻이다. 랜덤 초기화 쪽에서는 두 디코더 변환 모두(특히 SOP는 rank99=1로 완전히 붕괴) 원본 encoder layer(118)보다 훨씬 나쁘다 — 디코더 헤드는 ReZero 항등식 보호를 안 받는 별도의 랜덤 초기화 Linear라서, 학습 여부와 무관하게 그 자체로 표현을 강하게 재구성한다는 뜻으로 보인다.

- [`scripts/online2_v2/pilot_sae_layer_decoder_random_comparison.py`](../../../../../../scripts/online2_v2/pilot_sae_layer_decoder_random_comparison.py) — 랜덤 초기화 인코더(`load_untrained_encoder`, hparams는 체크포인트와 동일하되 `load_state_dict` 생략) + `model.mlm_decoder`/`model.cls_decoder`의 예측 직전 변환까지 포함해 8개 지점(6 layer + MLM/SOP 변환)을 학습된 모델과 나란히 비교. 위 layer sweep 스크립트의 타깃 선택·pooling 패턴을 그대로 재사용.

### 왜 랜덤 초기화조차 384차원을 다 못 채우는가(rank99=118) — task 재설계가 검토할 만한지

두 가지를 코드/데이터로 직접 확인했다.

**(A) 랜덤 초기화에서는 위치/시간 정보가 아예 꺼져 있다.** `Embeddings.forward`(`src/transformer/embeddings.py:53-69`)를 보면 `age`, `abspos`, `segment` 임베딩이 전부 `self.res_age`/`self.res_abs`/`self.res_seg`라는 별도의 `ReZero` 게이트를 거쳐 residual로 더해진다 — 이 게이트들도 encoder layer의 ReZero와 똑같이 `torch.zeros(1)`로 초기화된다. 즉 **학습 전에는 `self.token(tokens)` 토큰 정체성 임베딩만 남고 위치·시간·segment 정보는 전부 0으로 지워진다.** 랜덤 초기화 모델이 측정한 "다양성"은 사실 순수 토큰 lookup만의 다양성이다.

**(B) 그 토큰 lookup의 다양성 자체가 이미 데이터 쪽에서 제한돼 있다 — 직접 대조 실험으로 확인.** 같은 14,544개 target span에서 학습된 임베딩을 아예 빼고, 토큰 id의 등장 여부만으로 만든 **bag-of-tokens**(정규화된 728차원 one-hot 평균, 학습 가능한 파라미터 전혀 없음) 벡터의 PCA 유효 차원을 재봤다: 50%→1, 90%→40, 95%→76, **99%→135**. 랜덤 초기화 인코더가 보인 rank99=118과 거의 같은 자릿수다(등장한 distinct token id는 728개 중 530개로, vocab 용량이 병목이 아니다). **즉 랜덤 임베딩이라 해도 어떤 토큰들이 같이 등장하는지(co-occurrence)의 다양성 자체가 이미 ~120~135차원어치밖에 안 되고, 랜덤 embedding table은 그 구조를 R^384에 선형으로 옮겨 담을 뿐 새 독립 방향을 만들어내지 않는다.**

**결론 — "초기 layer가 100%가 안 되는 것"은 버그도 아니고 task 설계 문제도 아니다.** 이건 55농장×14일, 물리적으로 강하게 얽힌 센서값을 728개 구간(bin) 토큰으로 양자화한 이 데이터셋 자체가 가진 상한이다(앞서 raw 센서값 PCA로 확인한 384차원 중 29차원이라는 숫자와 같은 계열의 사실 — bag-of-tokens 단계에서는 아직 "값 자체"가 아니라 "어떤 토큰들이 같이 나타나는가"만 보므로 135로 더 높게 나오지만, 방향은 같다). architecture를 바꾸거나 masking/SOP 설계를 바꿔도 이 상한 자체는 못 올린다 — 원시 데이터가 더 다양해지지 않는 한 100%는 애초에 도달 불가능한 목표다. 여기까지는 "고칠 수 없는 것"이다.

**반면 학습이 118→15~35로 만드는 추가 압축(위 "이 압축이 학습으로 생긴 것인지" 절)은 별개 문제이고, 이쪽은 task 설계와 관련이 있어 보인다.**

**정정**: 처음엔 실제 실행 설정을 `conf/task/online2_v2_grouped_mlm.yaml`(Hydra 경로)에서 확인하고 "SOP 라벨 분포가 90/5/5"라고 썼는데, 이건 틀렸다 — 이 체크포인트는 그 Hydra 경로로 학습되지 않았다. 실제로는 `scripts/online2_v2/run_v2_pretrain_loop.py`(순수 PyTorch 루프, `launch_full_earlystop_detached.sh`로 기동)로 학습됐고, 그 실행의 `run_manifest_v2.json`을 직접 열어보면 `sop_reverse_probability=0.2`, `sop_shuffle_probability=0.2`다 — **정상(60%)/역순(20%)/셔플(20%)로 이미 상당히 균형 잡혀 있었다**, 90/5/5가 아니다. 설정 파일 이름만 보고 실제 실행에 쓰였다고 성급히 결론 내린 게 원인이다 — 앞으로는 `run_manifest_v2.json`처럼 실행 시점에 실제로 기록된 값을 먼저 확인하고, config yaml은 그 값과 대조해서만 참고한다.

- **SOP 라벨 분포(60/20/20)는 이미 어느 정도 균형 잡혀 있다.** 그런데도 학습된 `sop_transform`이 rank99=15, explained_variance=0.973으로 8개 지점 중 가장 "뾰족한" 분포로 나왔다는 건, **SOP 자체가 유력한 단일 원인이라는 근거가 약해졌다는 뜻이다** — 클래스 불균형이 아니라면 (1) label=0/1/2 세 종류를 구분하는 데 필요한 신호 자체가 원래 저차원(정상/역순/셔플을 가르는 데 CLS 표현의 몇 개 방향만 있으면 충분할 수 있음)이거나, (2) `CLS_Decoder`의 `ScaleNorm`이 모든 벡터를 같은 노름의 shell로 강제 정규화하는 구조적 특성 때문일 수 있다. 이 자체가 아직 결론이 아니라 재학습으로 검증할 질문이다 — 이미 균형 잡힌 60/20/20보다 더 균형(예: 33/33/33에 가깝게)을 주면 rank가 더 올라가는지, 아니면 안 올라가서 "class balance는 원인이 아니었다"고 결론 내리게 되는지 실측이 필요하다.
- **mask_ratio=0.30에 measurement-group 단위 masking**(`GroupedMLMMasker`, `src/online2/v2/masking.py`)이 걸려 있다 — 개별 토큰이 아니라 한 measurement group 전체를 30% 확률로 통째로 가린다. 이미 알고 있듯 이 데이터는 변수 간 상관이 매우 높다(연속 이벤트 코사인 유사도 평균 0.89, raw PCA 유효 차원 29). 상관이 이렇게 높으면 가려진 group도 남은 group들로부터 거의 선형 보간하듯 복원 가능해서, MLM이 "29차원짜리 충분통계량만 잘 압축해서 들고 있으면 풀리는" 얕은 지름길이 될 수 있다 — **현재 설정과 결과를 잇는 그럴듯한 메커니즘이지 확정된 인과관계는 아니다.**

두 가설 다 검증하려면 실제 재학습(지금까지의 실험은 전부 기존 체크포인트 재사용, 이쪽은 **GPU-시간 단위의 새 사전학습 실행**)이 필요하다 — 아래 "재학습 실측" 절 참조.

**요약**: (1) 초기 layer가 100%가 안 되는 건 데이터의 물리적 상한 때문이며 task 재설계로 해결할 문제가 아니다. (2) 학습이 만드는 추가 압축(118→15~35)에 대해서는 SOP 클래스 불균형이 구체적 근거가 있는 유력 후보이고, masking 방식은 그럴듯하지만 미검증인 후보다 — 둘 다 실제로 검증하려면 재학습이 필요하므로, 착수 전 사용자 확인이 필요하다.

### 재학습 실측 — SOP 재조정 + 서사 다양성 6-arm 비교

사용자 확인 후 실제로 6개 조건을 재학습했다(`scripts/online2_v2/run_narrative_ablation_pretrain_sweep.sh`, 전부 batch_size=40·max_length=1024·**동일 5000 step**, `--skip-vram-calibrate`로 배치 크기 자동조정 끔, `--no-early-stop`으로 조건 간 학습량을 강제로 동일하게 고정). 비교는 `scripts/online2_v2/compare_narrative_ablation_checkpoints.py`로, 앞서와 같은 14,544개 narrative-selected 타깃에 대해 layer5(최종)·mlm_transform·sop_transform 3개 지점의 PCA 유효 차원 + SAE(dict=64, top_k=8, 15 epoch) dead_feature_ratio를 측정했다.

**중요한 실행상 함정 두 가지, 실측 전에 발견해서 고쳤다**: (1) `best.ckpt`는 최고 val loss 시점을 저장하므로 조건마다 다른 step(4200~5500)에 해당했다 — "서사 내용 차이"와 "학습량 차이"가 섞일 뻔했다. 6개 조건 전부 정확히 `checkpoint_step_5000.pt`로 통일해서 비교했다. (2) `single_table_only` arm은 세션이 한 번 끊겨서 step 1200에서 재개했는데, `run_v2_pretrain_loop.py`의 `--resume`은 `--steps`를 누적으로 처리한다(`while step < start_step + steps`, `run_v2_pretrain_loop.py:687`) — 그래서 최종적으로 6200 step까지 갔다. 위와 같은 이유로 `checkpoint_step_5000.pt`를 써서 우회했다.

| arm | 코퍼스 | 행 수 | SOP 비율 | layer5 rank99 | layer5 dead | sop_transform rank99 | sop_transform dead |
|---|---|---|---|---|---|---|---|
| control | 전체 80개 서사 | 15.0M | 0.2/0.2(원본과 동일) | 9 | 0.734 | 6 | 0.781 |
| sop_balanced | 전체 80개 서사 | 15.0M | **0.33/0.33** | **17** | **0.391** | 7 | 0.547 |
| narrow_subset | 최빈 8개 서사만 | 8.7M | 0.2/0.2 | 13 | 0.672 | 5 | 0.875 |
| single_table_only | 단일 테이블 서사 34개 | 3.6M | 0.2/0.2 | **31** | 0.547 | 10 | 0.844 |
| multi_table_only | 복수 테이블 서사 46개 | 11.4M | 0.2/0.2 | 18 | 0.594 | 6 | 0.625 |
| dedup_reduced | (농장,타깃이벤트)당 1개로 중복 제거 | 4.4M | 0.2/0.2 | 12 | **0.312**(전체 중 최저) | 6 | 0.812 |

(전체 결과는 `outputs/online2/sae_pilot/report_narrative_ablation.json`.)

**결과 1 — SOP 재조정은 실제로 효과가 있다(재현된 긍정적 결과).** control(0.2/0.2)과 sop_balanced(0.33/0.33)는 서사 구성이 완전히 같고 SOP 비율만 다른 순수 대조 실험이다. 더 균형 잡힌 쪽이 layer5 rank99를 9→17로 거의 두 배 올리고, dead_feature_ratio를 0.734→0.391로 거의 절반으로 낮췄다 — 두 지표가 같은 방향으로 함께 움직였다는 점에서 잡음이 아니라 실제 효과로 보인다. 앞서 정정한 대로 원본 체크포인트는 이미 0.2/0.2로 어느 정도 균형(60/20/20)이었는데도, **거기서 한 번 더 균형을 주면 여전히 남은 여지가 있었다.**

**결과 2 — 서사 다양성의 영향은 직관과 반대이고, 명백한 교란변수가 있다(정직하게 기록).** 전체 80개 서사(control, rank9)보다 **더 좁은** 서사 구성(narrow_subset 13, single_table_only **31**, multi_table_only 18)이 전부 layer5 유효 차원이 더 높게 나왔다 — "서사가 다양할수록 표현이 풍부해진다"는 직관과 정반대다. 하지만 이건 그대로 믿을 수 없다: **행 수가 조건마다 크게 다른데(3.6M~15.0M) step 수는 전부 5000으로 고정했다** — 즉 코퍼스가 작을수록 같은 step 안에서 데이터를 상대적으로 더 많이 반복해서 봤다(single_table_only는 전체 코퍼스의 24%밖에 안 되는 행 수로 같은 step을 돌았다). 그래서 지금 관측된 "서사가 좁을수록 rank가 높다"는 "서사 내용 자체의 효과"인지 "같은 step 안에서 상대적으로 더 많이 epoch를 돈 효과"인지 이 실험만으로는 분리가 안 된다 — **다음에 검증하려면 step 수가 아니라 epoch 수(또는 unique row 노출량)를 조건마다 맞춰야 한다.**

**결과 3 — 서사 기반 중복(재윈도잉)을 제거하면 dead ratio가 가장 낮아진다.** `dedup_reduced`(같은 (농장, 타깃이벤트)에 대해 여러 서사가 중복 생성한 시퀀스 중 하나만 남김, 942,727→178,142 시퀀스)는 rank 자체는 중간(12)이지만 **dead_feature_ratio는 6개 조건 중 가장 낮다(0.312)**. 이건 이 세션 초반에 다른 방식(다운스트림 activation 표본 구성만 바꾼 것)으로 확인했던 "서사 기반 재윈도잉이 실질적 다양성을 늘려주지 않는다"는 결론과 같은 방향이지만, 이번엔 **사전학습 자체를 그 중복 제거된 데이터로 다시 돌려서** 얻은 직접적인 인과 증거라는 점이 다르다 — 다운스트림 표본 선택보다 사전학습 데이터 구성 자체를 바꾸는 쪽이 dead ratio에 측정 가능한 영향을 준다.

**결과 4 — 학습량과 압축의 관계가 단조롭지 않다(기존 서술을 복잡하게 만드는 새 발견).** control(SOP 0.2/0.2, 5000 step)의 layer5 rank99는 9인데, 똑같이 SOP 0.2/0.2로 학습했지만 **18600 step까지 돈 원본 체크포인트는 rank99=29**였다(위 "랜덤 초기화와 직접 비교" 절 참조). 랜덤 초기화(0 step, rank118) → 5000 step(rank9, 급격한 붕괴) → 18600 step(rank29, 부분 회복)이라는 그림이 된다. 즉 "학습이 진행될수록 다양성이 단조롭게 줄어든다"는 이전까지의 단순한 서술은 **정정이 필요하다** — 초반에 빠르게 collapse했다가 이후 다시 어느 정도 diversify하는, 비단조적인 동역학으로 보인다. 이건 아직 두 지점(5000, 18600)만 찍어본 것이라 궤적의 정확한 모양은 모른다 — 중간 step들을 더 찍어봐야 확정할 수 있다.

**결과 5 — sop_transform은 조건과 무관하게 항상 가장 압축된 지점이다.** 6개 조건 전부에서 sop_transform의 rank99(5~10)가 layer5(9~31)나 mlm_transform보다 낮다 — SOP 재조정이나 서사 구성을 바꿔도 이 상대적 순서 자체는 안 바뀐다. SOP 헤드의 구조(`CLS_Decoder`의 `ScaleNorm`이 모든 벡터를 같은 노름의 shell로 강제) 자체가 이 위치를 다른 지점보다 더 압축시키는 경향이 있어 보인다.

**결론**: 재학습 전 세웠던 두 가설 중 SOP 불균형 쪽은 **긍정적으로 검증됐다**(더 균형 잡을수록 더 좋아짐, 재현 가능한 대조 실험). 서사 다양성 쪽은 **효과가 있는 건 분명해 보이지만 방향과 원인이 예상과 다르고 행 수(epoch 노출량) 교란변수가 있어 이 실험만으로 확정할 수 없다** — narrative_id 필터가 아니라 epoch 수를 통제 변수로 고정한 후속 실험이 필요하다. dedup(중복 제거)은 dead ratio 쪽에서 뚜렷한 개선을 보였다.

### micro-event 분리(event 경계 재정의) — 결과 5의 원인 절반만 검증, 혼재된 결과

위 6-arm 실험과는 다른 축의 개입이다. `build_v2.py`의 `tokenize_events()`는 하나의 raw 관측 행(예: 한 시각의 E_environment 판독값)에 있는 여러 컬럼(co2_ppm, inside_humidity_pct, ...)을 `[MEAS_SEP]`로만 구분해 **하나의 event 토큰 스팬**으로 번들링한다. `GroupedMLMMasker`는 마스킹할 group(measurement_group_id)은 올바르게 고르지만, **고른 group 안에서는** 각 토큰(FEATURE/VALUE_ABS/VALUE_GLOBAL_REL/...)마다 독립적으로 80/10/10을 추첨한다 — 절대/전역상대/농장상대 3개 척도로 같은 물리량을 중복 표현한 토큰이 서로 다른 bin으로 랜덤 치환돼 물리적으로 불가능한 조합이 섞여 들어갈 수 있다는 뜻이다. 이게 **결과 5**(sop_transform이 조건과 무관하게 항상 가장 압축된 지점)의 원인일 수 있다는 가설을 세우고, `scripts/online2_v2/build_microevent_training_corpus.py`로 검증했다.

**주의 — 이 개입은 이 문제의 절반만 다룬다.** 위에서 말한 두 문제 중 "event 경계 자체가 여러 물리량을 부적절하게 묶는다"는 부분만 고쳤다(원래 life2vec처럼 원자값 하나 = 한 measurement group을 그 자체로 독립된 event로 재정의, `training_events_v2.parquet`을 스트리밍으로 다시 읽어 SENTENCE를 group별로 쪼갬, 원시 빌드 재실행 없음). **"고른 group 안에서도 각 토큰이 여전히 독립적으로 마스킹된다"는 절반의 문제는 그대로 남아 있다** — `masking.py`의 per-token 추첨 로직 자체는 손대지 않았다. 즉 이번 실험은 "물리량이 서로 다른 group끼리 한 event에 섞이는 것"만 없앴을 뿐, "같은 group 안에서도 절대/전역상대/농장상대 표현이 서로 다르게 마스킹되는 것"은 여전히 가능하다 — 이건 마스킹 알고리즘 자체를 고치는 별도 작업이 필요하다.

control과 동일 조건(batch_size=40·max_length=1024·5000 step·SOP 0.2/0.2·`--no-early-stop`)으로 micro-event 코퍼스를 재학습하고, 같은 `compare_narrative_ablation_checkpoints.py`(14,544개 narrative-selected 타깃, control과 동일한 원본 event 구조로 평가 — vocab·평가 window는 안 바꿨고 학습 데이터만 바뀌었다)로 비교했다:

| arm | layer5 rank99 | layer5 dead | mlm_transform rank99 | mlm_transform EV | mlm_transform dead | sop_transform rank99 | sop_transform EV | sop_transform dead |
|---|---|---|---|---|---|---|---|---|
| control | 9 | 0.734 | 19 | 0.925 | 0.500 | 6 | 0.920 | 0.781 |
| microevent_split | **17** | **0.625** | 19 | 0.855 | 0.547 | **10** | **0.319** | **0.875** |

(원본 checkpoint: `outputs/online2/v2_runs/narrative_ablation/microevent_split/checkpoint_step_5000.pt`, 전체 결과는 `outputs/online2/sae_pilot/report_narrative_ablation.json`의 `microevent_split` 항목.)

**결과가 깨끗하지 않다 — 지점마다 반대 방향으로 움직였다.** layer5는 sop_balanced와 비슷한 방향으로 개선됐다(rank99 9→17, dead 0.734→0.625) — event 경계를 좁히는 것도 SOP 재조정과 마찬가지로 최종 layer의 압축을 어느 정도 풀어주는 것으로 보인다. 하지만 애초에 이 개입으로 고치려던 sop_transform은 **오히려 악화됐다**: rank99는 6→10으로 올랐지만 dead_feature_ratio는 0.781→0.875로 더 나빠졌고, explained_variance는 0.920→0.319로 크게 떨어졌다 — SAE가 이 activation을 거의 재구성하지 못한다는 뜻이라 rank99 상승을 "다양성 개선"으로 곧이곧대로 읽을 수 없다(분산 구조 자체가 달라졌을 가능성이 크다). mlm_transform은 rank는 그대로(19)인데 EV·dead 모두 소폭 나빠졌다. 즉 **"결과 5의 원인이 event 경계 번들링이었다"는 가설은 기각**됐다 — 최소한 마스킹 로직을 손대지 않은 채로는 sop_transform의 압축을 풀지 못했고, 오히려 SAE 재구성 안정성만 떨어뜨렸다. 위에서 밝힌 대로 이 실험은 문제의 절반(event 경계)만 다뤘으므로, 나머지 절반(group 내부 per-token 독립 마스킹)을 고치기 전까지는 이 가설을 완전히 기각했다고 보기도 이르다 — 두 문제가 함께 얽혀 있어서 하나만 고쳐서는 개선이 안 보였을 가능성도 배제할 수 없다.

### 모델을 안 거치고 직접 확인 — 상대 bin 제거+세분화 / narrative·시각·장소를 추가하면 정보가 느는가

사용자 요청: "원본 데이터에 최대한 가까운 피쳐 사용(분석 목적), 지금 장소별·전체 데이터별 상대 bin을 제거, token 세분화, narrative 추가 — 이벤트 내용이 동일해도 어떤 서사에 속하는지, 어느 시점·장소인지가 중요할 것 같다." 이 네 가지를 사전학습 corpus/vocab을 바꾸지 않고(사용자 확인: 먼저 분석 전용으로 빠르게 확인) `scripts/online2_v2/check_raw_narrative_time_location_diversity.py`로 직접 검증했다 — 학습 가능한 파라미터가 전혀 없는 `check_token_bag_diversity.py`와 같은 계열의 대조 실험이다.

**왜 이게 필요한가.** 지금 `tokenize_events()`(`scripts/online2_v2/build_v2.py`)는 연속형 피쳐마다 `VALUE_ABS`(절대 bin, 기본 10개 안팎) + `VALUE_GLOBAL_REL`(전체 데이터셋 대비 상대 bin) + `VALUE_FARM_REL`(그 농장 대비 상대 bin) 3벌을 동시에 넣는다(`src/online2/v2/tokenizer.py:126-148`). 그리고 `narrative_id`는 순수 메타데이터일 뿐 이벤트 SENTENCE에 전혀 안 들어가고(코드로 확인: vocab 728개 토큰 중 `NARRATIVE` 접두어는 하나도 없음), 이벤트별 정확한 관측 시각·농장/구역 정체성도 이벤트 토큰엔 없다(`FARM_LOCAL`/`ZONE_LOCAL` 토큰은 있지만 시퀀스 레벨 static prefix로만 들어가고 이벤트별은 아님, `LOCAL_HOUR`/`DAY_FROM_START` 파생 bin만 일부 존재).

**방법.** `legacy_build/cell_occurrences.parquet`(build_v2.py의 `tokenize_events()`가 쓰는 것과 동일한 원본 raw 소스)에서 직접 raw 값을 읽어, GLOBAL_REL/FARM_REL 없이 VALUE_ABS만 남기고 bin 개수를 10→**100**(quantile)으로 세분화한 벡터를 만들었다(요청대로 "상대 bin 제거"+"bin 개수를 늘림"). 여기에 narrative_id/farm_id·zone_id/정확한 관측 시각(hour-of-day sin·cos + day-from-start)을 채널로 하나씩 추가하며, 같은 14,544개 narrative-selected 타깃(다른 모든 pilot과 동일 샘플링)에 대해 조합별 PCA 유효 차원을 쟀다. **주의**: 컬럼 표준화(z-score) 없이 돌렸더니 연속값(sin/cos/day)이 큰 스케일로 분산을 독차지해서 rank99가 오히려 떨어지는 인위적 결과가 나왔다 — 전체 컬럼을 표준화한 뒤 재측정했다(스케일이 다른 one-hot·연속 채널을 섞은 PCA는 표준화 없이는 신뢰할 수 없다는 걸 직접 확인).

| 조합 | 차원 | rank99 | rank90 | (추가 차원 대비) rank 기여율 |
|---|---|---|---|---|
| abs_fine(상대 bin 제거+100bin)만 | 2000 | 1267 | 1020 | — (기준) |
| + narrative_id | 2080(+80) | 1335(+68) | 1072 | **85%** |
| + 정확한 시각(sin/cos/day) | 2003(+3) | 1270(+3) | 1022 | 100% |
| + farm/zone 위치 | 2060(+60) | 1322(+55) | 1061 | 92% |
| + 전부 | 2143(+143) | 1391(+124) | 1114 | 87% |

(전체 결과는 `outputs/online2/sae_pilot/report_raw_narrative_time_location.json`.)

**결과 — 사용자 가설이 맞았다: narrative/시각/장소는 실제로 지금 모델이 못 보는 구분 가능한 정보다.** 넷 중 narrative_id가 가장 큰 기여(추가한 80차원 중 68개, 85%가 새 분산을 설명)를 보였고, 정확한 관측 시각은 100%(3차원 전부 기존 벡터로 설명 안 되는 새 정보), farm/zone 위치도 92% — 어느 것도 무시할 만한 수준이 아니다. **단, 이 숫자를 모델의 rank99(control layer5=9, sop_transform=6, 384차원 중)와 직접 비교하면 안 된다** — 여기 쓴 벡터는 대부분 one-hot(희소)이라 구조적으로 rank가 차원 수에 비례해 크게 나온다(학습된 dense 임베딩과는 다른 종류의 공간). 이 실험이 실제로 답하는 건 "raw 데이터 자체의 절대적 정보량"이 아니라 **"narrative/시각/장소가 raw 센서값만으로는 설명 안 되는 독립적 변량을 갖고 있는가"** — 답은 그렇다. 사전학습 자체가 이 정보를 아예 못 보는 지금 구조에서는, 다양성 결핍(§18)의 일부가 "물리 데이터의 근본 한계"가 아니라 "안 넣어준 정보"에서 온다는 뜻일 수 있다.

**다음 단계 — 실제로 진행함(아래 두 섹션).** 이 결과가 유의미해서, 사용자 확인 후 (1) 서사 카탈로그를 80→88개로 확장하고 (2) 실제 사전학습 corpus/vocab을 바꾸는 재학습(Phase 2)까지 진행했다.

### 서사 카탈로그 확장 — 80 → 88개, 짧은/긴 시간축·횡단·이미지 축 보강

기존 80개 서사는 세 축 모두 불균형했다(실측): 시간축은 60/80이 시간~2일 스케일에 몰려 있고 13일급은 `G01-G10`(장기생육) 10개뿐, 종단(`STRICT_CHRONOLOGICAL`) 72/80 대비 횡단(`SET_COMPARISON`)은 3개뿐(`A11`/`X04`류, 전부 동일 농장 내 구역 비교 — 농장 *간* 비교는 전무), 이미지 사용 서사는 3/80(`X08-X10`, 전부 "이미지 한 장 + 과거 센서 맥락"이라 이미지 자체의 시간 변화는 없음). `scripts/generate_online2_narrative_catalog.py`(SPECS 목록)와 `src/online2/materializers.py`(매처)에 8개를 추가했다:

- **횡단 4개(신규 카테고리 "환경비교")**: `X13`/`X14`(동시각 여러 농장 온습도·CO2 비교), `X15`(동일 생육조사 순번의 여러 농장 비교). **설계 이슈 하나 실측으로 발견**: 농장 간 절대 timestamp가 거의 안 겹친다(정확히 같은 시각을 공유하는 농장이 최대 9개/55개 — 농장마다 관측 캘린더 기간 자체가 다름). 그래서 절대 시각 대신 "지역시각(hour-of-day)"·"생육조사 순번(0=첫 조사, 1=둘째 조사)"으로 정렬했더니 이 기준으로는 55개 농장 전부가 존재해서 매칭이 됐다 — `_crossfarm_by_hour`/`_crossfarm_by_survey_index`(`materializers.py`).
- **이미지 장기 서사 1개**: `X12`(농장별 최초·최근 이미지 페어, 최대 약 13일 스팬). **실측으로 걸린 함정**: 이미지 이벤트는 토큰이 `IMAGE_EMBED_SLOT` 1개뿐이라 이미지 두 장만으로는 최소 유효 토큰 수(16)를 못 채워 16건 후보 전부 `TOO_SHORT`로 드롭됐다 — 각 이미지 직전 환경 맥락(최대 3개)을 붙여서 해결(`_image_longitudinal_pair`).
- **짧은(sub-hour) 단일시점 4개**: `D16`/`D17`(양액시스템 전환·순환팬 가동 순간)/`S13`/`S14`(풍속·pH 이상 순간) — 안 쓰이던 raw 컬럼 위주로 골라 max_events=1(사실상 지속시간 0)로 만듦.

원시 빌드(`scripts/online2 build`, 실제 `MaterializerRegistry` 매칭이 일어나는 단계 — `build_v2.py`가 아니라 `src/online2/cli.py`/`builder.py`라는 걸 이번에 확인함)를 88개 카탈로그로 재실행: **88/88 전부 정상 매칭**(`unsupported_template_count=0`), 967,012→1,006,023 시퀀스.

### Phase 2 — 상대 bin 제거 + narrative/시각/장소를 실제 토큰으로

사용자 요청("원본 데이터에 가까운 피쳐, 상대 bin 제거, token 세분화, narrative/시각/장소 추가")을 위 분석 전용 검증 이후 실제 사전학습 파이프라인에 반영:

1. **`src/online2/v2/tokenizer.py`**: `VALUE_GLOBAL_REL`/`VALUE_FARM_REL` 및 그 `_combined` 토큰 생성 코드를 제거, `VALUE_ABS`만 유지.
2. **`scripts/online2_v2/build_v2.py`**: `fit_binning_v2(..., bins=100)`로 절대값 bin을 10→100분위 세분화(분석 전용 스크립트와 동일 기준).
3. **이벤트별 narrative/farm/zone 토큰 추가** — 처음에 `build_v2.py`의 `materialize_sequences()`에 넣었지만, **이게 실제로는 죽은 코드였다는 걸 뒤늦게 발견**: control이 실제로 학습한 이벤트 단위 `training_events_v2.parquet`(1,500만 행, event_position/AGE 등 실제값)은 `build_v2.py`의 `export_training()`이 아니라 **별도 스크립트 `scripts/online2_v2/export_training_events_v2_event_grain.py`**가 만든다(`export_training()`은 코드 주석에 "Placeholder only for flat sequence export"라고 명시돼 있고, AGE=0.0·시퀀스당 1행 고정이라 실제 학습에 쓰인 적이 없다 — 처음엔 이걸 못 보고 그대로 재학습을 돌릴 뻔했다). 게다가 `export_training_events_v2_event_grain.py`는 SENTENCE를 `sequences_v2.parquet`가 아니라 `events_tokenized_v2.parquet`에서 이벤트별로 새로 조립하기 때문에, `materialize_sequences()`에 넣은 토큰은 이 실제 경로에 아예 도달하지 않았다. **`export_training_events_v2_event_grain.py`의 `expand_sequence_rows()`에 직접** 토큰 주입 로직을 옮겨 고쳤다 — 시퀀스 안에 실제로 등장하는 farm/zone만 골라 로컬 순서번호를 매기고, 매 이벤트 SENTENCE 앞에 `NARRATIVE|<id> FARM_LOCAL|<i> ZONE_LOCAL|<i>`를 붙인다. (정확한 관측 시각은 이미 `AGE`라는 연속값 컬럼으로 모델에 들어가고 있어서 — Phase 1이 확인한 "정확한 시각의 기여율 100%"를 이미 만족한다 — 별도 토큰을 추가하지 않았다.)
4. **`src/online2/v2/vocab.py`**: `FARM_LOCAL` 슬롯을 8→24로 확장(횡단 서사 `X13`/`X14`가 시퀀스 하나에 최대 20개 서로 다른 농장을 담으므로 기존 8칸으로는 out-of-vocab이 나서 전부 `[UNK]`로 새 나갈 뻔했다), 88개 `NARRATIVE|<id>` 토큰 추가. vocab_size 728→832.

새 lineage(`outputs/online2/v2_build_expanded88_phase2/`, `training_events_v2.parquet` 920,254시퀀스·14,348,896행)로 control과 동일 조건(batch=40·max_length=1024·5000 step·SOP 0.2/0.2)으로 재학습 후 같은 14,544~15,470개 narrative-selected 타깃으로 비교:

| arm | layer5 rank99 | layer5 dead | mlm_transform rank99 | mlm_transform EV | mlm_transform dead | sop_transform rank99 | sop_transform EV | sop_transform dead |
|---|---|---|---|---|---|---|---|---|
| control | 9 | 0.734 | 19 | 0.925 | 0.500 | 6 | 0.920 | 0.781 |
| expanded88_phase2 | **13** | **0.547** | 15 | 0.924 | **0.625** | 6 | **0.948** | **0.750** |

(전체 결과는 `outputs/online2/sae_pilot/report_narrative_ablation.json`의 `expanded88_phase2` 항목.)

**또 혼재된 결과다 — 지점마다 다른 방향.** layer5는 뚜렷이 개선됐다(rank99 9→13, dead 0.734→0.547) — micro-event 분리 실험(rank 9→17, dead 0.734→0.625)과 비슷한 방향·비슷한 크기다. sop_transform은 rank99는 그대로(6)지만 EV(0.920→0.948)와 dead(0.781→0.750) 둘 다 소폭 개선됐다 — 이번 세션에서 sop_transform이 개선된 유일한 개입이다. 반면 **mlm_transform은 악화됐다**: rank99가 19→15로 줄고 dead_feature_ratio도 0.500→0.625로 나빠졌다 — EV는 거의 그대로(0.925→0.924)인데 rank·dead 둘 다 나빠진 조합이라 이 지점에서는 새 토큰·bin 구성이 오히려 표현을 더 압축시킨 것으로 보인다. **결론**: "상대 bin 제거 + 세분화 + narrative/farm/zone 토큰 추가"는 이번 세션의 다른 개입들과 마찬가지로 §18(dead ratio)을 깨끗이 해소하지 못했다 — 한 지점(layer5)은 확실히 좋아졌고, 한 지점(sop_transform)은 소폭 좋아졌고, 한 지점(mlm_transform)은 나빠졌다. sop_balanced(SOP 재조정)가 지금까지 유일하게 세 지점 모두에서 일관되게 개선된 개입이라는 점은 여전히 유효하다.

### 서사 유형이 앞으로 늘어난다면 — 자동 매칭 vs 사람이 직접 검토하는 것의 차이

오늘 결과를 근거로 이 질문에 답한다: 서사 템플릿을 지금(80개)보다 늘리고 그에 따라 학습에 쓰이는 이벤트 시퀀스 구성이 자동으로 달라진다면, 자동 규칙 매칭(`src/online2/materializers.py`/`builder.py`/`catalog.py`)과 사람이 서사-원시데이터 관계를 직접 검토하는 것(`ui/data_grounding_curation/`의 §5.3 검토 큐가 이미 이 역할을 하는 화면이다)은 구조적으로 다른 실패 모드를 갖는다 — 오늘 실측이 그 차이를 구체적으로 보여준다.

- **자동 매칭은 "중복인지 아닌지"를 모른다.** 80개 템플릿이 같은 222,309건의 원시 이벤트 위에서 각자 독립적으로 매칭하기 때문에, 서사 유형이 늘수록 같은 원시 구간을 여러 템플릿이 중복으로 잡아내는 문제(현재도 942,727 시퀀스가 사실상 222,309건의 ~16배 재윈도잉)는 늘면 늘었지 저절로 줄지 않는다. **결과 3**(dedup_reduced가 dead ratio를 가장 크게 낮춤)이 보여주듯 이 중복은 실제로 학습된 표현의 품질에 측정 가능한 악영향을 준다 — 서사 유형이 늘어날수록 이 문제를 자동으로 잡아줄 장치(예: 지금의 `dedup_reduced`처럼 "같은 타깃 이벤트는 하나만" 같은 규칙)가 **더**, 아니라 인간이 "이 두 서사 매칭이 사실 같은 사건을 가리키는가"를 판단하는 것과 같은 종류의 일이 필요해진다. 지금 스크립트는 임의로 첫 번째 것만 남기지만, 사람이라면 여러 후보 중 어느 매칭이 더 정확한 서술인지 판단해서 남길 수 있다 — 자동 규칙은 이 판단을 못 한다.
- **서사 내용(구성) 자체가 학습에 영향을 주는 것으로 보인다(결과 2), 그런데 그 방향이 직관과 다르고 아직 깨끗이 검증되지 않았다.** "서사 유형을 늘리면 데이터가 더 다양해져서 좋다"는 암묵적 전제가 이 계획 전체에 깔려 있었는데, 오늘 결과(좁은 서사 구성이 오히려 더 높은 유효 차원을 보임)는 적어도 지금 실험 설계로는 그 전제가 자명하지 않다는 걸 보여준다 — 다만 위에서 밝힌 대로 행 수/epoch 교란변수 때문에 "서사 유형을 늘리는 것 자체가 나쁘다"고 결론 내릴 수도 없다. 사람이 서사-원시데이터 관계를 직접 보면 최소한 "이 서사가 원시 데이터의 어떤 부분을 실제로 잘 설명하는가"를 판단할 수 있지만, 그 판단이 SAE가 측정하는 "표현의 유효 차원"과 어떻게 연결되는지는 이번 실험으로는 아직 모른다 — 사람이 보기에 "좋은" 서사 매칭이 반드시 SAE 관점에서 "다양성을 늘리는" 매칭인지는 별도로 검증해야 하는 질문이다.
- **하지만 사람이 검토해도 넘을 수 없는 상한이 있다.** 이 세션 초반에 확인한 대로 원시 데이터 자체의 선형 유효 차원은 384차원 중 29차원뿐이다(55농장×14일, 물리적으로 강하게 얽힌 센서값). 서사 유형을 아무리 늘리고 사람이 아무리 정교하게 매칭을 골라도, 원시 데이터에 없는 정보를 만들어낼 수는 없다 — 사람의 검토가 할 수 있는 일은 **이미 있는 제한된 정보를 더 효율적으로(중복 없이, 더 정확하게) 학습 데이터로 포장하는 것**이지, 정보량 자체를 늘리는 게 아니다. `ui/data_grounding_curation/`이 실제로 하는 일도 이 범위 안에 있다 — §5.4 리포트가 측정하는 grounding 정확도는 "서사 서술이 실제 raw window와 얼마나 정합하는가"이지 "그 raw window가 얼마나 정보가 풍부한가"가 아니다.

**요약**: 서사 유형이 늘어날수록 (1) 자동 매칭만으로는 중복 문제가 구조적으로 악화되므로 사람의 검토(또는 그에 준하는 명시적 중복 제거 규칙)가 점점 더 필요해지고, (2) 서사 구성 자체가 학습 결과에 영향을 준다는 것 자체는 오늘 실측으로 뒷받침되지만 그 방향은 아직 깨끗하게 분리되지 않았으며, (3) 그럼에도 사람의 검토든 자동 규칙이든 원시 데이터의 물리적 정보량(~29차원) 자체를 늘려주지는 못한다 — 할 수 있는 건 그 안에서 더 잘 고르는 것뿐이다.

## 구현한 것

- [`schemas.py`](schemas.py) — `FeatureState`(5단계 상태 머신, 한 단계씩만 승격 가능하도록 `assert_valid_promotion`이 강제) + `CausalGrade`(E0~E5) + `assert_causal_claim_allowed`(**E3 미만이면 실행 시점에 예외** — "인과적"이라는 주장을 코드 레벨에서 막는다, 조용한 문서 문구가 아니다) + `SAEFeatureRef`(`INTERVENTION_SUPPORTED_FEATURE` 상태인데 `causal_grade < E3`이면 생성 자체가 막힘).
- [`model.py`](model.py) — `SparseAutoencoder`(표준 SAE 구성: pre-encoder bias 차감, encoder+decoder, L1 또는 Top-K sparsity, tied/untied weight 선택 가능) + `sae_loss`. 실제 torch forward/backward로 검증(합성 데이터에서 30 step 학습 시 reconstruction loss가 실제로 줄어드는 것까지 테스트).
- [`evaluation.py`](evaluation.py) — §9.4 6영역 중 라벨 없이 계산 가능한 두 영역만: `reconstruction_metrics`(MSE, explained variance), `sparsity_metrics`(평균 L0, activation frequency, dead feature ratio). `downstream_fidelity`는 `../representation_tracking/attribution.py`의 modality ablation과 같은 패턴 — 이미 계산된 두 출력(원본 activation 기반 vs SAE 재구성 기반)의 차이만 계산, 실제 downstream 모델 순전파는 호출자 책임.
- [`training.py`](training.py) — `resample_dead_features`: activation frequency가 threshold 이하인 feature의 encoder/decoder 가중치만 골라 재초기화(다른 feature는 안 건드림, 테스트로 확인). 표준 기법 그대로 씀 — "재구성 오차가 큰 방향으로 재초기화"하는 더 정교한 변형은 안 함(아래 "아직 없는 것").
- [`scripts/online2_v2/pilot_sae_stage_a_activations.py`](../../../../../../scripts/online2_v2/pilot_sae_stage_a_activations.py) — dense(CF1S) 표본 pilot. 재현 가능.
- [`scripts/online2_v2/pilot_sae_narrative_selected_activations.py`](../../../../../../scripts/online2_v2/pilot_sae_narrative_selected_activations.py) — narrative-template 타깃 표본 pilot. `training_events_v2.parquet`(1,500만 행)에서 시퀀스별 타깃 이벤트를 스트리밍 집계(전체를 메모리에 안 올림, 청크 단위로 처리), 80개 템플릿당 균등 샘플링한 뒤 `cache_stage_a_event_embeddings.py`의 인코딩 함수를 그대로 import해 재사용한다(활성화 계산 로직 중복 없음).
- [`scripts/online2_v2/pilot_sae_layer_sweep.py`](../../../../../../scripts/online2_v2/pilot_sae_layer_sweep.py) — 6개 encoder layer 전부의 pooled activation을 한 번의 순전파로 동시에 캡처해 비교(`Transformer.forward_finetuning`의 embedding+layer 루프를 그대로 복제, 원본 모델 클래스는 안 건드림). 위 narrative-selected 스크립트의 타깃 선택 함수를 그대로 import해 재사용.
- [`scripts/online2_v2/pilot_sae_layer_decoder_random_comparison.py`](../../../../../../scripts/online2_v2/pilot_sae_layer_decoder_random_comparison.py) — 학습된 인코더 vs 랜덤 초기화 인코더(`load_untrained_encoder`) + MLM/SOP 디코더 변환까지 8개 지점 비교.
- [`scripts/online2_v2/check_token_bag_diversity.py`](../../../../../../scripts/online2_v2/check_token_bag_diversity.py) — bag-of-tokens 대조 실험(학습 파라미터 없이 토큰 co-occurrence만으로 만든 벡터의 PCA 유효 차원).
- [`scripts/online2_v2/build_pretrain_narrative_ablation_subsets.py`](../../../../../../scripts/online2_v2/build_pretrain_narrative_ablation_subsets.py) — `training_events_v2.parquet`를 `narrative_id`로 스트리밍 필터링해 narrow_subset/single_table_only/multi_table_only/dedup_reduced 4종 부분 코퍼스 생성(원시 빌드 재실행 없음, 필터링만).
- [`scripts/online2_v2/run_narrative_ablation_pretrain_sweep.sh`](../../../../../../scripts/online2_v2/run_narrative_ablation_pretrain_sweep.sh) — SOP 재조정 + 서사 다양성 6-arm 재학습 드라이버. 이미 끝난 arm은 건너뛰고 중단된 arm은 `last.ckpt`에서 자동 resume하는 멱등 스크립트(세션 중단에도 안전하게 재실행 가능하도록 설계, 실제로 한 번 중단됐다가 이 덕분에 재개함).
- [`scripts/online2_v2/compare_narrative_ablation_checkpoints.py`](../../../../../../scripts/online2_v2/compare_narrative_ablation_checkpoints.py) — 재학습 체크포인트(6-arm + microevent_split)의 layer5/mlm_transform/sop_transform PCA 유효 차원 + SAE dead_feature_ratio 비교.
- [`scripts/online2_v2/build_microevent_training_corpus.py`](../../../../../../scripts/online2_v2/build_microevent_training_corpus.py) — 번들된 measurement group을 독립된 micro-event로 쪼갠 대안 학습 코퍼스 생성(원시 빌드 재실행 없음, `training_events_v2.parquet` 스트리밍 재파싱만).
- [`scripts/online2_v2/check_raw_narrative_time_location_diversity.py`](../../../../../../scripts/online2_v2/check_raw_narrative_time_location_diversity.py) — 모델 없이 raw 값(상대 bin 제거+세분화)+narrative_id+정확한 시각/장소를 조합별로 PCA 유효 차원 비교(`check_token_bag_diversity.py`와 같은 계열의 무학습 대조 실험).
- [`scripts/generate_online2_narrative_catalog_auto_expansion.py`](../../../../../../scripts/generate_online2_narrative_catalog_auto_expansion.py) — 88개 카탈로그를 넘어 feature×matcher×파라미터 변형을 조합적으로 생성하고 실데이터로 dry-run 검증하는 확장 생성기(아직 실제 카탈로그 미반영, 검토용). 733개(기존 88 + 검증 통과 645)까지 확장, 결과는 `outputs/online2/sae_pilot/narrative_auto_expansion_report.json`.
- [`ui/narrative_evidence_explorer/`](../ui/narrative_evidence_explorer/) — 서사(사람의 해석) ↔ 실제 매칭된 시퀀스(raw 데이터+토큰)를 나란히 보여주는 별도 Streamlit 앱(자체 디렉토리+README, `pipeline_explorer`와 별개).
- [`scripts/generate_online2_narrative_catalog.py`](../../../../../../scripts/generate_online2_narrative_catalog.py) — 서사 카탈로그 생성기에 8개 신규 spec 추가(`D16`/`D17`/`S13`/`S14`/`X12`/`X13`/`X14`/`X15`, 80→88개).
- [`src/online2/materializers.py`](../../../../../../src/online2/materializers.py) — `_crossfarm_by_hour`/`_crossfarm_by_survey_index`(농장 간 절대시각 대신 지역시각·조사순번 기준 비교) / `_image_longitudinal_pair`(농장별 최초·최근 이미지 페어 + 환경 맥락) 3개 매처 추가.
- [`scripts/online2_v2/export_training_events_v2_event_grain.py`](../../../../../../scripts/online2_v2/export_training_events_v2_event_grain.py) — control이 실제로 학습한 이벤트 단위 `training_events_v2.parquet`를 만드는 진짜 경로(`build_v2.py`의 `export_training()`은 placeholder, 쓰면 안 됨). Phase 2의 `NARRATIVE`/`FARM_LOCAL`/`ZONE_LOCAL` 토큰 주입을 여기 `expand_sequence_rows()`에 구현.

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

§18 "SAE feature 대부분이 dead 또는 불안정" — **실측으로 발동, 표본 구성 9종 비교 + 6-arm 재학습 + micro-event 분리 재학습 + 서사 88개 확장/상대 bin 제거/narrative·farm·zone 토큰 추가(Phase 2) 재학습 이후에도 해소 안 됨**(위 pilot 결과 표 참조). 근본 원인을 특정했다: (1) 이 데이터셋(55농장×14일, 원시 이벤트 222,309건) 자체의 낮은 유효 차원(PCA로 384차원 중 29개가 99% 분산 설명 — **원인 특정 완료, 표본을 dense/narrative-selected 어느 쪽으로 구성해도 동일함을 직접 검증**), (2) dict_size를 그 차원(29)에 맞춰도 남는 TopK "승자독식" 학습 역학(dict=32~64에서도 dead ratio가 0.40~0.50 바닥). (1)은 "더 넓거나 더 잘 고른 표본"으로 해결되는 문제가 아님이 확인됐지만, 개선 방향을 몇 개 더 실측으로 찾았다: **6개 encoder layer를 비교한 결과 최종 layer(지금까지 모든 pilot이 쓴 것) 대신 layer 2를 쓰면 dead ratio가 0.500→0.391로 개선**(seed 1개 결과, 재현성 확인 전), **SOP 비율을 더 균형 있게(0.2/0.2→0.33/0.33) 재학습하면 dead ratio가 0.734→0.391로 개선**(control vs sop_balanced 순수 대조 실험, 재현됨, 지금까지 유일하게 세 지점 모두에서 일관되게 개선). event 경계를 좁힌 micro-event 재학습과 서사 확장+상대 bin 제거+narrative/farm/zone 토큰 추가(Phase 2) 재학습은 둘 다 **layer5는 뚜렷이 개선**(dead 0.734→0.625, 0.734→0.547)시켰지만 **mlm_transform 또는 sop_transform 중 하나는 오히려 악화**시켰다 — micro-event는 sop_transform이 악화(dead 0.781→0.875, EV 0.920→0.319), Phase 2는 mlm_transform이 악화(rank99 19→15, dead 0.500→0.625)됐고 sop_transform만 소폭 개선(EV 0.920→0.948, dead 0.781→0.750). 즉 **지금까지 시도한 개입 중 SOP 재조정만 세 지점 모두에서 일관되게 좋아졌고, 나머지(micro-event 분리, 서사 확장+토큰화 개편)는 전부 "한 지점은 좋아지고 다른 지점은 나빠지는" 트레이드오프였다** — 아직 어느 것도 §18을 깨끗이 해소하지 못했다. §18 해소를 위한 다음 시도는 (a) layer 2/4 기준 dict_size·sparsity 재sweep, (b) SOP 0.33/0.33 이상으로 재학습한 체크포인트 기준 재sweep, (c) 서사 다양성의 실제 효과를 확인하려면 narrative_id가 아니라 epoch 수를 통제 변수로 고정한 후속 재학습, (d) `masking.py`의 group-내부 per-token 독립 마스킹을 group 전체 단일 치환으로 바꾼 뒤 micro-event 분리·Phase 2 토큰화와 함께 재실험, (e) SOP 재조정과 Phase 2 토큰화를 함께 적용한 재학습(둘 다 layer5는 개선시켰으므로 조합 효과 확인 가치 있음) — 이 순서로 시작할 것.
