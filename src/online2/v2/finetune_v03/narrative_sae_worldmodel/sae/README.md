# sae

**근거**: 계획서 §9 (SAE 기반 기계적 해석)
**Phase**: Phase 4 · **범위**: 현재 필수 (§3.1, "선택 layer 대상 SAE pilot"만) · **상태**: 핵심 구현 완료(상태 머신·모델·평가·dead feature resampling) + 표본 구성 9종 + 6개 encoder layer 비교 + 랜덤 초기화·MLM/SOP 디코더 헤드 비교 실측, 34개 테스트 통과. **원인 진단 완료(데이터의 유효 차원이 근본적으로 낮음, 표본 구성과 무관) + 압축이 학습으로 생긴 것임을 랜덤 초기화 대조로 확인 + 개선 방향 하나 발견(최종 layer 대신 layer 2/4), §18 중단 조건은 계속 발동 — 아래 참조**

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

**반면 학습이 118→15~35로 만드는 추가 압축(위 "이 압축이 학습으로 생긴 것인지" 절)은 별개 문제이고, 이쪽은 task 설계와 관련이 있어 보인다 — 실제로 구체적인 근거가 있다.** 실행에 쓰인 실제 설정(`conf/task/online2_v2_grouped_mlm.yaml`)을 확인하면:

- **SOP 라벨 분포가 90/5/5로 심하게 불균형하다**(`sop_reverse_probability: 0.05`, `sop_shuffle_probability: 0.05`, `src/tasks/mlm.py:38-39` 기본값과 동일하게 설정됨). "정상 순서"만 90%라 모델이 거의 항상 label=0을 찍기만 해도 90% 정확도가 나온다 — CLS 표현을 세밀하게 구분할 유인이 약하다. 실측과도 정확히 들어맞는다: 학습된 sop_transform은 rank99=15, explained_variance=0.973으로 8개 지점 중 가장 "뾰족한"(=한 방향으로 몰린) 분포다. **이건 검증된 사실이지 추측이 아니다 — SOP를 더 균형 잡힌 비율(예: reverse/shuffle을 각각 0.15~0.2 근처로)로 재설정하면 CLS 표현의 유효 차원이 올라가는지는 재학습으로 직접 검증 가능한 다음 실험이다.**
- **mask_ratio=0.30에 measurement-group 단위 masking**(`GroupedMLMMasker`, `src/online2/v2/masking.py`)이 걸려 있다 — 개별 토큰이 아니라 한 measurement group 전체를 30% 확률로 통째로 가린다. 이미 알고 있듯 이 데이터는 변수 간 상관이 매우 높다(연속 이벤트 코사인 유사도 평균 0.89, raw PCA 유효 차원 29). 상관이 이렇게 높으면 가려진 group도 남은 group들로부터 거의 선형 보간하듯 복원 가능해서, MLM이 "29차원짜리 충분통계량만 잘 압축해서 들고 있으면 풀리는" 얕은 지름길이 될 수 있다 — 이건 SOP 건과 달리 **현재 설정과 결과를 잇는 그럴듯한 메커니즘이지 확정된 인과관계는 아니다.** 검증하려면 실제 재학습(예: mask_ratio를 낮추거나, "쉽게 보간되는" group과 "안 되는" group에 다른 masking 확률을 주는 방식)이 필요하고, 지금까지의 실험(전부 기존 체크포인트 재사용)과 달리 **GPU-시간 단위의 새 사전학습 실행**이라는 뚜렷이 더 큰 비용이 든다.

**요약**: (1) 초기 layer가 100%가 안 되는 건 데이터의 물리적 상한 때문이며 task 재설계로 해결할 문제가 아니다. (2) 학습이 만드는 추가 압축(118→15~35)에 대해서는 SOP 클래스 불균형이 구체적 근거가 있는 유력 후보이고, masking 방식은 그럴듯하지만 미검증인 후보다 — 둘 다 실제로 검증하려면 재학습이 필요하므로, 착수 전 사용자 확인이 필요하다.

## 구현한 것

- [`schemas.py`](schemas.py) — `FeatureState`(5단계 상태 머신, 한 단계씩만 승격 가능하도록 `assert_valid_promotion`이 강제) + `CausalGrade`(E0~E5) + `assert_causal_claim_allowed`(**E3 미만이면 실행 시점에 예외** — "인과적"이라는 주장을 코드 레벨에서 막는다, 조용한 문서 문구가 아니다) + `SAEFeatureRef`(`INTERVENTION_SUPPORTED_FEATURE` 상태인데 `causal_grade < E3`이면 생성 자체가 막힘).
- [`model.py`](model.py) — `SparseAutoencoder`(표준 SAE 구성: pre-encoder bias 차감, encoder+decoder, L1 또는 Top-K sparsity, tied/untied weight 선택 가능) + `sae_loss`. 실제 torch forward/backward로 검증(합성 데이터에서 30 step 학습 시 reconstruction loss가 실제로 줄어드는 것까지 테스트).
- [`evaluation.py`](evaluation.py) — §9.4 6영역 중 라벨 없이 계산 가능한 두 영역만: `reconstruction_metrics`(MSE, explained variance), `sparsity_metrics`(평균 L0, activation frequency, dead feature ratio). `downstream_fidelity`는 `../representation_tracking/attribution.py`의 modality ablation과 같은 패턴 — 이미 계산된 두 출력(원본 activation 기반 vs SAE 재구성 기반)의 차이만 계산, 실제 downstream 모델 순전파는 호출자 책임.
- [`training.py`](training.py) — `resample_dead_features`: activation frequency가 threshold 이하인 feature의 encoder/decoder 가중치만 골라 재초기화(다른 feature는 안 건드림, 테스트로 확인). 표준 기법 그대로 씀 — "재구성 오차가 큰 방향으로 재초기화"하는 더 정교한 변형은 안 함(아래 "아직 없는 것").
- [`scripts/online2_v2/pilot_sae_stage_a_activations.py`](../../../../../../scripts/online2_v2/pilot_sae_stage_a_activations.py) — dense(CF1S) 표본 pilot. 재현 가능.
- [`scripts/online2_v2/pilot_sae_narrative_selected_activations.py`](../../../../../../scripts/online2_v2/pilot_sae_narrative_selected_activations.py) — narrative-template 타깃 표본 pilot. `training_events_v2.parquet`(1,500만 행)에서 시퀀스별 타깃 이벤트를 스트리밍 집계(전체를 메모리에 안 올림, 청크 단위로 처리), 80개 템플릿당 균등 샘플링한 뒤 `cache_stage_a_event_embeddings.py`의 인코딩 함수를 그대로 import해 재사용한다(활성화 계산 로직 중복 없음).
- [`scripts/online2_v2/pilot_sae_layer_sweep.py`](../../../../../../scripts/online2_v2/pilot_sae_layer_sweep.py) — 6개 encoder layer 전부의 pooled activation을 한 번의 순전파로 동시에 캡처해 비교(`Transformer.forward_finetuning`의 embedding+layer 루프를 그대로 복제, 원본 모델 클래스는 안 건드림). 위 narrative-selected 스크립트의 타깃 선택 함수를 그대로 import해 재사용.

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

§18 "SAE feature 대부분이 dead 또는 불안정" — **실측으로 발동, 표본 구성 9종 비교 이후에도 해소 안 됨**(위 pilot 결과 표 참조). 근본 원인을 특정했다: (1) 이 데이터셋(55농장×14일, 원시 이벤트 222,309건) 자체의 낮은 유효 차원(PCA로 384차원 중 29개가 99% 분산 설명 — **원인 특정 완료, 표본을 dense/narrative-selected 어느 쪽으로 구성해도 동일함을 직접 검증**), (2) dict_size를 그 차원(29)에 맞춰도 남는 TopK "승자독식" 학습 역학(dict=32~64에서도 dead ratio가 0.40~0.50 바닥). (1)은 "더 넓거나 더 잘 고른 표본"으로 해결되는 문제가 아님이 확인됐지만, **6개 encoder layer를 비교한 결과 최종 layer(지금까지 모든 pilot이 쓴 것) 대신 layer 2를 쓰면 dead ratio가 0.500→0.391로 개선됐다**(seed 1개 결과, 재현성 확인 전) — §18 해소를 위한 다음 시도는 layer 2/4 기준으로 dict_size·sparsity를 재sweep하는 것부터 시작할 것.
