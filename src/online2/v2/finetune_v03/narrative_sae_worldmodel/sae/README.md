# sae

**근거**: 계획서 §9 (SAE 기반 기계적 해석)
**Phase**: Phase 4 · **범위**: 현재 필수 (§3.1, "선택 layer 대상 SAE pilot"만) · **상태**: 미착수

## 목적

사전학습/미세조정 encoder의 dense activation에서 희소 특징을 추출하고, 의미적 순도·기능적 개입 효과가 검증된 특징만 해석 단위로 승격한다. "SAE latent = monosemantic feature"를 자동으로 확정하지 않는다.

## 상태 머신 (§9.1)

```
SAE_LATENT → CANDIDATE_SEMANTIC_FEATURE → EVALUATED_SEMANTIC_FEATURE
           → INTERVENTION_SUPPORTED_FEATURE → CIRCUIT_SUPPORTED_FEATURE
```

CF1S의 `classify_case_disposition()`과 동일한 설계 원칙을 따른다: 상태는 저장된 라벨이 아니라 매번 재계산한다(자기선언 금지).

## 구현 계획

1. `schemas.py` — `SAEFeatureRef`(SAE version, latent ID, activation, validation state).
2. `train_sae.py` — 3개 학습 지점(사전학습 encoder 중간층 / 사전학습·미세조정 encoder 최종층 / regression·dynamics head 입력 직전) × 4개 버전(`SAE_PRETRAIN`/`SAE_FINETUNE`/`SAE_WORLD`/`SAE_POLICY`).
3. `sweep_config.py` — dictionary expansion factor, L1/Top-K/BatchTopK sparsity, learning rate, activation normalization, dead feature resampling, target layer/position pooling.
4. `evaluation/` — 6개 평가 영역별 파일: `reconstruction.py`(reconstruction error, explained variance, downstream fidelity), `sparsity.py`(평균 L0, activation frequency, dead feature ratio), `semantic.py`(concept precision/recall, positive/negative consistency), `decomposition.py`(splitting/absorption/duplicate/composition), `stability.py`(farm·생육단계·계절·checkpoint 간), `functional.py`(ablation/steering/activation patching/prediction delta).
5. `concept_mapping.py` — SAE feature ↔ ontology concept N:M 매핑(1:1/1:N/N:1/계층/contextual relation), 1:1 강제하지 않음.
6. `feature_alignment.py` — 사전학습→미세조정 특징 상태 분류: `PRESERVED|SPLIT|MERGED|SUPPRESSED|NEWLY_EMERGED|UNMATCHED`. activation correlation, decoder similarity, 사례 중첩, concept score, intervention effect 기반.
7. `causal_grades.py` — 증거 등급 E0_OBSERVED ~ E5_CIRCUIT. **E3 미만은 인과적 특징이라고 부르지 않는 가드를 코드 레벨에서 강제**(등급 미달 feature에 대한 인과 주장 API 호출을 막음). SAE Top-K 활성값만으로 circuit tracing을 주장하지 않는다.

## 의존성

- 기존: 없음 (신규)
- 신규: `../representation_tracking/`(activation 소스)

## Acceptance 연결

D1–D6 (계획서 §17-D)

## 중단 조건 연결

SAE downstream fidelity 부족, feature 대부분 dead/불안정 시 다음 Phase(concept_governance, world_model) 진행 금지 (§18)
