# sae_feature_dashboard

**근거**: 계획서 §15 · **백엔드**: `../../sae/` · **상태**: 미착수

## 기능

- feature heatmap (activation frequency, 대상 layer/position)
- 사례 뷰 (feature가 활성화된 실제 window/서사 사례)
- 품질 지표 (`sae.evaluation.*` 6개 영역: reconstruction/sparsity/semantic/decomposition/stability/functional)
- 개입 실험 트리거 (ablation/steering) 및 결과 확인

## 산출물

feature evaluation (버전 관리, `sae.schemas.SAEFeatureRef`와 1:1 연결)

## 구현 계획

1. feature 목록 + 상태 머신 단계 표시(`SAE_LATENT → ... → CIRCUIT_SUPPORTED_FEATURE`), 저장 라벨이 아니라 `sae.causal_grades`의 재계산 결과를 그대로 렌더링.
2. `sae.concept_mapping`의 N:M 매핑 뷰(1:1 강제 아님 — 여러 개념이 한 feature에, 또는 한 개념이 여러 feature에 걸릴 수 있음을 UI가 명시적으로 표현).
3. `sae.feature_alignment`의 사전학습↔미세조정 정렬 상태(`PRESERVED|SPLIT|MERGED|SUPPRESSED|NEWLY_EMERGED|UNMATCHED`) 뷰.
4. **가드**: E3(`E3_INTERVENTION`) 미만 등급 feature에는 "인과적"이라는 표현을 UI 텍스트에서 사용하지 않는다.
