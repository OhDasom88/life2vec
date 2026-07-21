# narrative_sae_worldmodel

`스마트농업 시계열–서사–SAE–월드모델 통합 수행계획서` (원본 V1.0, life2vec 범위판 V1.1)의 신규 구현 서브트리.

- 원본 계획서: `/data/datasets/agrichallenge/스마트농업_시계열_서사_SAE_월드모델_통합_수행계획서_20260720.md`
- life2vec 범위판: [reports/스마트농업_시계열_서사_SAE_월드모델_통합_수행계획서_20260720_life2vec범위.md](../../../../../../reports/스마트농업_시계열_서사_SAE_월드모델_통합_수행계획서_20260720_life2vec범위.md)
- 기본 원칙: `versioned`, `reproducible`, `fail-closed`, `holdout-blind` (계획서 전체에 적용)

기존 `counterfactual/` 트리(CF1S 등)와 이름이 겹치지 않도록 이 서브트리 하나로 신규 산출물을 격리했다. 이유: `counterfactual/grounding/`이 이미 CF1S 편집용 raw-target grounding(다른 의미)으로 쓰이고 있어, §5 서사↔시계열 grounding을 같은 이름으로 두면 혼동된다.

## 디렉토리 맵

| 디렉토리 | 계획서 절 | Phase | 상태 |
|---|---|---|---|
| [narrative_grounding/](narrative_grounding/README.md) | §5 | Phase 1 | §5.1/§5.2/§5.3/§5.4 구현 + 실측 재보정 + 반례 탐지 + [ui/data_grounding_curation/](ui/data_grounding_curation/README.md) 3탭(검색→검토 큐→리포트)이 하나의 루프로 연결됨(66개 테스트 통과). 남은 건 `auto_accept_error_and_review_rate`의 전체 모집단 로깅(부분 연결) |
| [sequence_curation/](sequence_curation/README.md) | §6.2/§6.3 | Phase 2 | 구현 완료(절단 알고리즘·편집 트랜잭션·§5.3 연결, 실행 배치는 아직) |
| [multimodal_pretrain/](multimodal_pretrain/README.md) | §7 | Phase 2 | MLM/SOP는 `src/tasks/grouped_mlm.py`+`run_v2_pretrain_loop.py`에 이미 구현·학습 중임을 확인(재구현 안 함) — 진짜 신규였던 time reconstruction/time-text contrastive/cross-modal matching 3개 손실 + 결합 로직 + pair 품질 등급 구현 완료(33개 테스트). 실제 학습 루프 연결은 아직 |
| [representation_tracking/](representation_tracking/README.md) | §8.3 | Phase 3 | 구현 완료 — activation 캡처·event/token attribution은 이미 존재함을 확인(재사용), 없던 ActivationRef 타입 + modality ablation만 신규 구현(22개 테스트) |
| [sae/](sae/README.md) | §9 | Phase 4 | 핵심 구현 완료(상태 머신·모델·평가·dead feature resampling, 34개 테스트) + 표본 구성 9종 + encoder 6개 layer 비교 + 랜덤 초기화·MLM/SOP 디코더 헤드 비교 실측 — **근본 원인 특정**(데이터셋 자체의 PCA 유효 차원이 384 중 29뿐, 표본 선별 방식과 무관) **+ 개선 방향 발견**(최종 layer 대신 layer 2 사용 시 dead ratio 0.500→0.391, 재현성 확인 전) **+ 압축이 학습으로 생긴 것임을 확인**(랜덤 초기화는 ReZero 항등식으로 6개 layer가 전부 동일하고 유효 차원도 훨씬 높음(118) — MLM/SOP 학습이 15~35차원까지 3~7배 압축) → §18 중단 조건 여전히 발동하나 다음 시도 경로가 명확해짐 |
| [representation_explorer/](representation_explorer/README.md) | §10 | Phase 4 | 구현 완료 — 진짜 신규(기존 코드는 1회성 노트북뿐)였음을 확인, 실제 scikit-learn/umap-learn으로 PCA/UMAP+trustworthiness+split overlap 구현(29개 테스트) |
| [concept_governance/](concept_governance/README.md) | §11 | Phase 5 | 미착수 (기존 `cf1s/core_locks.py` 원칙 확장) |
| [world_model/](world_model/README.md) | §12 | Phase 6 | 미착수 — §14 실측 리스크 있음 |
| [edit_policy_rl/](edit_policy_rl/README.md) | §13 | Phase 7 | 미착수 — 1~2단계는 기존 CF1S 재사용, 3단계부터 신규 |
| [ui/](ui/README.md) | §15 | Phase 1~7 병행 | 미착수, 8개 서브앱 |

## 실행 순서 (계획서 §21 고정 우선순위)

```
계보와 데이터 계약 (Phase 0, 기존 cf1s/core_locks.py·core_canonical.py 확장)
→ narrative_grounding (Phase 1)
→ sequence_curation + multimodal_pretrain (Phase 2)
→ representation_tracking (Phase 3, 미세조정과 표현 추적)
→ sae + representation_explorer (Phase 4)
→ concept_governance (Phase 5)
→ world_model (Phase 6)
→ edit_policy_rl (Phase 7, 조건부 offline RL)
→ 회로 분석 (E5_CIRCUIT, sae/ 내부 후속 단계)
```

각 디렉토리는 다음 단계 착수 전 계획서 §17 Acceptance Criteria와 §18 중단·보류 조건을 통과해야 한다. 통과하지 못하면 다음 Phase로 자동 진행하지 않는다 — 이는 실패 은폐가 아니라 정상적인 readiness 결과로 기록한다(§18).

## 착수 전 공통 전제 (Phase 0에서 먼저 확인)

1. **기존 산출물 중복 점검 — 확인 완료, 실제로 겹쳤다**: `narrative_grounding/` 착수 전 `narratives/v8` 산출물을 확인한 결과, §5.2(시계열→서사)와 §6(이벤트·토큰·시퀀스)의 핵심 로직은 `src/online2/`(`materializers.py`, `builder.py`, `catalog.py`)에 이미 구현되어 있었고, `outputs/online2/build-v8-active80-r3/`에 967,012개 sequence가 실제로 빌드까지 끝나 있었다. `narrative_grounding/`은 이를 재사용하는 어댑터로 범위를 좁혀 구현했다 — 자세한 내용은 [narrative_grounding/README.md](narrative_grounding/README.md) 참조.
2. **데이터 규모 제약**: 원시 데이터 전체 27MB, 학습 케이스 35건 + holdout 20건. `multimodal_pretrain/`(5-loss 멀티모달 사전학습)과 `sae/`(dictionary learning)는 이 규모에서 통계적 유효성이 낮을 수 있으므로 착수 전 최소 데이터 요건을 별도 검토한다.
3. **신호 부재 실측**: CF1S 55건 검증에서 44건이 `CONSTRUCTIBLE_SELECTED`까지 도달했으나 전부 `CONTROL_WITHIN_LOCKED_THRESHOLD`로 판정 — 현재 2-event 편집 범위에서 임계값을 넘는 인과효과가 관측된 케이스가 0건이다. `world_model/`, `edit_policy_rl/` 착수 여부는 이 실측을 반영해 §12.3 승격 기준으로 게이트한다.
4. **vocab 변경 비용**: `concept_governance/`에서 vocab 변경을 승인하면 CF1S 기존 55건(Dev3 3 + Primary32 32 + Validation20 20) stable lock 전체가 무효화되어 재인증이 필요하다. 승인 전 재인증 소요시간을 명시적으로 보고한다.

## 기존 코드와의 관계

이 서브트리는 CF1S(`counterfactual/cf1s/`)를 대체하지 않는다. `disposition_profiles.py`, `core_verifier.py`(G1–G12), `core_locks.py`, `core_raw_transaction.py`는 그대로 재사용하며, 여기서는 CF1S에 없는 신규 서브시스템(서사 grounding, SAE, 월드모델, RL 정책)만 구현한다. CF1S 내부 파일을 직접 수정할 때는 이 트리가 아니라 `counterfactual/cf1s/`에서 확장한다.
