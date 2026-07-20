# multimodal_pretrain

**근거**: 계획서 §7 (사전학습)
**Phase**: Phase 2 · **범위**: 현재 필수 (§3.1) · **상태**: 미착수 (기존 `pipeline_m2.py`/`stage_a_reencoder.py` 확장)

## 목적

정형 시계열 이벤트 토큰·서사·이미지·hyperspectral Cube를 입력으로 받는 멀티목적 사전학습. 기존 `counterfactual/pipeline_m2.py`, `counterfactual/stage_a_reencoder.py`가 처리하는 Stage-A 사전학습을 확장 대상으로 삼는다 — 이 디렉토리는 새 손실 모듈과 sweep 설정만 담고, 학습 루프 자체는 기존 파이프라인 파일을 직접 확장한다(중복 파이프라인을 새로 만들지 않는다).

## 학습 목적 (§7.2)

```
L_total = λ1·L_MLM + λ2·L_SOP + λ3·L_time + λ4·L_time-text + λ5·L_cross-modal
```

| 손실 | 상태 | 비고 |
|---|---|---|
| `MLM` | 확장 | 기존 event/token 문맥 복원 objective 확장 |
| `SOP`/temporal order | 확장 | 기존 `abspos_reference` 시간 정보 확장 |
| `time reconstruction` | 신규 | |
| `time-text contrastive` | 신규 | `../narrative_grounding/` pair 필요 |
| `cross-modal matching` | 신규 | Online1 문서의 Cube C1/C2 arm과 연계 |

## 구현 계획

1. `losses/` 하위에 5개 손실 모듈 각각 파일 분리, 기존 `finetune_v03/losses.py`와 이름 충돌 없도록 `pretrain_*` 접두어 사용.
2. `input_modality.py` — modality missingness mask를 포함한 입력 스키마 (시계열/서사 embedding·narrative token/이미지 embedding/Cube embedding/시간·공간·생육단계).
3. `contrastive_pairs.py` — pair 품질 6단계 라벨(`GOLD_EXPERT`/`SILVER_RULE_GROUNDED`/`SILVER_MODEL_GROUNDED`/`WEAK_TEMPORAL_MATCH`/`UNVERIFIED`/`CONTRADICTED`), multi-positive 허용, hard negative 구성(생육단계·시간방향·지속시간·farm/zone 유사), shortcut 검사(farm ID/날짜/modality 존재만으로 정답 판별 가능한지 ablation).
4. `sweep_config.py` — 손실 가중치와 손실 적용 여부(off 포함) 둘 다 W&B sweep 대상으로 노출.
5. `wandb_logging.py` — §7.4 체크리스트(dataset/vocab/ontology/narrative registry hash, 구조·checkpoint, loss별/modality별 지표, retrieval/reconstruction 지표, gradient·activation norm, dead modality, resume parity, sweep selection rule, development-only selection 여부).

## 의존성

- 기존: `counterfactual/pipeline_m2.py`, `counterfactual/stage_a_reencoder.py`, `RegistryVocabulary` (vocab v2, 고정)
- 신규: `../narrative_grounding/` (time-text contrastive pair)

## 리스크

원시 데이터 전체 27MB, 케이스 35건(train)/20건(holdout) 규모. 5-loss 멀티모달 사전학습이 이 규모에서 통계적으로 유의미한지 착수 전 별도 검증 필요(§4.3 transductive 활용 범위 포함). 데이터가 부족하면 손실 일부(특히 cross-modal matching)를 sweep에서 off 상태로 두는 것도 정상 결과로 취급한다(§7.2 마지막 문장).

## Acceptance 연결

C1–C4 (계획서 §17-C)
