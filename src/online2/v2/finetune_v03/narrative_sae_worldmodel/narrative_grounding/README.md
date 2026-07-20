# narrative_grounding

**근거**: 계획서 §5 (시계열–서사 양방향 Grounding)
**Phase**: Phase 1 · **범위**: 현재 필수 (§3.1) · **상태**: 미착수

## 목적

원시 시계열/이미지/Cube 관측과 자연어 서사를 양방향으로 연결한다. 서사→시계열은 서사 문장에서 근거 window를 찾고, 시계열→서사는 window에서 구조화된 서사를 생성한다. 이 결과가 §7 `multimodal_pretrain`의 time-text contrastive pair 소스가 된다.

## 착수 전 확인

`/data/datasets/agrichallenge/online2/narratives/`에 이미 `baseline_inventory_v8.json`(604KB), `normalized_catalog_v8.csv`(154KB), `online2_pretraining_master_prompt_v8.md`(38KB, 최근 수정)가 존재한다. 이 모듈을 처음부터 새로 설계하기 전에 v8 산출물이 `DataWindow`/`Narrative` 스키마 요구사항을 얼마나 충족하는지, 그대로 흡수할 수 있는지 먼저 확인한다.

## 데이터 계약 (계획서 §4.1)

- `DataWindow`: 시작·종료, 포함 point, 집계·결측·변화점 정보 — 6개 단위(변화점/임계이탈 지속/생육단계별 상태/다변량 동시변화/회복-악화-안정/이미지-Cube 연결 관측)로 정의 (§5.2)
- `Narrative`: 관측·파생 사실·해석·인과 상태·추천 상태·근거 window (§5.2 JSON 구조 참조)

## 구현 계획

1. `schemas.py` — `DataWindow`, `Narrative` 데이터클래스. `Narrative`는 `observation`/`derived_state`/`interpretation`/`causal_status`(기본 `NOT_ESTABLISHED`)/`recommendation_status`(기본 `NOT_GENERATED`)/`supporting_windows`/`contradicting_windows`/`confidence`/`sequence_recommendation`(`INCLUDE|EXCLUDE|REVIEW`) 필드.
2. `text_to_window.py` (서사→시계열) — 서사를 관측/파생/해석/인과/추천 문장으로 분해 → feature·생육단계·farm/zone·시간조건 구조화 → 원시값/변화점/지속구간/다변량 동시변화 window 검색 → ontology 규칙 + Time-Text 임베딩 + SAE feature(가용 시) + 전문가 mapping 결합 랭킹 → `AUTO_ACCEPT_CANDIDATE|REVIEW|QUARANTINE|REJECT` 분기.
3. `window_to_text.py` (시계열→서사) — 6개 window 단위별 서사 생성기, 위 JSON 구조로 출력.
4. `review_queue.py` — 사람 검토 우선순위 규칙 7종 구현 (§5.3): 모델·규칙·전문가 불일치 / 신규·저빈도 개념 / 인과·추천 포함 / 근거·반례 동시 강함 / 예측 영향도 큼 / concept split·merge 후보 / holdout 유사·권한 불명확.
5. `evaluation.py` — Narrative→Window Recall@K, Window→Narrative Precision@K, temporal IoU, feature-set overlap, farm/zone scope accuracy, cycle consistency, unsupported claim rate, expert acceptance rate, 자동 승인 오류율/검토율 (§5.4).

## 순환 의존성 주의

랭킹 단계(2번)는 SAE feature를 입력으로 쓸 수 있다고 되어 있으나 SAE(`../sae/`)는 사전학습 이후에나 생성된다. 초기 버전은 SAE feature 없이 ontology 규칙 + Time-Text 임베딩만으로 동작하도록 만들고, SAE pilot(Phase 4) 완료 후 랭킹 신호를 추가하는 2단계 구현으로 간다.

## 의존성

- 기존: 없음 (신규). Time-Text 임베딩은 이 모듈 내부에서 별도 학습하거나 `multimodal_pretrain/` 사전학습 중간 산출물을 재사용.
- 하위 소비자: `sequence_curation/`(REVIEW 판정 입력), `multimodal_pretrain/`(pair 소스)

## Acceptance 연결

B1–B4 (계획서 §17-B)

## 중단 조건 연결

Grounding unsupported claim 허용치 초과 시 다음 Phase 진행 금지 (§18)
