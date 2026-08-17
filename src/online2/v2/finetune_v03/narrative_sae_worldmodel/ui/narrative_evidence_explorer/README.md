# narrative_evidence_explorer

**목적**: 서사(narrative)는 사람이 "이 raw 구간은 이런 패턴이다"라고 써 둔 **해석**
(카탈로그의 `purpose`/`expected_pattern`/`agronomic_interpretation`)이고, 모델이
실제로 학습에 쓰는 건 그 해석이 골라낸 raw 구간을 토큰화한 **시퀀스**뿐이다. 이 둘은
별개의 것이라 — 서사 이름이 그럴듯해도 실제로 매칭된 시퀀스가 그 이름에 맞는 패턴을
담고 있는지는 직접 봐야 안다. 이 앱은 서사 하나를 고르면 그 해석 텍스트, 실제로
매칭된 시퀀스 목록, 그리고 그중 하나의 토큰화된 SENTENCE + 원본 raw 센서 CSV 값을
한 화면에 나란히 보여줘서 이 대응관계를 사람이 눈으로 확인할 수 있게 한다.

## 실행

```bash
conda run -n life2vec streamlit run \
  src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/narrative_evidence_explorer/app.py
```

사전 준비물(이미 있어야 동작):
- `datasets/agrichallenge/online2/online2_narrative_catalog.csv` — 활성 서사 카탈로그
- `outputs/online2/v2_build_expanded88_phase2/training_events_v2.parquet` 및/또는
  `outputs/online2/v2_build/training_events_v2.parquet` — 코퍼스 선택 드롭다운에서 고를 두 코퍼스
- `datasets/agrichallenge/online2/data/{E_environment,R_rootzone,A_actuator,G_growth}/` — 원본 raw CSV
- (선택) `outputs/online2/sae_pilot/narrative_auto_expansion_report.json` — "자동생성 후보" 탭용

## 구성

- [`data_access.py`](data_access.py) — Streamlit 비의존 순수 조회 함수. `../pipeline_explorer/data_access.py`의
  `parse_token_trace`(SENTENCE→토큰 목록)/`lookup_raw_rows`(farm/zone/timestamp→원본 CSV 행)를
  그대로 import해서 재사용한다(같은 로직 두 번 안 만듦). 이 모듈이 새로 추가하는 건
  "서사 카탈로그 ↔ 실제 매칭된 시퀀스" 조회 하나뿐이다 — `load_matched_instances()`가
  `training_events_v2.parquet`을 narrative_id로 pyarrow predicate pushdown 필터링해서
  1500만 행 전체를 메모리에 안 올리고 조회한다.
- [`app.py`](app.py) — 위를 감싸는 Streamlit 렌더링 레이어. 탭 2개:
  - **활성 서사**: 코퍼스(88개 확장판 / control 원본 80개) 선택 → 서사 선택 → 해석 텍스트
    → 실제 매칭 시퀀스 목록 → 시퀀스 하나 선택 → 이벤트별 토큰 트레이스 + 원본 raw
    센서 값(±6시간 윈도, 시계열 차트 포함).
  - **자동생성 후보**: `scripts/generate_online2_narrative_catalog_auto_expansion.py`가
    만든 검증 리포트(dry-run만 통과, 아직 실제 카탈로그에 미반영)를 표로 훑어볼 수 있다.
    이 탭은 시퀀스 조회가 안 된다(실제로 빌드된 적이 없어서 매칭 인스턴스 자체가 없음).

## 알려진 한계

- 횡단비교 서사(`crosszone_hourly`/`crossfarm_hourly`/`crossfarm_growth`)는 `zone_id`가
  비어있어 단일 zone 기준 raw 조회를 건너뛴다 — 대신 토큰 트레이스의 `FARM_LOCAL|`/`ZONE_LOCAL|`
  토큰으로 어느 농장·구역들이 묶였는지 확인해야 한다.
- `training_events_v2.parquet`의 `narrative_id`는 그 파일이 만들어진 시점의 카탈로그
  버전을 따른다 — 예를 들어 control 코퍼스에는 88개 확장판의 새 서사(`X12`-`X15`,
  `D16`/`D17`/`S13`/`S14`)가 없다. 코퍼스 선택 드롭다운과 서사가 안 맞으면 "매칭된
  시퀀스를 못 찾음" 경고가 뜬다.
- raw CSV 조회(`lookup_raw_rows`)는 파일명 패턴(`{farm_id}_z{zone_id}.csv` 또는
  단일 결합 CSV)에 의존한다 — 원본 데이터 배치가 바뀌면 `../pipeline_explorer/data_access.py`
  쪽을 고쳐야 한다(이 앱은 그 함수를 그대로 가져다 쓸 뿐 수정하지 않음).
