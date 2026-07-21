# pipeline_explorer

**근거**: 계획서 §15, 사용자 요청("원시데이터에서 SAE에 이르는 전과정을 유기적으로 상세하게 분석") · **백엔드**: `../../narrative_grounding/`(narrative_id 매칭) + `../../sae/`(encoder·SAE) + 이 세션에서 만든 6-arm 재학습 산출물 · **상태**: 4탭 구현 완료, 실행 확인됨(`AppTest`로 4탭 전부 렌더링 + 버튼 트리거 경로까지 예외 없이 통과)

## 왜 별도 앱인가

`sequence_composer_ui`(원시↔시퀀스 편집), `sae_feature_dashboard`(SAE 분석), `run_artifact_control`(run 비교)는 계획서 §15에서 서로 다른 서브앱으로 나뉘어 있었다. 하지만 사용자가 요청한 건 "원시데이터에서 SAE에 이르는 전과정을 유기적으로" 보는 것 — 세 서브앱을 따로 열면 이 요청과 정반대(단편적)가 된다. 그래서 이 세 서브앱이 계획했던 기능 중 **조회(read-only) 영역**만 하나의 앱으로 통합했다. 각 서브앱에 원래 계획됐던 **편집/개입 액션**(sequence_composer_ui의 curation_actions 트리거, sae_feature_dashboard의 개입 실험 트리거)은 여기 없다 — 아래 "안 하는 것" 참조.

## 4개 탭

1. **원시데이터 ↔ 토큰 ↔ 시퀀스 추적** — narrative_id로 시퀀스를 고르면, 그 시퀀스를 이루는 모든 이벤트를 순서대로 보여준다. 각 이벤트마다 (a) 토큰을 role(`feature_identity`/`value_abs`/`value_global`/`value_farm`/`quality`/`sep`/`meta`/`derived`/`literal`)별로 색칠해서 보여주고, (b) 그 이벤트의 farm_id/zone_id/timestamp로 원본 raw CSV(`datasets/agrichallenge/online2/data/{E_environment,R_rootzone,A_actuator,G_growth}/`)에서 일치하는 행을 찾아 나란히 보여준다.
2. **인코딩 트레이스** — 1번 탭에서 고른 시퀀스를 실제로 순전파해서(`pilot_sae_layer_decoder_random_comparison.encode_all_positions` 재사용) 6개 encoder layer + MLM/SOP 디코더 변환, 총 8개 지점의 pooled activation norm을 보여주고, 그 옆에 (미리 계산해 둔 리포트에서 조회하는) 지점별 dead_feature_ratio를 같이 보여준다. 원본 체크포인트와 6개 ablation arm 중 아무거나 골라 비교 가능.
3. **개념/시퀀스 유사도** — 두 시퀀스를 골라 (a) raw activation 코사인 유사도(항상 계산 가능), (b) SAE 코드 코사인 유사도 + 어떤 feature가 공통으로 활성화됐는지(저장된 SAE가 있는 조합만 — 기본은 `original`/`control`/`sop_balanced`의 layer5).
4. **학습 버전 브라우저** — 원본 체크포인트 + 6개 ablation arm을 `run_manifest_v2.json` 기준 config/지표 표로 한눈에 비교하고, 두 개를 골라 상세 diff.

## 사전 준비 (한 번만 실행)

```bash
conda run -n life2vec python scripts/online2_v2/build_pipeline_explorer_index.py
conda run -n life2vec python scripts/online2_v2/train_and_save_pipeline_explorer_sae.py
conda run -n life2vec streamlit run src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/pipeline_explorer/app.py --server.port 8502
```

- `build_pipeline_explorer_index.py`: 80개 narrative_id마다 최대 15개 시퀀스씩(총 1,200개 시퀀스, 16,362개 이벤트 행) 샘플링해 `outputs/online2/sae_pilot/pipeline_explorer_index.parquet`에 저장. **주의**: event_id만으로 필터링하면 안 된다 — narrative 기반 재윈도잉 때문에 같은 raw 이벤트가 여러 시퀀스에 걸쳐 재사용돼서, event_id 단위로 필터링했다가 1,519개 타깃에서 97,829행이 나온 적이 있다(원인 파악 후 sequence_id 단위로 고침).
- `train_and_save_pipeline_explorer_sae.py`: `DEFAULT_COMBOS`에 나열된 (run, position) 조합만 SAE를 학습해 `outputs/online2/sae_pilot/pipeline_explorer_saes/`에 저장(기본: original/control/sop_balanced × layer5). 다른 조합을 3번 탭에서 보고 싶으면 이 리스트에 추가하고 재실행.

## 안 하는 것 (원래 계획과의 차이, 정직하게 기록)

- **편집 액션 없음**: `sequence_composer_ui`가 원래 계획했던 `INCLUDE|EXCLUDE|MASK|REPLACE|SPLIT|MERGE|REORDER|REWINDOW` 트리거는 여기 없다. 이 앱은 조회 전용이다 — `sequence_curation.curation_actions`를 사람이 직접 트리거하는 화면은 여전히 미착수.
- **SAE 상태 머신/개념 매핑 없음**: `sae_feature_dashboard`가 원래 계획했던 feature 상태 머신 뷰(`SAE_LATENT → ... → CIRCUIT_SUPPORTED_FEATURE`), `concept_mapping`의 N:M 뷰, `feature_alignment`의 `PRESERVED|SPLIT|...` 뷰는 여기 없다 — 그 백엔드 모듈(`sae/concept_mapping.py`, `sae/feature_alignment.py`) 자체가 아직 없기 때문(`../../sae/README.md`의 "아직 없는 것" 참조). 3번 탭의 "공통 활성 feature" 목록은 그 대신 볼 수 있는 가장 가벼운 근사치이고, feature 번호를 "개념"이라고 자동으로 부르지 않는다(화면에 이 경고를 항상 표시).
- **W&B 연동 없음**: `run_artifact_control`이 원래 계획했던 W&B run/sweep 조회는 없다 — 이번 세션의 모든 run이 `--no-wandb`로 실행됐기 때문에 로컬 `run_manifest_v2.json`만 읽는다.
- **원시→토큰 매칭이 근사치인 경우 있음**: raw CSV 조회는 `(farm_id, zone_id, timestamp)` 정확 일치로 찾는다 — `G_growth`처럼 zone 구분이 없거나 timestamp 단위가 다른 테이블은 못 찾을 수 있고, 그 경우 화면에 "일치하는 raw 행을 못 찾음"이라고 명시적으로 표시한다(조용히 빈 화면을 보여주지 않는다).

## 구현한 것

- [`data_access.py`](data_access.py) — Streamlit에 의존하지 않는 순수 조회 함수(단위 테스트 가능): `parse_token_trace`(SENTENCE+역할+그룹을 토큰별 레코드로), `lookup_raw_rows`(raw CSV 조인), `list_local_runs`(manifest 인덱싱), `load_ablation_report`/`load_layer_decoder_random_report`(기존 SAE 리포트 조회).
- [`app.py`](app.py) — 4탭 Streamlit 렌더링. 인코딩·SAE 로직은 전부 이 세션에서 이미 만든 함수(`encode_all_positions`, `SparseAutoencoder`)를 그대로 import해서 재사용 — 새 인코딩/SAE 로직을 따로 만들지 않았다.
- [`scripts/online2_v2/build_pipeline_explorer_index.py`](../../../../../../scripts/online2_v2/build_pipeline_explorer_index.py)
- [`scripts/online2_v2/train_and_save_pipeline_explorer_sae.py`](../../../../../../scripts/online2_v2/train_and_save_pipeline_explorer_sae.py)
