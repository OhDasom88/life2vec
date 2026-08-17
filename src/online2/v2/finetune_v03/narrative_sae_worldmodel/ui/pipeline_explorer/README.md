# pipeline_explorer

**근거**: 계획서 §15, 사용자 요청("원시데이터에서 SAE에 이르는 전과정을 유기적으로 상세하게 분석" / "concept space, tcav, entity summary를 한눈에 확인") · **백엔드**: `../../narrative_grounding/`(narrative_id 매칭) + `../../sae/`(encoder·SAE) + `../../representation_explorer/`(projection·분리 probe) + `../../concept_tcav/`(TCAV) + `../../../llm_evidence/`(nearest examples) + 이 세션에서 만든 6-arm 재학습 산출물 · **상태**: 6탭 구현 완료, 실행 확인됨(`AppTest`로 6탭 전부 렌더링 + 버튼 트리거 경로까지 예외 없이 통과)

## 속성 그래프 — zone별 색 구분 (지그재그 선 버그 수정) (2026-07-26)

사용자 지적("특정 시점에 여러 데이터 포인트가 찍힘, 각 데이터포인트의 차이점도 드러나게"): finetune 케이스가 여러 zone에 걸치면(예: zone 1/3/4), co2_ppm처럼 farm 단위 센서값이 zone별 raw 파일에 각각 기록돼 있어서 같은 컬럼에 대해 매 시각 여러 zone의 값이 존재하는데, 기존 코드는 이걸 구분 없이 시각순으로만 하나의 선에 이어붙여서 서로 다른 zone 값이 지그재그로 섞여 보였다(사용자가 스크린샷으로 보여준 co2_ppm 그래프의 톱니 모양이 이 증상). 원인은 `render_property_trace_charts()`가 raw 값을 모을 때 `zone_id`를 아예 기록하지 않았던 것.

`long_rows`에 `zone_id`를 추가하고, 선/점 모두 `color=zone_id`로 인코딩해서 zone마다 선이 끊어져 따로 그려지게 했다(Vega-Lite는 `color`/`detail` 인코딩이 다르면 선을 자동으로 분리해서 이음 — zone이 섞여 지그재그가 되는 문제가 원천적으로 사라짐). 기존에 색으로 표현하던 "시퀀스 이벤트/raw 중간값" 구분은 색 채널을 zone에 내줬으므로 투명도(opacity, 진한 점=1.0/0.95, 옅은 점=0.35/0.4)로 옮기고 점 크기 구분은 유지했다. tooltip에도 zone_id/table을 추가해 점 하나하나가 어느 zone/테이블에서 왔는지 바로 확인 가능. zone이 1개뿐이면 범례를 숨겨 불필요한 시각 잡음을 줄인다. `AppTest`로 렌더링 예외 없음 확인.

## IMAGE 이벤트 포함 시 크래시 수정 — zone_id 없는 이벤트 처리 (2026-07-26)

사용자 보고("이미지 토큰을 사용하려고 하니 UI에서 오류 발생"): 7번 탭에서 "IMAGE 이벤트 포함" 체크박스를 켜면 `ValueError: invalid literal for int() with base 10: ''`로 크래시. 원인은 IMAGE 이벤트가 farm 단위지 zone 단위가 아니라 `zone_id`가 빈 문자열인데, `data_access.py::_load_raw_zone_csv`가 파일명 후보를 만들 때 무조건 `int(zone_id)`를 호출해서 빈 문자열에 터진 것 — "시퀀스 전체 속성 변화" 그래프가 케이스의 모든 이벤트(IMAGE 포함)를 순회하며 raw 조회를 하다 걸림. `int(zone_id)` 후보 생성을 `try/except`로 감싸 zone_id가 실제 숫자로 보일 때만 그 후보를 추가하도록 수정(빈 문자열이면 zone 무관 조회로 자연히 폴백). `AppTest`로 `finetune_tok_include_image=True` 상태에서 예외 없이 렌더링되는 것 확인.

## 속성 그래프 — 시간 구간 슬라이더 + 확대/이동 + 라벨 과밀 방지 (2026-07-26)

사용자 지적("그래프를 볼 때 한 화면에 너무 많은 데이터가 들어가서 알아보기 힘듦"): finetune 케이스(최대 2주, 수천 이벤트)를 7번 탭에서 보면 "시퀀스 전체 속성 변화" 그래프가 전체 구간의 raw 점을 한 번에 다 그려서(narrative 시퀀스는 보통 몇 시간~하루라 문제없었지만) 알아보기 어려웠다. `render_property_trace_charts()`(1번/7번 탭 공유)에 세 가지를 추가: (1) 전체 구간이 72시간을 넘으면 기본으로 최근 72시간만 그리고, 슬라이더로 구간을 넓히거나 다른 시점으로 옮길 수 있게 함(narrative처럼 원래 짧은 시퀀스는 슬라이더가 전체 구간을 그대로 보여줘 동작 변화 없음), (2) 각 그래프에 `.interactive()`를 붙여 마우스 스크롤 확대/드래그 이동 지원, (3) 화면에 보이는 구간의 bin 라벨이 40개를 넘으면 텍스트 라벨을 생략하고 tooltip으로만 보여줘(라벨끼리 겹쳐 안 보이는 것 방지) 캡션에 몇 개가 생략됐는지 표시. 높이도 170→260px로 키움. `AppTest`로 검증: narrative 시퀀스(6시간 구간)는 슬라이더가 (0,6) 전체를 그대로 보여줌, finetune 케이스(326시간=13.6일 구간)는 기본으로 (254,326) 즉 최근 72시간만 표시됨, 둘 다 예외 없음.

## 7번 탭 신설 — Finetune 케이스 시퀀스 ↔ 사전학습 vocab ↔ raw (2026-07-26)

사용자 요청("finetuning용 case별 시퀀스가 사전학습에 사용된 단어사전을 기반으로 어떻게 tokenizing 되는지 확인할 수 있도록 UI 조정 / raw 값과 매칭이 되어야 해 / token은 행단위로 매칭되는 것이 아니라 특정행의 특정 feature와 매칭됨을 고려"). 1번 탭은 narrative 카탈로그가 고른 시퀀스만 보여줘서, finetune(정상/비정상 진단)이 실제로 쓰는 케이스별(`case_manifest.csv`의 farm_id+기간) 시퀀스는 확인할 방법이 없었다.

새 7번 탭은 `case_manifest.csv`에서 케이스를 고르면, Stage A 캐싱(`cache_stage_a_event_embeddings.py::case_events_from_frame`)이 **실제로 쓰는 것과 동일한 이벤트 목록/순서**를 그대로 재사용해(별도 재구현 없음 — 로직 drift 방지) 보여준다. 선택한 run의 `build_dir_for_run()`으로 그 run이 실제로 학습된 코퍼스의 `life2vec_token_registry_v2.json`을 읽고, 케이스 전체 토큰에 대해 `vocab.token2index.get(t, unk)` — finetune 캐싱이 실제로 쓰는 것과 정확히 같은 lookup — 을 적용해 UNK 비율을 즉석 계산한다(`audit_finetune_vocab_unk.py`가 배치로 하던 걸 UI에서 케이스 단위로 실시간 확인). 이벤트 하나를 골라 펼치면 토큰이 role별 색 + UNK는 빨간 테두리로 표시되고(사전학습 vocab에 없어 `[MASK]`가 아니라 `[UNK]`로 대체되는 토큰을 그 자리에서 확인), 오른쪽엔 그 이벤트의 원본 raw CSV 행이 `source_row_id` 계보로 정확히 매칭되어 뜬다.

사용자가 지적한 "토큰이 행단위가 아니라 특정 행의 특정 feature와 매칭"은 이미 1번 탭이 만들 때 확보해둔 성질(`FEATURE|value` 결합 토큰을 `|`로 분리해 feature 이름을 raw 컬럼명과, group_id/토큰화 시각을 raw 행의 정확한 시각과 맞춰야만 라벨을 붙임)을 그대로 물려받는다 — 이번에 새로 푼 문제가 아니라 기존 매칭 정밀도를 finetune 케이스에도 적용한 것. 이 재사용을 위해 "시퀀스 전체 속성 변화" 그래프 블록을 `render_property_trace_charts()`로 뽑아 1번 탭과 7번 탭이 공유하게 리팩터했고, `data_access.py`에 `load_case_events_full()`(Stage A와 동일한 이벤트 선정을 하되 measurement_group_ids/token_roles/source_file_ids/source_row_ids까지 보존)을 추가했다. `AppTest`로 검증: run 선택→케이스 목록(55개)→새 run(`full_event_grain_simplified_2026-07-26`)에서 특정 케이스 UNK 비율 0.00%(3928이벤트/45828토큰) 확인, 이벤트 상세(event_position=50) 토큰 트레이스+raw 매칭까지 예외 없이 렌더링, 1번 탭도 리팩터 후 정상 동작 확인.

## 5번 탭 — emb_dir→인코더 run 역산 버그 수정 (옛 토큰 스킴이 계속 보이던 원인) (2026-07-26)

사용자 지적("왜 시퀀스 토큰에 QUALITY OK 같은 것들이 들어가 있는거지? / 선택한 checkpoint는 최근 finetune 모델이고 이 finetune 모델은 MEAS_SEP, QUALITY 같은 토큰을 전부 없애고 새로 학습한 pretrain 인코더를 쓰기 때문에 단어사전도 공유하고 있어야 하는거 아님?") — 정확한 지적이었다. 바로 위 "emb_dir/코퍼스 자동 해석"을 고칠 때 넣은 `encoder_run_for_emb_dir()`에 실제 버그가 있었다: `emb_dir`은 항상 `.../{코퍼스_태그}/event_embeddings` 형태인데, 구분해야 할 값(`v2_finetune_v03_simplified` 등)은 **부모** 폴더 이름이고, 마지막 폴더 이름은 항상 고정 문자열 `"event_embeddings"`다. 그런데 코드가 `Path(emb_dir).name`(마지막 폴더)으로 매핑 테이블을 조회해서 **항상 매치 실패 → `"original"`(구 코퍼스/vocab)로 폴백**하고 있었다 — 어떤 run을 골라도 항상 옛 토큰 스킴(`QUALITY|`/`[MEAS_SEP]`/분리된 `FEATURE|`+`VALUE_ABS|`)을 보여준 이유. `Path(emb_dir).parent.name`으로 수정. 겸사겸사 `finetune_run_emb_dir()`가 `config.json`의 상대경로(`args.emb_dir`이 절대경로가 아닌 걸 발견)를 Streamlit 프로세스의 CWD가 아니라 항상 저장소 루트 기준으로 풀도록 고쳐서 더 견고하게 만들었다. `AppTest`로 검증: 고치기 전엔 토큰 청크(HTML span) 14개 중 하나(옛 스킴)가 5번 탭 것이었는데, 고친 뒤엔 5번 탭 청크 전부(0~12번, 13번은 별개 탭인 7번 tab의 기본 선택 run="original"이라 정상적으로 구 스킴을 보여주는 것) `INSIDE_HUMIDITY_PCT|ABS_B27`류의 새 결합 토큰만 나옴 확인.

## 7번 탭 — ZONE_LOCAL을 실제로 계산해서 보여줌 (2026-07-26)

사용자 지적("현재 분석 UI에 보면 raw 데이터의 zone 정보를 참조한다는 내용으로 안보임"): 토큰 목록에 `ZONE_LOCAL`이 실제로 안 보였다. 원인 분석 — 이건 버그가 아니라 **`events_tokenized_v2.parquet`의 원본 SENTENCE에는 애초에 ZONE_LOCAL이 없는 게 맞다**: 사전학습도 finetune Stage A도 시퀀스/윈도우를 조립하는 순간에 그 안의 zone 구성을 보고 즉석으로 `ZONE_LOCAL|i`를 끼워 넣기 때문에(다른 윈도우에 들어가면 인덱스가 달라짐), corpus 파일 자체에는 저장될 수가 없다. 이전엔 이 사실을 caption 텍스트로만 설명해서 "말로만 있고 화면엔 안 보인다"는 인상을 줬다.

이제 이 이벤트를 실제로 Stage A 타겟으로 삼아 윈도우를 구성해보고(`construct_target_window`, 기본 1024토큰), 그 윈도우 안의 zone 구성으로 `ZONE_LOCAL|i`를 계산해서 **점선 테두리 칩으로 토큰 목록 맨 앞에 실제로 표시**한다(corpus에 저장된 토큰과 구분되도록 점선). caption도 "여기선 고정값으로 보여줄 수 없다"에서 "지금 계산한 값이고, 다른 윈도우에 포함되면 달라질 수 있다"로 바꿔 실제 값을 보여주면서 한계도 같이 명시. `AppTest`로 `ZONE_LOCAL|0` 칩이 실제로 렌더링되는 것 직접 확인(8개 markdown 블록에서 발견 — role-only 참고 목록과 IxG 결과 양쪽에 다 반영).

## 5번 탭 예측/근거 — 진짜 토큰 IxG + run별 emb_dir/코퍼스 자동 해석 (2026-07-26)

사용자 요청("finetune의 summary space와 예측 결과 그리고 그 예측결과와 관련된 saliency score(주요 이벤트, 주요 토큰)을 확인할 수 있도록"). summary space(개념 공간, "개념 공간 + Split검증" 하위 탭)와 예측 결과·이벤트 단위 saliency는 이미 있었지만, 두 가지가 비어 있었다: (1) "케이스 요약 · 예측/근거"가 `emb_dir`/`events_path`를 `v2_finetune_v02`/`v2_build`로 하드코딩해서, 5번 탭 상단에서 다른 run(예: 이번 세션에 새로 학습한 `cv_repeated_zone_local_simplified_2026-07-26`)을 골라도 항상 옛 코퍼스를 참조했다. (2) 주요 이벤트 안의 토큰은 role 색만 입혔을 뿐 "**토큰별 기여도는 근사치다 — 진짜 attribution인 척하지 않는다**"고 스스로 명시해뒀을 정도로 실제 gradient가 아니었다.

(1)은 `data_access.py`에 `finetune_run_emb_dir(run_dir)`(그 run의 `config.json`에 기록된 실제 `--emb-dir`을 읽음)와 `encoder_run_for_emb_dir(emb_dir)`(그 emb_dir을 인코딩한 사전학습 run 이름 역산 — `EMB_DIR_TO_ENCODER_RUN` 매핑)을 추가해 해결. 이제 `build_case_evidence`/`DiagnosisEventDatasetV03`가 선택된 run에 맞는 emb_dir/vocab/코퍼스를 자동으로 쓴다.

(2)는 방금 만든 `token_saliency.py::token_ixg_for_event_v03`를 실제로 연결해서 해결 — "예측+근거 계산" 버튼을 누르면 이벤트 단위 saliency(top_abnormal_events/top_normal_events) 각각에 대해 그 이벤트만 다시 grad 활성화 상태로 인코딩해 진짜 토큰별 IxG를 계산하고, 빨강(비정상 쪽으로 밈)/파랑(정상 쪽으로 밈) 농도로 표시한다(테두리는 기존 role 색 유지, 참고용 role-only 토큰 목록도 아래에 남겨둠). `AppTest`로 새 run 선택 → 케이스 선택 → "예측+근거 계산" 버튼 클릭까지 실제 forward pass + 여러 이벤트의 token IxG 계산을 예외 없이 확인(예측=기형과_생리장해, 정답=탄저병_발생_위험, p_abnormal=0.949).

## run 목록 하드코딩 제거 + run별 vocab/registry 자동 해석 (2026-07-26)

사용자 지적("사전학습 모델이 ui에서 안보여"): 새로 학습한 run(`full_event_grain_simplified_2026-07-26`, `v2_build_expanded_full1517` 코퍼스로 학습, 셀당 토큰 1개 단순화 스킴)이 UI 어느 run 선택 드롭다운에도 나타나지 않았다. 원인은 `app.py`의 `run_names = ["original"] + list(da.ABLATION_ARMS)`가 7개 run으로 고정 하드코딩돼 있었던 것(3곳: encoding_run/sim_run/vc_run 셀렉트박스). `data_access.py`에 `resolve_run_dir(run_name)`(원래 `original`/ablation-arm/`v2_runs/` 직속 서브디렉터리 3가지 케이스를 모두 지원)과 `discover_all_run_names()`(`v2_runs/` 밑에서 `best.ckpt`나 `checkpoint_step_*.pt`/`last.ckpt`가 있는 디렉터리를 전부 스캔)를 추가하고, 3개 셀렉트박스 모두 이 함수로 교체. `list_local_runs`/`run_created_at`/`checkpoint_path_for_run`도 `resolve_run_dir` 기반으로 재작성.

여기서 그치지 않고 더 깊은 잠재 버그를 하나 더 발견해 같이 고쳤다: `load_vocab_and_abspos()`/`load_farm_events_cached()`가 선택된 run과 무관하게 `outputs/online2/v2_build`(728토큰, 구 스킴) 경로에 하드코딩돼 있었다 — 새 run은 `v2_build_expanded_full1517`(3404토큰, 셀당-토큰-1개 스킴)로 학습됐으므로, 새 run을 2/3/6번 탭에서 실제로 선택하면 이번 세션 초반에 겪었던 것과 같은 종류의 CUDA `indexSelectLargeIndex` 임베딩 인덱스 초과 크래시가 났을 것. `build_dir_for_run(run_name)`을 추가해 각 run의 `run_manifest_v2.json`의 `parquet` 필드에서 그 run이 실제로 학습된 코퍼스 디렉터리를 역산하도록(값이 없으면 `v2_build`로 폴백) 하고, `load_vocab_and_abspos`/`load_farm_events_cached`/`load_encoder_cached`/`encode_sequence`에 `run_name`을 관통시켜 각자 자기 run의 vocab/registry를 쓰게 함. `AppTest`로 실제 forward pass까지 검증: `encoding_run`에서 `full_event_grain_simplified_2026-07-26` 선택 → encode 버튼 클릭까지 예외 없이 통과.

## 왜 별도 앱인가

`sequence_composer_ui`(원시↔시퀀스 편집), `sae_feature_dashboard`(SAE 분석), `run_artifact_control`(run 비교)는 계획서 §15에서 서로 다른 서브앱으로 나뉘어 있었다. 하지만 사용자가 요청한 건 "원시데이터에서 SAE에 이르는 전과정을 유기적으로" 보는 것 — 세 서브앱을 따로 열면 이 요청과 정반대(단편적)가 된다. 그래서 이 세 서브앱이 계획했던 기능 중 **조회(read-only) 영역**만 하나의 앱으로 통합했다. 각 서브앱에 원래 계획됐던 **편집/개입 액션**(sequence_composer_ui의 curation_actions 트리거, sae_feature_dashboard의 개입 실험 트리거)은 여기 없다 — 아래 "안 하는 것" 참조.

## 1번 탭 속성 그래프 — raw 중간값 포함 + 시퀀스/비-시퀀스 구분 (2026-07-25)

사용자 지적("앞에 두개만 토큰 정보가 표시되고 나머지 선에는 표시되지 않아 / 이벤트에 속한 정보만 표시되는게 아니라 raw 데이터상 중간 값이 있으면 표시(단 시퀀스에 표시된 값과 표시되지 않은 값이 구분되게)")에 따라 그래프 데이터 소스를 바꿨다. 기존엔 시퀀스의 각 "이벤트"마다 `lookup_raw_rows(..., window_hours=N)`로 그 이벤트 시각 ±N시간만 따로 조회해서 이어붙였다 — 이벤트 자체가 드문드문 있으면 그 사이 raw 표본이 있어도 안 보였다. 이제는 `data_access.py`에 새로 추가한 `lookup_raw_rows_range(farm_id, zone_id, start, end)`로, 그 시퀀스의 첫 이벤트~마지막 이벤트 구간(± `raw_match_window`시간)의 raw 표본을 **연속으로** 전부 가져온다(기존 `lookup_raw_rows`는 이 range 함수 위에 재구현해 동작은 그대로). 각 raw 행은 그 시퀀스의 실제 이벤트 시각과 정확히 일치하는지로 "시퀀스 이벤트"(진한 점, 초록) vs "raw 중간값(비-시퀀스)"(옅은 점, 회색)를 나눠 표시하고, bin 라벨은 여전히 시퀀스 이벤트 중에서도 그 feature를 실제로 토큰화한 경우에만 붙는다(2026-07-25 앞선 수정 유지). `AppTest`로 검증: window=0에서 58/58 raw 행이 전부 시퀀스 이벤트(이 narrative는 원래 표본 간격=이벤트 간격이라 우연히 전부 일치), window=6h로 넓히면 58/232로 늘어 실제 중간값이 들어오는 것 확인, 예외 없음.

## 5번 탭(Entity Summary) 하위 탭으로 정리 (2026-07-25)

사용자 지적("예측 및 근거 탭이 안보이는데 혹시 탭이 여러 프로젝트로 나뉘어 져있으면 하나로 합쳐줘")으로 확인: 여러 프로젝트로 나뉜 게 아니라(streamlit 프로세스 1개, 터널 1개, `ps`/`ss`로 재확인) 5번 탭 안에 개념공간→split검증→SAE분석→케이스요약→예측근거→TCAV 6개 섹션이 순서대로 쌓여 있어서 스크롤로 찾기 어려웠던 것. `st.tabs`로 4개 하위 탭("개념 공간 + Split검증", "SAE 분석", "케이스 요약 · 예측/근거", "TCAV")으로 재구성했고, `case_id`/`ckpt_path`/`rep_choice` 선택은 모든 하위 탭이 공유하므로 탭 밖으로 뺐다. 체크포인트가 없는 run이어도 개념 공간/SAE 분석은 계속 볼 수 있도록, 기존의 `if not ckpt_options: return`(전체 탭을 막아버리던 조기 종료)을 없애고 예측/근거·TCAV 하위 탭에서만 개별적으로 안내 문구를 보여주게 바꿨다. `AppTest`로 4개 하위 탭 전부 검증: 예측+근거(예측=동해_난방실패, p_abnormal=0.782), SAE 대조, TCAV(sign_mean=0.941, concept 30,630건/random 16,506건) 모두 예외 없이 실행됨.

## 1번 탭 속성 그래프 — 라벨 오귀속 버그 수정 (2026-07-25)

사용자 지적("co2_ppm이 490과 410이 전부 ABS_B09로 되어 있어 bin이 제대로 생성된거야?")으로 발견: raw 행 매칭 오차 슬라이더(`raw_match_window`)를 0시간보다 넓게 잡으면, 한 이벤트가 실제로 토큰화한 bin(`bin_by_feature`, 그 이벤트의 정확한 시각 하나에서만 계산됨)이 그 창 안에서 함께 조회된 **다른 시각의** raw 행들에까지 그대로 찍혔다 — 서로 다른 raw 값(예: co2_ppm=490과 co2_ppm=410)이 같은 bin 라벨을 갖는 것처럼 보이는 원인. 이진 탐색 자체(`BinRuleV2.token_suffix`)는 정상 확인(410→ABS_B05, 490→ABS_B09, 실제 edges 기준). 수정: 이제 raw 행의 timestamp가 그 이벤트의 정확한 토큰화 시각과 tz-정규화 후 완전히 일치할 때만 라벨을 붙이고, 창 확장으로 함께 표시되는 주변 시각 행은 라벨 없이 선(raw 값)만 그린다. `AppTest`로 검증: window=24h에서 매칭 수가 1127/4669(버그) → 14/4669(수정 후, exact-timestamp 행만)로 줄었고 예외 없음.

또한 vocab에 bare `QUALITY`/`VALUE_ABS` 토큰이 남아있는지도 같이 확인 요청받음 — 실제로는 아니었다: `vocab_v2.json`의 `family` 메타 필드 값(`"family": "QUALITY"` 등, 마스킹 정책 조회용 분류 라벨)과 실제 `token` 필드를 구분 없이 grep해서 착시가 생겼을 뿐, 실제 vocab에 등록된 토큰은 전부 `QUALITY|OK` / `VALUE_ABS|ABS_B00` 같은 `PREFIX|suffix` 형식이고 bare 토큰은 없다(`build_vocab_v2`가 항상 `f"{fam}|{suffix}"`로만 append).

## 1번 탭에 서사 설명 + 속성별 시간 그래프 추가 (2026-07-24)

- **서사 설명**: narrative_id 선택 직후 카탈로그(`load_narrative_catalog_full`)에서 이름·목적·기대 패턴·해석을 info box로 표시.
- **시퀀스 전체 속성 변화 그래프**: 시퀀스에 포함된 모든 이벤트의 raw 수치 컬럼(4개 테이블 전부)을 시간축으로 이어붙여 그린다. y축은 항상 raw 연속값(대소 비교 기준), 각 점 위에는 그 시점 tokenizer가 실제로 매긴 bin 라벨(`*_combined` role 토큰이 이미 `FEATURE|bin` 형식이라 그대로 재사용, 소문자화해서 raw 컬럼명과 매칭 — 실측: 매칭률 14/58, actuator on/off류처럼 bin이 없는 값은 라벨 없이 선만 표시).

## 예측/근거의 시간 메타정보 + raw 그래프 (2026-07-24)

사용자 지적("이벤트 정보에 시간 정보가 누락", "raw 데이터도 그래프로 같이 보고 싶다")에 따라 Entity Summary "예측 및 근거"의 주요 이벤트 표시에 절대 시각(`observation_timestamp`)·farm·zone을 항상 같이 보여주고, `narrative_evidence_explorer`가 이미 갖고 있던 raw-데이터 시각화 패턴(altair 시계열 + 이벤트 시각 표시선)을 재사용해 그 이벤트 전후(±N시간, 슬라이더로 조절) 원본 raw 수치를 컬럼별 그래프로 붙였다.
- **실제로 잡은 버그**: `observation_timestamp`(tz-aware, UTC)와 raw CSV의 `timestamp`(tz-naive)를 그대로 비교하면 pandas가 `TypeError`를 던진다 — `data_access.py::lookup_raw_rows`에 tz 정규화를 추가해 근본 수정(호출부마다 tz를 벗기게 시키지 않고, 공용 함수 안에서 한 번에 처리).
- **별개로 잡은 UX 버그**: 슬라이더/컬럼 선택 위젯을 `st.button(...)` 블록 안에 두면, 그 위젯을 조작한 rerun에서 버튼의 "눌림" 상태가 사라져 전체 결과가 통째로 사라진다 — `st.session_state`에 계산 결과를 캐싱하고 표시 로직을 버튼 블록 밖으로 빼서 해결(재계산 없이 위젯 조작 가능).

## UX 개선 (2026-07-24)

- **narrative_id 가독성**: 1번/3번 탭에서 `A05` 같은 ID만 보이던 걸 `datasets/agrichallenge/online2/online2_narrative_catalog.csv`(narrative_name_ko/purpose)와 조인해 `A05 — 순환팬 장기 무가동 후보`처럼 표시. 3번 탭 Concept Space는 선택한 narrative들의 purpose 전문도 펼쳐볼 수 있음.
- **Entity Summary 모델 선택**: run 디렉토리를 직접 타이핑하는 대신, `outputs/online2/v2_finetune_v03/runs/` 아래 OOF representation이 있는 run을 자동 감지해 `run이름 (저장시각, macro_f1=...)` 형식 드롭다운으로 제공(최신순). 필요하면 "직접 경로 입력" 체크박스로 수동 전환 가능.

## 6개 탭

1. **원시데이터 ↔ 토큰 ↔ 시퀀스 추적** — narrative_id로 시퀀스를 고르면, 그 시퀀스를 이루는 모든 이벤트를 순서대로 보여준다. 각 이벤트마다 (a) 토큰을 role(`feature_identity`/`value_abs`/`value_global`/`value_farm`/`quality`/`sep`/`meta`/`derived`/`literal`)별로 색칠해서 보여주고, (b) 그 이벤트의 farm_id/zone_id/timestamp로 원본 raw CSV(`datasets/agrichallenge/online2/data/{E_environment,R_rootzone,A_actuator,G_growth}/`)에서 일치하는 행을 찾아 나란히 보여준다.
2. **인코딩 트레이스** — 1번 탭에서 고른 시퀀스를 실제로 순전파해서(`pilot_sae_layer_decoder_random_comparison.encode_all_positions` 재사용) 6개 encoder layer + MLM/SOP 디코더 변환, 총 8개 지점의 pooled activation norm을 보여주고, 그 옆에 (미리 계산해 둔 리포트에서 조회하는) 지점별 dead_feature_ratio를 같이 보여준다. 원본 체크포인트와 6개 ablation arm 중 아무거나 골라 비교 가능.
3. **개념/시퀀스 유사도** — 두 시퀀스를 골라 (a) raw activation 코사인 유사도(항상 계산 가능), (b) SAE 코드 코사인 유사도 + 어떤 feature가 공통으로 활성화됐는지(저장된 SAE가 있는 조합만 — 기본은 `original`/`control`/`sop_balanced`의 layer5). **+ Concept Space 시각화**(신규, 2026-07-24): 여러 narrative_id의 시퀀스를 골라 한 번에 인코딩해서 `representation_explorer/projection.py`로 2D(PCA/UMAP) 투영, narrative_id별 색상으로 개념 군집을 산점도로 보여줌 — 두 시퀀스 비교로는 안 보이던 "여러 개념이 실제로 갈려 있는가"를 시각적으로 확인.
4. **학습 버전 브라우저** — 원본 체크포인트 + 6개 ablation arm을 `run_manifest_v2.json` 기준 config/지표 표로 한눈에 비교하고, 두 개를 골라 상세 diff.
5. **Entity Summary** (신규, 2026-07-24) — finetune 진단 케이스(정상/비정상)의 out-of-fold 표현을 `analysis/visualisation/person_space.ipynb`(원본 life2vec)와 같은 "개체별 표현 공간 위치" 방식으로 본다: (a) `representation_explorer/label_separation_probe.py`의 AUROC/실루엣 실측치, (b) `representation_explorer/projection.py`로 만든 PCA 산점도(정상/비정상 색상 구분, disclaimer 항상 표시), (c) **Split 검증** — 각 케이스가 held-out된 fold를 라벨로 `representation_explorer/overlap_check.py`(기존 모듈 재사용)를 돌려, 표현 공간에서 fold끼리 얼마나 섞여 있는지 표로 보여줌, (d) 케이스 하나를 고르면 코사인 유사도 기준 유사 학습 사례(`llm_evidence/nearest_examples.py` 재사용), (e0) **SAE 기반 분석**(신규) — 3번 탭 학습 산출물(`pipeline_explorer_saes/`, 새 SAE 학습 없이 재사용)로 라벨된 35개 케이스 전체 이벤트(137,480개)를 인코딩해서, feature별 정상/비정상 평균 활성·발화율 차이를 표+막대그래프로 대조 — 실측: dead_feature_ratio=0.125(이벤트 단위 큰 표본이라 기존 SAE pilot의 시퀀스 단위 표본보다 훨씬 낮음), (e) **예측 및 근거**(신규) — 선택한 케이스를 실제 forward pass해서 예측(fine 확률 argmax) vs 정답 vs `p_abnormal`을 나란히 보여주고, saliency 상위 이벤트(`llm_evidence/evidence_bundle.py::build_case_evidence` 재사용)를 그 이벤트를 이루는 토큰까지 role별 색상으로 펼쳐 보여준다 — 토큰 기여도는 이벤트 단위 saliency를 role로 덧씌운 근사치임을 UI에 항상 명시(사전학습 encoder가 토큰을 이벤트로 pooling한 뒤라 독립적 토큰 gradient는 이 아키텍처에서 계산 불가), (f) narrative_id를 입력하고 버튼을 누르면 그 자리에서 TCAV(`concept_tcav/`) 계산 — `NOT_RUN`이 아니라 진짜 계산. (e)/(f)는 체크포인트 선택을 공유한다.
6. **Vocab Concept Space** (신규, 2026-07-24) — 사용자 요청("life2vec의 concept space(단어사전에 등록된 token 단위)를 그려줘")에 답한 탭. 3/5번 탭이 문맥별 activation을 투영하는 것과 달리, 이 탭은 `model.transformer.embedding.token.weight`(고정 vocab embedding table, 계획서 §8.3의 "Token embedding" — sequence/checkpoint/layer에 안 흔들리는 고정값) 자체를 투영한다. `RegistryVocabulary.vocab()`이 제공하는 728개 token × `CATEGORY`(58종, 예: `INSIDE_TEMP_C`/`VALUE_ABS`/`QUALITY`/`FEATURE` 등) 라벨로 색상 구분. 실측: PCA로 728개 token 전부 투영 시 neighborhood_preservation=0.820.

## 사전 준비 (한 번만 실행)

```bash
conda run -n life2vec python scripts/online2_v2/build_pipeline_explorer_index.py
conda run -n life2vec python scripts/online2_v2/train_and_save_pipeline_explorer_sae.py
conda run -n life2vec streamlit run src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/pipeline_explorer/app.py --server.port 8502
```

- `build_pipeline_explorer_index.py`: 80개 narrative_id마다 최대 15개 시퀀스씩(총 1,200개 시퀀스, 16,362개 이벤트 행) 샘플링해 `outputs/online2/sae_pilot/pipeline_explorer_index.parquet`에 저장. **주의**: event_id만으로 필터링하면 안 된다 — narrative 기반 재윈도잉 때문에 같은 raw 이벤트가 여러 시퀀스에 걸쳐 재사용돼서, event_id 단위로 필터링했다가 1,519개 타깃에서 97,829행이 나온 적이 있다(원인 파악 후 sequence_id 단위로 고침).
- `train_and_save_pipeline_explorer_sae.py`: `DEFAULT_COMBOS`에 나열된 (run, position) 조합만 SAE를 학습해 `outputs/online2/sae_pilot/pipeline_explorer_saes/`에 저장(기본: original/control/sop_balanced × layer5). 다른 조합을 3번 탭에서 보고 싶으면 이 리스트에 추가하고 재실행.
- 5번 탭(Entity Summary)은 별도 산출물이 필요하다: `run_diagnosis_finetune_v03.py --mode cv_repeated`(OOF representation npz)와 `probe_normal_abnormal_separation_v03.py`(AUROC 리포트) — 없으면 탭이 경고만 보여주고 죽지 않는다.

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
