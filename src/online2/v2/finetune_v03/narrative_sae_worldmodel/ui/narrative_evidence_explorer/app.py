"""Narrative Evidence Explorer (Streamlit) — 서사(해석) ↔ 실제 시퀀스(데이터) 대조.

실행:
  conda run -n life2vec streamlit run \
    src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/narrative_evidence_explorer/app.py

서사 카탈로그(`online2_narrative_catalog.csv`)의 `purpose`/`expected_pattern`/
`agronomic_interpretation`은 사람이 "이 raw 구간은 이런 패턴이다"라고 써 둔
**해석**이다. 모델이 실제로 학습에 쓰는 건 그 해석이 골라낸 raw 구간을
토큰화한 **시퀀스**뿐이다. 이 앱은 이 둘을 나란히 놓고, 서사 하나를 고르면
(1) 그 서사의 해석 텍스트, (2) 실제로 매칭된 시퀀스 목록, (3) 그중 하나를
고르면 토큰화된 SENTENCE와 원본 raw 센서 CSV 값을 같이 보여준다 — "이 서사가
주장하는 패턴이 실제 데이터에도 있는지" 사람이 직접 확인할 수 있게.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data_access as da  # noqa: E402

st.set_page_config(page_title="Narrative Evidence Explorer", layout="wide")
st.title("Narrative Evidence Explorer")
st.caption("서사(사람의 해석) ↔ 실제 매칭된 시퀀스(raw 데이터) 대조")


@st.cache_data(show_spinner=False)
def _catalog() -> pd.DataFrame:
    return da.load_catalog()


@st.cache_data(show_spinner=False)
def _auto_candidates() -> pd.DataFrame:
    return da.load_auto_expansion_candidates()


@st.cache_data(show_spinner="시퀀스 조회 중 ...")
def _matched_instances(corpus_label: str, narrative_id: str) -> pd.DataFrame:
    corpus = next(c for c in da.CORPUS_CHOICES if c.label == corpus_label)
    return da.load_matched_instances(corpus, narrative_id)


catalog = _catalog()
if catalog.empty:
    st.error(f"카탈로그를 못 찾음: {da.CATALOG_PATH}")
    st.stop()

tab_active, tab_candidates = st.tabs(["활성 서사 (실제 학습에 반영됨)", "자동생성 후보 (아직 미반영, dry-run만)"])

with tab_active:
    col_pick, col_corpus = st.columns([3, 2])
    with col_corpus:
        corpus_label = st.selectbox("코퍼스", [c.label for c in da.CORPUS_CHOICES])
    with col_pick:
        cat_filter = st.multiselect("카테고리 필터", sorted(catalog["category"].unique()))
        view = catalog if not cat_filter else catalog[catalog["category"].isin(cat_filter)]
        narrative_id = st.selectbox(
            "서사 선택",
            view["narrative_id"].tolist(),
            format_func=lambda nid: f"{nid} — {view.set_index('narrative_id').loc[nid, 'narrative_name_ko']}",
        )

    row = da.narrative_row(catalog, narrative_id)
    if row is None:
        st.warning("카탈로그에서 못 찾음")
        st.stop()

    st.subheader(f"[{row['narrative_id']}] {row['narrative_name_ko']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("카테고리", row["category"])
    c2.metric("materialization_key", row["materialization_key"])
    c3.metric("실측 unique_instance_count", row.get("unique_instance_count", "?"))

    st.markdown("#### 서사의 해석 (사람이 쓴 주장)")
    st.markdown(f"- **목적(purpose)**: {row['purpose']}")
    st.markdown(f"- **기대 패턴(expected_pattern)**: {row['expected_pattern']}")
    st.markdown(f"- **농학적 해석(agronomic_interpretation)**: {row['agronomic_interpretation']}")
    st.markdown(f"- **혼재요인(confounders)**: {row['confounders']}")
    with st.expander("전체 카탈로그 행 보기"):
        st.json(row)

    st.markdown("#### matcher 상세 — 실제로 어떻게 raw 데이터를 골라내는지")
    parsed = da.parse_matcher(str(row["materialization_key"]))
    st.markdown(f"**판정 규칙**: {parsed['description']}")
    mp1, mp2, mp3 = st.columns(3)
    mp1.metric("관련 컬럼", ", ".join(parsed["columns"]) or "—")
    mp2.metric("window (시간)", parsed["window_hours"] if parsed["window_hours"] else "단일 시점")
    if parsed["thresholds"]:
        mp3.metric("임계값 개수", len(parsed["thresholds"]))
        st.dataframe(pd.DataFrame(parsed["thresholds"]), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 실제로 매칭된 시퀀스")
    instances = _matched_instances(corpus_label, narrative_id)
    if instances.empty:
        st.warning(
            "이 코퍼스에서 매칭된 시퀀스를 못 찾음 — narrative_id가 이 코퍼스의 "
            "카탈로그 버전과 다르거나(예: 88개 확장 이전 서사를 control에서 조회), "
            "training_events_v2.parquet 경로를 확인하세요."
        )
        st.stop()

    seq_ids = instances["sequence_id"].unique().tolist()
    st.write(f"{len(seq_ids)}개 시퀀스 인스턴스 (최대 200개까지만 로드)")
    chosen_seq = st.selectbox("시퀀스 선택", seq_ids)

    seq_events = instances[instances["sequence_id"] == chosen_seq]
    detail = da.instance_detail(seq_events)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("farm_id", detail["farm_id"])
    d2.metric("zone_id", detail["zone_id"] or "(없음/횡단비교)")
    d3.metric("이벤트 수", detail["n_events"])
    d4.metric("order_semantics", detail["order_semantics"])

    st.markdown("#### 토큰화된 시퀀스 (모델이 실제로 보는 것)")
    for _, ev in seq_events.iterrows():
        with st.expander(f"event_position={ev['event_position']}  event_id={ev['event_id'][:24]}..."):
            entries = da.parse_token_trace(
                str(ev["SENTENCE"]), ev.get("measurement_group_ids", "[]"), ev.get("token_roles", "[]")
            )
            st.dataframe(
                pd.DataFrame([{"pos": e.position, "token": e.token, "role": e.role, "group": e.group_id} for e in entries]),
                use_container_width=True, height=200,
            )

    st.markdown("#### 원본 raw 센서 데이터 (서사 ↔ 실제 값 대조)")
    n_pairs = seq_events[["farm_id", "zone_id"]].drop_duplicates().shape[0]
    ctx1, ctx2 = st.columns([1, 3])
    with ctx1:
        context_hours = st.slider(
            "앞뒤 문맥 시간(±h)", min_value=1, max_value=48, value=6, step=1,
            key=f"ctx_hours_{narrative_id}_{chosen_seq}",
        )
    with ctx2:
        show_context = st.checkbox(
            "매칭 안 된 문맥 시점도 표시(끄면 매칭된 시점만)", value=True,
            key=f"show_ctx_{narrative_id}_{chosen_seq}",
        )

    raw, is_multi, anchors = da.fetch_evidence_raw(seq_events, context_hours=context_hours)
    if not raw:
        st.warning("원본 CSV에서 이 서사가 매칭한 farm/zone/시각 구간을 못 찾음(원본 파일 구조가 다를 수 있음).")
    else:
        if is_multi:
            st.info(
                f"이 서사는 {n_pairs}개의 (farm, zone)이 얽힌 비교형 서사라 "
                f"단일 절대시간축으로는 안 겹칩니다(각자 매칭 시각이 서로 다른 날짜일 수 있음) — "
                f"아래 그래프는 각 farm/zone을 **자기 매칭 시각 기준 상대시간(0=매칭 시점)**"
                f"으로 정렬해 겹쳐 그립니다. (최대 {da.MAX_EVIDENCE_ENTITIES}개 farm/zone만 표시)"
            )
        numeric_cols = da.available_numeric_columns(raw)
        relevant = [c for c in parsed["columns"] if c in numeric_cols]
        default_cols = relevant or numeric_cols[:3]
        plot_cols = st.multiselect(
            "그래프로 볼 컬럼 (서사 matcher가 실제로 쓰는 컬럼이 기본 선택됨)",
            numeric_cols, default=default_cols,
            key=f"plot_cols_{narrative_id}_{chosen_seq}",
        )

        with st.expander("이 구간의 원본 행 보기 (테이블별)"):
            raw_tabs = st.tabs(list(raw.keys()))
            for tab, (table_name, df) in zip(raw_tabs, raw.items()):
                with tab:
                    st.dataframe(df, use_container_width=True)

        if plot_cols:
            long_df = da.build_evidence_long_frame(
                raw, plot_cols, is_multi=is_multi, anchors=anchors, m_keys=da.matched_keys(seq_events),
            )
            if long_df.empty:
                st.warning("선택한 컬럼이 이 구간의 raw 데이터에 없습니다.")
            else:
                if not long_df["matched"].any():
                    st.warning(
                        "이 인스턴스가 가리키는 정확한 매칭 시점이 현재 raw CSV 스냅샷 범위 안에서 "
                        "안 보입니다 (farm별 14일 창의 경계 부근 — 코퍼스 빌드 당시 쓰인 원본 이력과 "
                        "지금 배포된 CSV 스냅샷 사이에 며칠의 경계 차이가 있는 것으로 보임, "
                        "주로 crossfarm/crosszone류 서사에서 관측됨). 아래는 문맥 데이터만 표시합니다."
                    )
                if not show_context:
                    long_df = long_df[long_df["matched"]]
                try:
                    import altair as alt

                    x_field, x_type, x_title = (
                        ("x_hours:Q", "Q", "매칭 시점 기준 상대시간 (h)")
                        if is_multi
                        else ("timestamp:T", "T", "시각")
                    )
                    zoom = alt.selection_interval(bind="scales", encodings=["x"])
                    charts = []
                    for col in plot_cols:
                        col_df = long_df[long_df["feature"] == col]
                        if col_df.empty:
                            continue
                        color_enc = (
                            alt.Color("farm_zone:N", legend=alt.Legend(title="farm/zone"))
                            if is_multi
                            else alt.value("#7aa6c2")
                        )
                        context_line = (
                            alt.Chart(col_df).mark_line(opacity=0.35 if is_multi else 0.55)
                            .encode(x=alt.X(x_field, title=x_title), y=alt.Y("value:Q", title=col), color=color_enc)
                        )
                        matched_pts = (
                            alt.Chart(col_df[col_df["matched"]])
                            .mark_point(filled=True, size=70)
                            .encode(
                                x=alt.X(x_field, title=x_title), y="value:Q",
                                color=color_enc if is_multi else alt.value("#e4572e"),
                                tooltip=["timestamp:T", "farm_zone:N", "value:Q"],
                            )
                        )
                        layers = [context_line, matched_pts]
                        table_thresholds = [t for t in parsed["thresholds"] if t["column"] == col]
                        if table_thresholds:
                            rules = alt.Chart(pd.DataFrame(table_thresholds)).mark_rule(
                                strokeDash=[4, 4], color="red"
                            ).encode(y="value:Q")
                            layers.append(rules)
                        chart = alt.layer(*layers).add_params(zoom).properties(height=170)
                        charts.append(chart)
                    if charts:
                        st.altair_chart(
                            alt.vconcat(*charts).resolve_scale(x="shared"), use_container_width=True,
                        )
                        st.caption(
                            "굵은 점 = 서사가 실제로 근거로 쓴 raw 시점(matched); "
                            "옅은 선 = 앞뒤 문맥(매칭 안 됨) — 위 체크박스로 문맥 표시를 끌 수 있습니다. "
                            "그래프 중 하나를 드래그해 확대/이동하면 나머지도 같은 시간축으로 같이 움직입니다."
                            + ("  빨간 점선 = matcher가 실제로 쓰는 임계값." if any(parsed["thresholds"]) else "")
                        )
                except ImportError:
                    st.line_chart(long_df.pivot_table(index="timestamp", columns="feature", values="value"))

with tab_candidates:
    st.markdown(
        "실제 카탈로그에는 아직 반영 안 된 자동생성 후보(dry-run 검증만 통과) — "
        "여기는 시퀀스 조회가 안 됩니다(실제로 빌드된 적 없음). "
        "검토 후 반영하기로 확정되면 `scripts/generate_online2_narrative_catalog.py`에 "
        "spec으로 옮기고 원시 빌드를 다시 돌려야 '활성 서사' 탭에서 조회 가능해집니다."
    )
    candidates = _auto_candidates()
    if candidates.empty:
        st.info(f"자동생성 리포트를 못 찾음: {da.AUTO_EXPANSION_REPORT}")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            cat_filter2 = st.multiselect("카테고리 필터 (관계 유형)", sorted(candidates["category"].unique()), key="auto_cat")
        with col_b:
            tier_options = sorted(candidates["caption_tier"].unique()) if "caption_tier" in candidates.columns else []
            tier_filter = st.multiselect("캡션 계층 (SensorLM 3단계)", tier_options, key="auto_tier")
        view2 = candidates
        if cat_filter2:
            view2 = view2[view2["category"].isin(cat_filter2)]
        if tier_filter and "caption_tier" in view2.columns:
            view2 = view2[view2["caption_tier"].isin(tier_filter)]
        st.write(f"{len(view2)}개 후보")
        cols = ["narrative_id", "category", "name", "matcher", "trigger_row_count", "unique_instance_count", "farms", "max_events"]
        if "caption_tier" in view2.columns:
            cols.insert(2, "caption_tier")
        st.dataframe(view2[cols], use_container_width=True, height=600)

        st.divider()
        st.markdown("#### 후보 하나 골라 matcher 로직 상세 보기")
        pick_id = st.selectbox(
            "narrative_id 선택", view2["narrative_id"].tolist(),
            format_func=lambda nid: f"{nid} — {view2.set_index('narrative_id').loc[nid, 'name']}",
            # 필터를 바꾸면 view2가 줄어드는데 위젯 key가 고정이면 이전 선택값이
            # session_state에 남아 새 view2에 없는 narrative_id를 그대로 참조해
            # IndexError로 앱이 죽는 문제가 있었다(실측 확인) -- 필터 조합을 key에
            # 반영해 필터가 바뀌면 selectbox도 새로 초기화되게 한다.
            key=f"auto_pick_{'_'.join(cat_filter2)}_{'_'.join(tier_filter)}",
        )
        matched_rows = view2[view2["narrative_id"] == pick_id]
        if matched_rows.empty:
            st.stop()
        pick_row = matched_rows.iloc[0]
        pparsed = da.parse_matcher(str(pick_row["matcher"]))
        st.markdown(f"**판정 규칙**: {pparsed['description']}")
        p1, p2, p3 = st.columns(3)
        p1.metric("관련 컬럼", ", ".join(pparsed["columns"]) or "—")
        p2.metric("window (시간)", pparsed["window_hours"] if pparsed["window_hours"] else "단일 시점")
        p3.metric("실측 unique_instance_count", pick_row["unique_instance_count"])
        if pparsed["thresholds"]:
            st.dataframe(pd.DataFrame(pparsed["thresholds"]), use_container_width=True, hide_index=True)
        st.caption("이 탭은 아직 실제 빌드가 안 돼서 raw 데이터/시퀀스 조회는 안 됩니다 — 매칭 규칙 자체만 확인.")
