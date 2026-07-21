"""§5.3 사람 검토 큐 UI (Streamlit) — narrative_grounding의 Data Grounding & Curation UI.

실행:
  conda run -n life2vec streamlit run \
    src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/data_grounding_curation/app.py

UI 화면 상태는 정본이 아니다(계획서 §15). 모든 결정은
``outputs/online2/narrative_grounding_review/decisions.jsonl``에 append-only로
기록된다 — 이 파일을 지우지 않는 한 재실행해도 이미 검토한 항목은 다시
큐에 뜨지 않는다(마지막 결정 기준). 렌더링 외 로직은 전부 ``review_batch.py``에
있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[7]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.ui.data_grounding_curation.review_batch import (  # noqa: E402
    append_decision,
    build_batch_queue,
    distinct_farms,
    load_corpus,
    load_decisions,
    make_decision_record,
    rows_for_farm,
)

BUILD_DIR = _REPO_ROOT / "outputs" / "online2" / "build-v8-active80-r3"
DECISIONS_PATH = _REPO_ROOT / "outputs" / "online2" / "narrative_grounding_review" / "decisions.jsonl"


@st.cache_resource(show_spinner="online2 코퍼스 로딩 중 (최초 1회, 967K sequence)...")
def _cached_corpus():
    return load_corpus(BUILD_DIR)


@st.cache_data(show_spinner=False)
def _cached_farms(_table) -> list[str]:
    return distinct_farms(_table)


def main() -> None:
    st.set_page_config(page_title="narrative_grounding 검토 큐", layout="wide")
    st.title("Data Grounding & Curation — §5.3 사람 검토 큐")
    st.caption(
        "UI 화면 상태는 정본이 아닙니다. 모든 결정은 "
        f"`{DECISIONS_PATH.relative_to(_REPO_ROOT)}`에 append-only로 기록됩니다."
    )

    if not BUILD_DIR.exists():
        st.error(f"빌드 산출물이 없습니다: {BUILD_DIR}")
        return

    catalog, table = _cached_corpus()
    farms = _cached_farms(table)

    with st.sidebar:
        st.header("배치 설정")
        farm_id = st.selectbox("farm_id", farms, index=0)
        per_template = st.slider("템플릿당 샘플 수", 1, 10, 3)
        seed = st.number_input("샘플링 seed", value=0, step=1)
        low_freq_threshold = st.number_input("저빈도 개념 임계값(count)", value=50, step=10)
        load_clicked = st.button("배치 로드 / 다시 샘플링", type="primary")

    batch_key = (farm_id, per_template, seed, low_freq_threshold)
    if st.session_state.get("batch_key") != batch_key or load_clicked:
        with st.spinner("narrative 생성 · 반례 탐지 · 검토 큐 구성 중..."):
            farm_rows = rows_for_farm(table, farm_id)
            queue, stats = build_batch_queue(
                farm_rows,
                catalog,
                per_template=per_template,
                seed=seed,
                low_frequency_threshold=low_freq_threshold,
            )
        st.session_state["batch_key"] = batch_key
        st.session_state["queue"] = queue
        st.session_state["stats"] = stats

    queue = st.session_state["queue"]
    stats = st.session_state["stats"]
    decisions = load_decisions(DECISIONS_PATH)
    pending = [entry for entry in queue if entry.narrative.narrative_instance_id not in decisions]

    metric_cols = st.columns(4)
    metric_cols[0].metric("farm 전체 인스턴스", stats["farm_row_count"])
    metric_cols[1].metric("이번 배치 샘플", stats["sampled_count"])
    metric_cols[2].metric("검토 큐(기준에 걸린 것)", stats["queue_length"])
    metric_cols[3].metric("검토 대기", len(pending))

    if not pending:
        st.success("이 배치의 검토 큐가 비었습니다 — 전부 검토했거나 걸린 항목이 없습니다.")
        return

    entry = pending[0]
    narrative = entry.narrative
    window = narrative.supporting_windows[0] if narrative.supporting_windows else None

    st.subheader(f"[{narrative.narrative_instance_id}]  priority_score={entry.priority_score:.1f}")
    st.warning(
        "검토 사유: "
        + " / ".join(f"**{reason.code}** ({reason.detail})" for reason in entry.reasons)
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**observation**")
        st.write(narrative.observation)
        st.markdown("**derived_state**")
        st.write(narrative.derived_state)
        st.markdown("**interpretation** _(템플릿 저자 저작, 이 인스턴스에서 재확인된 것 아님)_")
        st.write(narrative.interpretation)
        st.markdown("**confidence**")
        st.json(narrative.confidence)
        st.markdown(
            f"**causal_status** = `{narrative.causal_status}` · "
            f"**recommendation_status** = `{narrative.recommendation_status}` · "
            f"**sequence_recommendation** = `{narrative.sequence_recommendation}`"
        )
    with right:
        if window is not None:
            st.markdown("**근거 window (supporting)**")
            st.write(f"farm={window.farm_ids} zone={window.zone_ids}")
            st.write(f"{window.start_timestamp} ~ {window.end_timestamp}")
            st.write(f"event_views={window.event_views}, quality_flags={window.quality_flags}")
        if narrative.contradicting_windows:
            st.markdown(f"**반례 window (contradicting, {len(narrative.contradicting_windows)}개)**")
            for contradicting_window in narrative.contradicting_windows[:5]:
                st.write(
                    f"- 템플릿 {contradicting_window.narrative_template_id} · "
                    f"{contradicting_window.start_timestamp} ~ {contradicting_window.end_timestamp}"
                )
        else:
            st.markdown("**반례 window (contradicting)**: 없음")

    st.text_input("검토자", value=st.session_state.get("reviewer_name", ""), key="reviewer_name")

    action_cols = st.columns(3)
    if action_cols[0].button("ACCEPT", type="primary", use_container_width=True):
        append_decision(DECISIONS_PATH, make_decision_record(entry, "ACCEPT", st.session_state["reviewer_name"]))
        st.rerun()
    if action_cols[1].button("REJECT", use_container_width=True):
        append_decision(DECISIONS_PATH, make_decision_record(entry, "REJECT", st.session_state["reviewer_name"]))
        st.rerun()
    if action_cols[2].button("SKIP (나중에)", use_container_width=True):
        append_decision(DECISIONS_PATH, make_decision_record(entry, "SKIP", st.session_state["reviewer_name"]))
        st.rerun()


if __name__ == "__main__":
    main()
