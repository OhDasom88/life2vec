"""Data Grounding & Curation UI (Streamlit) — narrative_grounding의 §5.1/§5.3/§5.4 화면.

실행:
  conda run -n life2vec streamlit run \
    src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/data_grounding_curation/app.py

UI 화면 상태는 정본이 아니다(계획서 §15). §5.3 탭의 모든 결정은
``outputs/online2/narrative_grounding_review/decisions.jsonl``에 append-only로
기록된다 — 이 파일을 지우지 않는 한 재실행해도 이미 검토한 항목은 다시
큐에 뜨지 않는다(마지막 결정 기준). §5.1 탭은 그 자체로는 아무것도 디스크에
기록하지 않지만, "REVIEW/QUARANTINE을 §5.3 큐로 보내기" 버튼을 누르면
``narrative_grounding.search_to_review``를 거쳐 §5.3 탭의 세션 내 큐에
편입된다 — 실제 기록(append)은 그 항목을 §5.3 탭에서 ACCEPT/REJECT/SKIP할
때 일어난다. §5.4 탭은 그 decisions.jsonl을 읽기만 하는 조회 전용이다.
렌더링 외 로직은 전부 ``review_batch.py``/``narrative_grounding.*``에 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[7]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.decision_metrics import (  # noqa: E402
    summarize_decisions,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.search_to_review import (  # noqa: E402
    queue_from_candidates,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.text_to_window import (  # noqa: E402
    Qwen3EmbeddingProvider,
    TemplateEmbeddingIndex,
    search_text_to_window_over_table,
)
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

_DECISION_BADGE = {
    "AUTO_ACCEPT_CANDIDATE": "🟢",
    "REVIEW": "🟡",
    "QUARANTINE": "🟠",
    "REJECT": "🔴",
}


@st.cache_resource(show_spinner="online2 코퍼스 로딩 중 (최초 1회, 967K sequence)...")
def _cached_corpus():
    return load_corpus(BUILD_DIR)


@st.cache_data(show_spinner=False)
def _cached_farms(_table) -> list[str]:
    return distinct_farms(_table)


@st.cache_resource(show_spinner="Qwen3-Embedding-0.6B 로딩 및 80개 템플릿 임베딩 중 (최초 1회)...")
def _cached_provider_and_index(_catalog):
    provider = Qwen3EmbeddingProvider()
    index = TemplateEmbeddingIndex(_catalog, provider)
    return provider, index


def _render_review_queue_tab(catalog, table, farms: list[str]) -> None:
    with st.sidebar:
        st.header("§5.3 배치 설정")
        farm_id = st.selectbox("farm_id", farms, index=0, key="review_farm_id")
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
    # §5.1 탭에서 "검토 큐로 보내기"를 누른 항목(narrative_instance_id로 중복 제거).
    # 배치를 새로 불러와도 사라지지 않는다 — 독립적인 큐다.
    search_queue = st.session_state.setdefault("search_queue_entries", {})
    combined_by_id = {entry.narrative.narrative_instance_id: entry for entry in queue}
    combined_by_id.update(search_queue)  # 같은 항목이 양쪽에 있으면 §5.1 기원 쪽을 우선 표시
    all_entries = sorted(combined_by_id.values(), key=lambda entry: entry.priority_score, reverse=True)

    decisions = load_decisions(DECISIONS_PATH)
    pending = [entry for entry in all_entries if entry.narrative.narrative_instance_id not in decisions]

    metric_cols = st.columns(5)
    metric_cols[0].metric("farm 전체 인스턴스", stats["farm_row_count"])
    metric_cols[1].metric("이번 배치 샘플", stats["sampled_count"])
    metric_cols[2].metric("배치 큐(기준에 걸린 것)", stats["queue_length"])
    metric_cols[3].metric("§5.1에서 보낸 큐", len(search_queue))
    metric_cols[4].metric("검토 대기(합산)", len(pending))

    if not pending:
        st.success("검토 큐가 비었습니다 — 전부 검토했거나 걸린 항목이 없습니다.")
        return

    entry = pending[0]
    narrative = entry.narrative
    window = narrative.supporting_windows[0] if narrative.supporting_windows else None

    origin = (
        f"§5.1 검색 (판정: {entry.grounding_search_decision})"
        if entry.grounding_search_decision is not None
        else "§5.3 배치 샘플링"
    )
    st.subheader(f"[{narrative.narrative_instance_id}]  priority_score={entry.priority_score:.1f}")
    st.caption(f"출처: {origin}")
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


def _render_search_tab(catalog, table) -> None:
    st.caption(
        "임의 서사 텍스트로 근거 window를 검색합니다. 조회 전용이며 아무것도 "
        "기록하지 않습니다. `F######`/`zone N` 형식으로 farm·zone을 언급하면 "
        "검색이 그걸로 후보 풀을 먼저 좁혀 훨씬 빨라집니다."
    )

    query = st.text_area(
        "서사 질의",
        value=st.session_state.get("search_query", "F130230 zone 1에서 실내온도가 급격히 점프했다"),
        key="search_query",
        height=80,
    )
    col1, col2, col3 = st.columns(3)
    top_k_templates = col1.slider("top_k_templates", 1, 10, 3, key="search_top_k")
    max_windows_per_template = col2.slider("템플릿당 최대 window", 1, 50, 10, key="search_max_windows")
    max_candidate_rows = col3.number_input(
        "후보 행 상한(응답성 vs 정확도)", value=5000, step=1000, key="search_max_rows"
    )

    if st.button("검색", type="primary"):
        provider, template_index = _cached_provider_and_index(catalog)
        with st.spinner("질의 임베딩 · 템플릿 랭킹 · window 검색 중..."):
            candidates = search_text_to_window_over_table(
                query,
                catalog=catalog,
                template_index=template_index,
                provider=provider,
                table=table,
                top_k_templates=top_k_templates,
                max_windows_per_template=max_windows_per_template,
                max_candidate_rows=max_candidate_rows,
            )
        st.session_state["search_results"] = candidates

    candidates = st.session_state.get("search_results")
    if candidates is None:
        return
    if not candidates:
        st.info("후보를 찾지 못했습니다 (질의와 매칭되는 템플릿·farm 조합이 없음).")
        return

    sendable = [c for c in candidates if c.decision in {"REVIEW", "QUARANTINE"}]
    header_cols = st.columns([3, 2])
    header_cols[0].write(f"{len(candidates)}개 후보 (결합 점수 내림차순)")
    if header_cols[1].button(
        f"REVIEW/QUARANTINE {len(sendable)}건을 §5.3 큐로 보내기",
        disabled=not sendable,
    ):
        new_entries = queue_from_candidates(candidates)
        search_queue = st.session_state.setdefault("search_queue_entries", {})
        added = sum(
            1
            for e in new_entries
            if e.narrative.narrative_instance_id not in search_queue
        )
        search_queue.update({e.narrative.narrative_instance_id: e for e in new_entries})
        # main()이 검토 큐 탭을 검색 탭보다 먼저 렌더링하므로, 이 rerun 없이는
        # 검토 큐 탭이 "이번 실행에서 막 갱신된" search_queue_entries를 반영하지
        # 못하고 한 번의 rerun만큼 지연된 상태를 보여준다(관찰로 확인한 버그).
        # st.success는 rerun 이후에도 보이도록 세션에 잠깐 남겨 다음 렌더에서 띄운다.
        st.session_state["search_queue_flash"] = (
            f"{len(new_entries)}건 중 신규 {added}건을 §5.3 큐에 추가했습니다 "
            f"(AUTO_ACCEPT_CANDIDATE/REJECT {len(candidates) - len(new_entries)}건은 "
            "설계상 사람 검토 없이 제외). '§5.3 사람 검토 큐' 탭에서 확인하세요."
        )
        st.rerun()

    if st.session_state.get("search_queue_flash"):
        st.success(st.session_state.pop("search_queue_flash"))

    for candidate in candidates:
        window = candidate.narrative.supporting_windows[0]
        badge = _DECISION_BADGE.get(candidate.decision, "")
        with st.expander(
            f"{badge} {candidate.decision}  ·  combined={candidate.combined_score:.3f}  ·  "
            f"[{window.narrative_template_id}] {candidate.narrative.narrative_instance_id}"
        ):
            st.write(
                f"template_similarity={candidate.template_similarity:.3f}, "
                f"structured_score={candidate.structured_score:.3f}"
            )
            st.caption(candidate.decision_reason)
            st.markdown("**observation**")
            st.write(candidate.narrative.observation)
            st.markdown("**근거 window**")
            st.write(f"farm={window.farm_ids} zone={window.zone_ids}")
            st.write(f"{window.start_timestamp} ~ {window.end_timestamp}")


def _render_report_tab() -> None:
    st.caption(
        "decisions.jsonl(§5.3 탭에서 쌓인 사람 결정)을 §5.4 evaluation.expert_acceptance_rate에 "
        "실시간으로 연결한 리포트입니다. ACCEPT/REJECT를 누를 때마다 이 탭도 즉시 갱신됩니다."
    )

    decisions = load_decisions(DECISIONS_PATH)
    report = summarize_decisions(decisions)

    metric_cols = st.columns(4)
    metric_cols[0].metric("전체 결정", report["n_total"])
    metric_cols[1].metric("ACCEPT", report["n_accept"])
    metric_cols[2].metric("REJECT", report["n_reject"])
    metric_cols[3].metric("SKIP(분모 제외)", report["n_skipped"])

    if report["n_decided"] == 0:
        st.info("아직 ACCEPT/REJECT 결정이 없습니다 — §5.3 탭에서 검토를 진행하면 여기 반영됩니다.")
        return

    st.metric("overall_expert_acceptance_rate", f"{report['overall_expert_acceptance_rate']:.3f}")

    st.markdown("**검토 사유 코드별 승인율** — `review_queue._REASON_WEIGHTS` 재보정 근거")
    st.caption(
        "승인율이 높은 사유는 사람이 대체로 '문제 없다'고 판단했다는 뜻(과잉 플래그 가능성, "
        "가중치를 낮출 후보). 승인율이 낮은 사유는 대체로 '문제 있다'는 뜻(가중치 유지·강화 후보)."
    )
    by_reason = report["by_reason_code"]
    if by_reason:
        rows = sorted(by_reason.items(), key=lambda item: item[1]["acceptance_rate"])
        st.dataframe(
            {
                "reason_code": [code for code, _ in rows],
                "acceptance_rate": [round(v["acceptance_rate"], 3) for _, v in rows],
                "n": [v["n"] for _, v in rows],
            },
            hide_index=True,
        )
    else:
        st.write("검토 사유가 기록된 결정이 없습니다.")

    with st.expander("연결되지 않은 §5.4 지표"):
        for metric_name, reason in report["not_connected"].items():
            st.write(f"**{metric_name}**: {reason}")


def main() -> None:
    st.set_page_config(page_title="narrative_grounding — Data Grounding & Curation", layout="wide")
    st.title("Data Grounding & Curation")
    st.caption(
        "UI 화면 상태는 정본이 아닙니다. §5.3 탭의 결정만 "
        f"`{DECISIONS_PATH.relative_to(_REPO_ROOT)}`에 append-only로 기록됩니다."
    )

    if not BUILD_DIR.exists():
        st.error(f"빌드 산출물이 없습니다: {BUILD_DIR}")
        return

    catalog, table = _cached_corpus()
    farms = _cached_farms(table)

    review_tab, search_tab, report_tab = st.tabs(
        ["§5.3 사람 검토 큐", "§5.1 텍스트 → window 검색", "§5.4 리포트"]
    )
    with review_tab:
        _render_review_queue_tab(catalog, table, farms)
    with search_tab:
        _render_search_tab(catalog, table)
    with report_tab:
        _render_report_tab()


if __name__ == "__main__":
    main()
