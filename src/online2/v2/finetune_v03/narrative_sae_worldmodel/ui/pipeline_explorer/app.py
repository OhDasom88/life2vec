"""Pipeline Explorer (Streamlit) — 원시데이터 → 토큰 → 시퀀스 → encoder → SAE까지
전 과정을 하나의 화면에서 유기적으로 조회한다.

실행:
  conda run -n life2vec streamlit run \
    src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/pipeline_explorer/app.py

UI 화면 상태는 정본이 아니다(계획서 §15) — 이 앱은 다른 스크립트가 이미 만든
산출물(부분 코퍼스, 6-arm 체크포인트, SAE 비교 리포트, 인덱스 parquet)을
조회·시각화만 한다. 사전 준비가 필요하다:
  1. scripts/online2_v2/build_pipeline_explorer_index.py  (시퀀스 인덱스)
  2. scripts/online2_v2/train_and_save_pipeline_explorer_sae.py  (SAE 저장, 3탭용)
둘 다 이미 실행돼 있으면 이 앱은 그 결과만 읽는다.

4개 탭:
  1. 원시데이터 ↔ 토큰 ↔ 시퀀스 추적
  2. 인코딩 트레이스 (layer별 + MLM/SOP 변환, dead ratio 포함)
  3. 개념/시퀀스 유사도 (raw activation 코사인 유사도 + SAE 코드 유사도)
  4. 학습 버전 브라우저 (원본 + 6개 ablation arm 비교)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts" / "online2_v2") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "online2_v2"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data_access as da  # noqa: E402
from cache_stage_a_event_embeddings import (  # noqa: E402
    construct_target_window,
    load_abspos_reference,
    load_events_frame,
    load_frozen_encoder,
    window_to_tensors,
)
from pilot_sae_layer_decoder_random_comparison import encode_all_positions  # noqa: E402
from pilot_sae_narrative_selected_activations import build_farm_event_lists  # noqa: E402
from src.data_new.vocabulary import RegistryVocabulary  # noqa: E402
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.model import SparseAutoencoder  # noqa: E402

POSITIONS = ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5", "mlm_transform", "sop_transform"]
ROLE_COLORS = {
    "feature_identity": "#2563eb",
    "value_abs": "#16a34a",
    "value_abs_combined": "#16a34a",
    "value_global": "#ca8a04",
    "value_global_combined": "#ca8a04",
    "value_farm": "#dc2626",
    "value_farm_combined": "#dc2626",
    "quality": "#7c3aed",
    "sep": "#9ca3af",
    "meta": "#9ca3af",
    "derived": "#0891b2",
    "literal": "#be185d",
    "circular": "#be185d",
}


@st.cache_data
def load_index() -> pd.DataFrame:
    path = _REPO_ROOT / "outputs/online2/sae_pilot/pipeline_explorer_index.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource
def load_vocab_and_abspos():
    vocab = RegistryVocabulary(
        registry_path=str(_REPO_ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json"),
        registry_version="v2",
    )
    abspos_reference = load_abspos_reference(_REPO_ROOT / "outputs/online2/v2_build/abspos_reference.json")
    return vocab, abspos_reference


@st.cache_resource
def load_farm_events_cached(farm_ids_key: tuple[str, ...]):
    events_df = load_events_frame(_REPO_ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet", set(farm_ids_key))
    return build_farm_event_lists(events_df, set(farm_ids_key))


@st.cache_resource
def load_encoder_cached(run_name: str):
    vocab, _ = load_vocab_and_abspos()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = da.checkpoint_path_for_run(run_name)
    model, hparams, _ = load_frozen_encoder(ckpt_path, vocab, device)
    return model, device


@st.cache_resource
def load_persisted_sae_cached(run_name: str, position: str) -> SparseAutoencoder | None:
    path = _REPO_ROOT / f"outputs/online2/sae_pilot/pipeline_explorer_saes/{run_name}__{position}.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(
        input_dim=payload["input_dim"], dict_size=payload["dict_size"],
        sparsity_mode=payload["sparsity_mode"], top_k=payload["top_k"], tied_weights=payload["tied_weights"],
    )
    sae.load_state_dict(payload["state_dict"])
    sae.eval()
    return sae


def encode_sequence(run_name: str, farm_id: str, target_event_id: str) -> dict[str, np.ndarray] | None:
    """한 시퀀스의 타깃 이벤트 span을 8개 지점(6 layer + MLM/SOP 변환)에서 pooled mean으로 인코딩."""
    vocab, abspos_reference = load_vocab_and_abspos()
    farm_events = load_farm_events_cached((farm_id,))
    events = farm_events.get(farm_id)
    if not events:
        return None
    local_idx = next((i for i, ev in enumerate(events) if ev.event_id == target_event_id), None)
    if local_idx is None:
        return None
    model, device = load_encoder_cached(run_name)
    window = construct_target_window(events, local_idx, max_length=1024)
    case_t0 = events[0].timestamp
    x, mask, _ = window_to_tensors(
        events, window, vocab=vocab, abspos_reference=abspos_reference, case_t0=case_t0, max_length=1024
    )
    per_position = encode_all_positions(model, [x], [mask], [(window.target_token_start, window.target_token_end)], device)
    return {pos: per_position[pos][0][0] for pos in POSITIONS}


def render_sequence_trace_tab(index_df: pd.DataFrame) -> None:
    st.subheader("1. 원시데이터 ↔ 토큰 ↔ 시퀀스 추적")
    st.caption(
        "narrative가 raw 이벤트를 골라 시퀀스로 만든 결과를 그대로 보여준다 — 왼쪽은 그 시퀀스의 토큰 "
        "(역할별 색), 오른쪽은 각 이벤트가 나온 원본 raw CSV 행(E_environment/R_rootzone/A_actuator/G_growth)."
    )
    if index_df.empty:
        st.warning("인덱스가 없다 — 먼저 `scripts/online2_v2/build_pipeline_explorer_index.py`를 실행하세요.")
        return

    narratives = sorted(index_df["narrative_id"].unique())
    narrative_id = st.selectbox("narrative_id", narratives, key="trace_narrative")
    seq_options = (
        index_df[index_df["narrative_id"] == narrative_id][["sequence_id", "farm_id"]]
        .drop_duplicates(subset=["sequence_id"])
        .sort_values("sequence_id")
    )
    seq_ids = seq_options["sequence_id"].tolist()
    farm_by_seq = dict(zip(seq_options["sequence_id"], seq_options["farm_id"]))
    seq_label = st.selectbox(
        "sequence_id",
        seq_ids,
        format_func=lambda sid: f"{sid}  (farm={farm_by_seq[sid]})",
        key="trace_sequence",
    )
    st.session_state["selected_sequence_id"] = seq_label

    seq_rows = index_df[index_df["sequence_id"] == seq_label].sort_values("event_position")
    st.session_state["selected_farm_id"] = seq_rows.iloc[0]["farm_id"]
    st.session_state["selected_target_event_id"] = seq_rows.iloc[-1]["event_id"]

    st.markdown(f"**{len(seq_rows)}개 이벤트** (event_position 순)")
    for _, row in seq_rows.iterrows():
        with st.expander(
            f"event_position={row['event_position']}  event_id={row['event_id']}  "
            f"({row['event_kind']}, {row['START_DATE']})",
            expanded=(row["event_position"] == seq_rows["event_position"].max()),
        ):
            col_tok, col_raw = st.columns([3, 2])
            with col_tok:
                st.markdown("**토큰 (색 = role)**")
                entries = da.parse_token_trace(row["SENTENCE"], row["measurement_group_ids"], row["token_roles"])
                html_parts = []
                for e in entries:
                    color = ROLE_COLORS.get(e.role, "#374151")
                    html_parts.append(
                        f'<span title="role={e.role} group={e.group_id}" '
                        f'style="background:{color}22;border:1px solid {color};border-radius:4px;'
                        f'padding:1px 4px;margin:2px;display:inline-block;font-size:0.85em;">{e.token}</span>'
                    )
                st.markdown(" ".join(html_parts), unsafe_allow_html=True)
                role_legend = ", ".join(f"{r}" for r in sorted({e.role for e in entries}))
                st.caption(f"등장한 role: {role_legend}")
            with col_raw:
                st.markdown("**원본 raw CSV 행**")
                raw = da.lookup_raw_rows(row["farm_id"], str(row["zone_id"]), str(row["START_DATE"]), window_hours=0)
                if not raw:
                    st.caption("(이 이벤트 시각/zone에 정확히 일치하는 raw 행을 못 찾음 — G_growth처럼 zone 구분이 없거나 timestamp 단위가 다를 수 있음)")
                for table, df in raw.items():
                    st.caption(table)
                    st.dataframe(df, use_container_width=True, hide_index=True)


def render_encoding_trace_tab() -> None:
    st.subheader("2. 인코딩 트레이스 — layer별 + MLM/SOP 변환, 각 단계 dead ratio")
    st.caption(
        "선택된 시퀀스(1번 탭에서 고른 것)를 실제로 순전파해서 6개 encoder layer + MLM/SOP 디코더 변환, "
        "총 8개 지점의 pooled activation을 뽑고, 각 지점의 크기(norm)와 (미리 계산해 둔) dead_feature_ratio를 같이 보여준다."
    )
    farm_id = st.session_state.get("selected_farm_id")
    target_event_id = st.session_state.get("selected_target_event_id")
    if not farm_id:
        st.info("먼저 1번 탭에서 시퀀스를 하나 골라주세요.")
        return
    st.caption(f"현재 선택: farm={farm_id}, target_event={target_event_id}")

    run_names = ["original"] + list(da.ABLATION_ARMS)
    run_name = st.selectbox("체크포인트(run)", run_names, key="encoding_run")

    if st.button("이 시퀀스 인코딩하기", key="encode_button"):
        with st.spinner(f"{run_name} 인코더로 순전파 중..."):
            vecs = encode_sequence(run_name, farm_id, target_event_id)
        if vecs is None:
            st.error("이 이벤트를 해당 farm에서 못 찾았다 — 인덱스와 원본 파일이 어긋난 것일 수 있음.")
            return
        st.session_state["last_encoded"] = {"run_name": run_name, "vecs": vecs}

    encoded = st.session_state.get("last_encoded")
    if not encoded or encoded["run_name"] != run_name:
        st.caption("버튼을 눌러 인코딩을 실행하세요 (자동 실행 안 함 — GPU 순전파 비용 때문).")
        return

    norms = {pos: float(np.linalg.norm(v)) for pos, v in encoded["vecs"].items()}
    st.markdown("**단계별 pooled activation의 norm**")
    st.bar_chart(pd.Series(norms))

    st.markdown("**단계별 dead_feature_ratio (미리 계산된 리포트에서 조회)**")
    dead_by_position: dict[str, float | None] = {}
    if run_name == "original":
        report = da.load_layer_decoder_random_report()
        if report:
            trained = report["results"].get("trained", {})
            for pos in POSITIONS:
                dead_by_position[pos] = trained.get(pos, {}).get("sae", {}).get("dead_feature_ratio")
    else:
        report = da.load_ablation_report()
        if report:
            arm = report["results"].get(run_name, {})
            for pos in ["layer5", "mlm_transform", "sop_transform"]:
                dead_by_position[pos] = arm.get(pos, {}).get("sae", {}).get("dead_feature_ratio")
    if any(v is not None for v in dead_by_position.values()):
        st.bar_chart(pd.Series({k: v for k, v in dead_by_position.items() if v is not None}))
    else:
        st.caption("이 run에 대한 사전 계산된 dead_ratio 리포트가 없다.")


def render_similarity_tab(index_df: pd.DataFrame) -> None:
    st.subheader("3. 개념/시퀀스 유사도")
    st.caption("두 시퀀스를 골라 raw activation 코사인 유사도(항상 계산)와 SAE 코드 유사도(저장된 SAE가 있을 때만)를 비교한다.")
    if index_df.empty:
        st.warning("인덱스가 없다.")
        return

    run_names = ["original"] + list(da.ABLATION_ARMS)
    run_name = st.selectbox("체크포인트(run)", run_names, key="sim_run")
    position = st.selectbox("지점(position)", POSITIONS, key="sim_position")

    seq_choices = index_df[["sequence_id", "farm_id", "narrative_id"]].drop_duplicates(subset=["sequence_id"])
    seq_ids = seq_choices["sequence_id"].tolist()
    seq_label_by_id = {
        r["sequence_id"]: f"{r['sequence_id']} ({r['narrative_id']}, farm={r['farm_id']})"
        for r in seq_choices.to_dict("records")
    }

    col_a, col_b = st.columns(2)
    with col_a:
        seq_a = st.selectbox("시퀀스 A", seq_ids, format_func=lambda s: seq_label_by_id[s], key="sim_seq_a")
    with col_b:
        seq_b = st.selectbox("시퀀스 B", seq_ids, format_func=lambda s: seq_label_by_id[s], key="sim_seq_b")

    if st.button("유사도 계산", key="sim_button"):
        row_a = index_df[index_df["sequence_id"] == seq_a].sort_values("event_position").iloc[-1]
        row_b = index_df[index_df["sequence_id"] == seq_b].sort_values("event_position").iloc[-1]
        with st.spinner("두 시퀀스 인코딩 중..."):
            vecs_a = encode_sequence(run_name, row_a["farm_id"], row_a["event_id"])
            vecs_b = encode_sequence(run_name, row_b["farm_id"], row_b["event_id"])
        if vecs_a is None or vecs_b is None:
            st.error("인코딩 실패 — 인덱스와 원본 파일이 어긋난 것일 수 있음.")
            return
        a, b = vecs_a[position], vecs_b[position]
        raw_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        st.metric("Raw activation 코사인 유사도", f"{raw_cos:.4f}")

        sae = load_persisted_sae_cached(run_name, position)
        if sae is None:
            st.caption(
                f"저장된 SAE가 없다({run_name}/{position}) — "
                "`scripts/online2_v2/train_and_save_pipeline_explorer_sae.py`의 DEFAULT_COMBOS에 이 조합을 추가해서 실행하면 여기서 SAE 코드 유사도도 볼 수 있다."
            )
            return
        with torch.no_grad():
            code_a = sae.encode(torch.from_numpy(a).unsqueeze(0)).squeeze(0).numpy()
            code_b = sae.encode(torch.from_numpy(b).unsqueeze(0)).squeeze(0).numpy()
        sae_cos = float(np.dot(code_a, code_b) / (np.linalg.norm(code_a) * np.linalg.norm(code_b) + 1e-8))
        st.metric("SAE 코드 코사인 유사도", f"{sae_cos:.4f}")

        fired_a = set(np.nonzero(code_a)[0].tolist())
        fired_b = set(np.nonzero(code_b)[0].tolist())
        shared = fired_a & fired_b
        st.write(f"A에서 활성화된 feature: {sorted(fired_a)}")
        st.write(f"B에서 활성화된 feature: {sorted(fired_b)}")
        st.write(f"공통으로 활성화된 feature ({len(shared)}개): {sorted(shared)}")
        st.caption(
            "주의: 이 SAE는 evaluation.py의 reconstruction/sparsity 지표만 검증됐다 — "
            "여기 나온 feature 번호가 사람이 이해할 수 있는 '개념'이라고 자동으로 확정하지 않는다(sae/schemas.py의 상태 머신 참조)."
        )


def render_run_browser_tab() -> None:
    st.subheader("4. 학습 버전 브라우저")
    st.caption("원본 체크포인트 + 이번 세션에서 재학습한 6개 ablation arm을 config/지표 기준으로 비교한다.")
    runs = da.list_local_runs()
    if not runs:
        st.warning("run_manifest_v2.json을 가진 run이 없다.")
        return

    rows = []
    ablation_report = da.load_ablation_report()
    for r in runs:
        arm_metrics = {}
        if ablation_report and r["name"] in ablation_report.get("results", {}):
            arm_metrics = ablation_report["results"][r["name"]].get("layer5", {})
        rows.append({
            "run": r["name"],
            "final_step": r.get("final_step"),
            "batch_size": r.get("batch_size"),
            "sop_reverse": r.get("sop_reverse_probability"),
            "sop_shuffle": r.get("sop_shuffle_probability"),
            "best_val_loss": r.get("best_val_loss"),
            "layer5_rank99": arm_metrics.get("effective_rank", {}).get("pc_for_99pct"),
            "layer5_dead_ratio": arm_metrics.get("sae", {}).get("dead_feature_ratio"),
        })
    df = pd.DataFrame(rows).set_index("run")
    st.dataframe(df, use_container_width=True)

    st.markdown("**두 run 상세 비교**")
    names = df.index.tolist()
    col_a, col_b = st.columns(2)
    with col_a:
        run_a = st.selectbox("run A", names, key="diff_run_a")
    with col_b:
        run_b = st.selectbox("run B", names, index=min(1, len(names) - 1), key="diff_run_b")
    manifest_a = next(r for r in runs if r["name"] == run_a)
    manifest_b = next(r for r in runs if r["name"] == run_b)
    keys = sorted(set(manifest_a) | set(manifest_b))
    diff_rows = [
        {"key": k, run_a: str(manifest_a.get(k)), run_b: str(manifest_b.get(k))}
        for k in keys
        if k not in ("history_tail", "val_history_tail", "attempts") and manifest_a.get(k) != manifest_b.get(k)
    ]
    st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Pipeline Explorer", layout="wide")
    st.title("Pipeline Explorer — 원시데이터 → 토큰 → 시퀀스 → encoder → SAE")

    index_df = load_index()
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 원시↔토큰↔시퀀스", "2. 인코딩 트레이스", "3. 개념/시퀀스 유사도", "4. 학습 버전 브라우저",
    ])
    with tab1:
        render_sequence_trace_tab(index_df)
    with tab2:
        render_encoding_trace_tab()
    with tab3:
        render_similarity_tab(index_df)
    with tab4:
        render_run_browser_tab()


if __name__ == "__main__":
    main()
