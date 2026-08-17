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

7개 탭:
  1. 원시데이터 ↔ 토큰 ↔ 시퀀스 추적 (narrative 카탈로그 시퀀스)
  2. 인코딩 트레이스 (layer별 + MLM/SOP 변환, dead ratio 포함)
  3. 개념/시퀀스 유사도 (raw activation 코사인 유사도 + SAE 코드 유사도)
  4. 학습 버전 브라우저 (원본 + 6개 ablation arm 비교)
  5. Entity Summary (finetune 진단 케이스: 정상/비정상 concept space 투영,
     OOF 선형분리 AUROC, 케이스별 유사사례, narrative TCAV 즉석 계산)
  6. Vocab Concept Space
  7. Finetune 케이스 토큰화 (case_manifest 케이스 시퀀스를 선택한 run의 사전학습
     vocab 기준으로 토큰화 + UNK 비율 + raw 매칭까지 확인)
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
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
import cache_stage_a_event_embeddings as stage_a_module  # noqa: E402
from cache_stage_a_event_embeddings import (  # noqa: E402
    case_events_from_frame,
    construct_target_window,
    load_abspos_reference,
    load_events_frame,
    load_frozen_encoder,
    window_to_tensors,
    zone_present,
)
from pilot_sae_layer_decoder_random_comparison import encode_all_positions  # noqa: E402
from pilot_sae_narrative_selected_activations import build_farm_event_lists  # noqa: E402
from src.data_new.vocabulary import RegistryVocabulary  # noqa: E402
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.model import SparseAutoencoder  # noqa: E402
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.projection import (  # noqa: E402
    project_activations,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.metadata import (  # noqa: E402
    DISCLAIMER as PROJECTION_DISCLAIMER,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.overlap_check import (  # noqa: E402
    split_overlap_in_projection_space,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.concept_tcav.narrative_concept_probe import (  # noqa: E402
    run_narrative_tcav,
)
from src.online2.v2.finetune_v03.llm_evidence.nearest_examples import (  # noqa: E402
    nearest_labeled_examples,
)
from src.online2.v2.finetune_v03.llm_evidence.evidence_bundle import (  # noqa: E402
    build_case_evidence,
)
from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03  # noqa: E402
from src.online2.v2.finetune_v03.dataset import (  # noqa: E402
    DiagnosisEventDatasetV03,
    collate_diagnosis_batch_v03,
    load_label_map,
    normal_class_id_from_map,
)
from src.online2.v2.finetune_v03.token_saliency import token_ixg_for_event_v03  # noqa: E402

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
    "context": "#0891b2",  # ZONE_LOCAL/FARM_LOCAL/NARRATIVE -- token_saliency.classify_token_type
}


def run_select_label(run_name: str) -> str:
    """체크포인트 selectbox용 라벨 — 생성 시점을 옆에 붙여 선후 비교 가능하게."""
    created = da.run_created_at(run_name)
    return f"{run_name} ({created})" if created else f"{run_name} (생성시점 미상)"


def scatter_with_tooltip(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    tooltip: list[str],
    size: int = 60,
) -> None:
    """마우스 오버 시 tooltip 전체(라벨/좌표 등)를 보여주는 산점도.

    `st.scatter_chart`는 tooltip을 색상 컬럼 하나로 제한한다 — 어떤 점이
    어떤 token/sequence/case인지(값) 안 보인다는 사용자 지적(2026-07-24)으로
    altair 직접 사용으로 교체.
    """
    chart = (
        alt.Chart(df)
        .mark_circle(size=size)
        .encode(
            x=alt.X(x, type="quantitative"),
            y=alt.Y(y, type="quantitative"),
            color=alt.Color(color, type="nominal"),
            tooltip=tooltip,
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


@st.cache_data
def load_narrative_catalog() -> dict[str, tuple[str, str]]:
    """narrative_id -> (narrative_name_ko, purpose), from the human-authored catalog
    (`generate_online2_narrative_catalog.py`'s output) -- ID alone (e.g. "A05") isn't
    readable, this is the lookup that makes it so."""
    path = _REPO_ROOT / "datasets/agrichallenge/online2/online2_narrative_catalog.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        str(r["narrative_id"]): (str(r["narrative_name_ko"]), str(r["purpose"]))
        for _, r in df.iterrows()
    }


@st.cache_data
def load_narrative_catalog_full() -> dict[str, tuple[str, str, str, str, str]]:
    """narrative_id -> (name_ko, purpose, category, expected_pattern, agronomic_interpretation) --
    richer version of load_narrative_catalog() for tab 1's per-narrative explanation box."""
    path = _REPO_ROOT / "datasets/agrichallenge/online2/online2_narrative_catalog.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        str(r["narrative_id"]): (
            str(r["narrative_name_ko"]),
            str(r["purpose"]),
            str(r["category"]),
            str(r["expected_pattern"]),
            str(r["agronomic_interpretation"]),
        )
        for _, r in df.iterrows()
    }


def narrative_label(narrative_id: str, catalog: dict[str, tuple[str, str]]) -> str:
    name, purpose = catalog.get(narrative_id, ("", ""))
    return f"{narrative_id} — {name}" if name else narrative_id


@st.cache_data
def load_index() -> pd.DataFrame:
    path = _REPO_ROOT / "outputs/online2/sae_pilot/pipeline_explorer_index.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource
def load_vocab_and_abspos(run_name: str = "original"):
    """run_name이 실제로 학습에 쓴 build_dir(vocab_v2.json 계보)에서 vocab/abspos를
    읽는다 -- 예전엔 outputs/online2/v2_build로 고정돼 있어서, 2026-07-26 이후
    다른 코퍼스(v2_build_expanded_full1517, 3404 vocab)로 학습한 run을 선택하면
    옛 vocab(728)과 어긋나 embedding index-out-of-range로 죽었다."""
    build_dir = da.build_dir_for_run(run_name)
    vocab = RegistryVocabulary(
        registry_path=str(build_dir / "life2vec_token_registry_v2.json"),
        registry_version="v2",
    )
    abspos_reference = load_abspos_reference(build_dir / "abspos_reference.json")
    return vocab, abspos_reference


@st.cache_resource
def load_farm_events_cached(run_name: str, farm_ids_key: tuple[str, ...]):
    build_dir = da.build_dir_for_run(run_name)
    events_df = load_events_frame(build_dir / "events_tokenized_v2.parquet", set(farm_ids_key))
    return build_farm_event_lists(events_df, set(farm_ids_key))


@st.cache_resource
def load_encoder_cached(run_name: str):
    vocab, _ = load_vocab_and_abspos(run_name)
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
    vocab, abspos_reference = load_vocab_and_abspos(run_name)
    farm_events = load_farm_events_cached(run_name, (farm_id,))
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


def render_property_trace_charts(
    seq_rows: pd.DataFrame,
    *,
    raw_match_window: int,
    key_prefix: str,
) -> None:
    """'시퀀스 전체 속성 변화' 그래프 -- tab1(narrative 시퀀스)과 finetune 케이스 탭이 공유.
    `seq_rows`는 farm_id/zone_id/timestamp/SENTENCE/measurement_group_ids/token_roles
    컬럼을 가진 DataFrame이어야 한다(출처가 narrative 시퀀스든 finetune 케이스든 무관)."""
    st.markdown("**시퀀스 전체 속성 변화 (시간에 따른 그래프 — raw 연속값 + token bin)**")
    st.caption(
        "이 시퀀스(또는 케이스) 이벤트들이 걸쳐 있는 시간 구간의 raw 수치를 **연속으로** 이어붙여 그린다 — "
        "이벤트와 이벤트 사이에 있는 raw 중간값도 전부 포함한다(이벤트만 찍는 게 아니다). "
        "y축은 항상 raw 연속값이라 대소 비교는 여기로 판단한다. **진한 점**은 이 시퀀스에 실제로 "
        "포함된 이벤트 시각, **옅은 점**은 그 사이/전후의 raw 원본값(이 시퀀스에 토큰화되어 들어가지 "
        "않음)이다(투명도로 구분 — 색은 아래 zone 구분에 씀). **색은 zone_id**를 나타낸다 — 케이스가 "
        "여러 zone에 걸치면(farm 단위 센서가 zone별 raw 파일에 각각 기록되는 경우 등) 같은 컬럼이 zone마다 "
        "따로 그려지고 선도 zone별로 끊어서 이어진다(시각순으로만 이으면 서로 다른 zone 값이 섞여 지그재그로 "
        "보이는 문제 방지). 점 위 라벨/tooltip의 bin(예: `ABS_B07`)은 그 시점에 tokenizer가 실제로 매긴 "
        "값이라, 시퀀스 이벤트 중에서도 그 feature를 실제로 토큰화한 경우에만 붙는다 — 모든 진한 점에 "
        "라벨이 붙는 건 아니다(narrative/케이스가 그 이벤트에서 이 feature를 안 썼을 수 있음)."
    )
    # exact-timestamp -> feature(lower) -> bin token, only for this sequence's own events.
    bin_lookup: dict[tuple[str, str, pd.Timestamp], dict[str, str]] = {}
    seq_ts_by_key: dict[tuple[str, str], set[pd.Timestamp]] = {}
    for _, row in seq_rows.iterrows():
        key = (row["farm_id"], str(row["zone_id"]))
        event_ts = pd.Timestamp(row["timestamp"])
        if event_ts.tzinfo is not None:
            event_ts = event_ts.tz_localize(None)
        seq_ts_by_key.setdefault(key, set()).add(event_ts)

        entries = da.parse_token_trace(row["SENTENCE"], row["measurement_group_ids"], row["token_roles"])
        # *_combined role tokens are already "{FEATURE_NAME}|{bin}" -- e.g. "EC_SENSOR|ABS_B07" --
        # no need to correlate group_id with a separate feature_identity token.
        bin_by_feature: dict[str, str] = {}
        for e in entries:
            if e.role.endswith("_combined") and "|" in e.token:
                feat, bin_val = e.token.split("|", 1)
                bin_by_feature.setdefault(feat.lower(), bin_val)
        bin_lookup[(key[0], key[1], event_ts)] = bin_by_feature

    IN_SEQ, NOT_IN_SEQ = "시퀀스 이벤트", "raw 중간값(비-시퀀스)"
    long_rows: list[dict] = []
    for (farm_id_k, zone_id_k), ts_set in seq_ts_by_key.items():
        lo = min(ts_set) - pd.Timedelta(hours=raw_match_window)
        hi = max(ts_set) + pd.Timedelta(hours=raw_match_window)
        raw_by_table = da.lookup_raw_rows_range(farm_id_k, zone_id_k, lo, hi)
        for table, tdf in raw_by_table.items():
            numeric_cols = [c for c in tdf.columns if pd.api.types.is_numeric_dtype(tdf[c])]
            for col in numeric_cols:
                for _, r in tdf.iterrows():
                    r_ts = pd.Timestamp(r["timestamp"])
                    in_sequence = r_ts in ts_set
                    bin_token = (
                        bin_lookup.get((farm_id_k, zone_id_k, r_ts), {}).get(col.lower(), "")
                        if in_sequence
                        else ""
                    )
                    long_rows.append(
                        {
                            "timestamp": r["timestamp"],
                            "table": table,
                            "zone_id": str(zone_id_k) if zone_id_k else "(zone 없음)",
                            "column": col,
                            "raw_value": r[col],
                            "구분": IN_SEQ if in_sequence else NOT_IN_SEQ,
                            "bin_token": bin_token,
                        }
                    )

    if not long_rows:
        st.caption("이 시퀀스에서 raw 수치 데이터를 못 찾음.")
        return

    long_df = pd.DataFrame(long_rows).sort_values("timestamp")
    long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])
    all_cols = sorted(long_df["column"].unique())
    plot_cols = st.multiselect(
        "그래프로 볼 컬럼 (전체 raw 수치 컬럼)",
        all_cols,
        default=all_cols[: min(4, len(all_cols))],
        key=f"{key_prefix}_plot_cols",
    )

    ts_min, ts_max = long_df["timestamp"].min(), long_df["timestamp"].max()
    span_hours = max(1, int((ts_max - ts_min) / pd.Timedelta(hours=1)))
    default_window_hours = min(span_hours, 72)
    if span_hours > default_window_hours:
        st.caption(
            f"전체 구간이 {span_hours}시간(약 {span_hours / 24:.1f}일)치라 한 화면에 다 넣으면 "
            f"알아보기 어렵다 — 기본으로 최근 {default_window_hours}시간만 표시한다. 아래에서 구간을 넓히거나 "
            "옮길 수 있고, 각 그래프는 마우스 스크롤로 확대/드래그로 이동도 된다."
        )
    win_start_h, win_end_h = st.slider(
        "표시할 시간 구간 (구간 시작 기준 경과 시간, 좁힐수록 그래프가 덜 빽빽해짐)",
        min_value=0, max_value=span_hours,
        value=(max(0, span_hours - default_window_hours), span_hours),
        key=f"{key_prefix}_time_window",
    )
    win_start = ts_min + pd.Timedelta(hours=win_start_h)
    win_end = ts_min + pd.Timedelta(hours=win_end_h)
    view_df = long_df[(long_df["timestamp"] >= win_start) & (long_df["timestamp"] <= win_end)]

    zone_count = view_df["zone_id"].nunique()
    for col in plot_cols:
        col_df = view_df[view_df["column"] == col]
        if col_df.empty:
            continue
        zone_color = alt.Color(
            "zone_id:N",
            title="zone",
            legend=alt.Legend(title="zone") if zone_count > 1 else None,
        )
        # zone별로 선을 끊어서 이어야 한다 -- 같은 컬럼(예: co2_ppm)이 여러 zone 파일에
        # 각각 존재하면(farm 단위 센서가 zone별 CSV에 중복 기록되는 경우 등), 시각 순으로만
        # 이으면 서로 다른 zone 값이 지그재그로 섞여 하나의 선처럼 보이는 문제가 있었다.
        line = (
            alt.Chart(col_df)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="시각"),
                y=alt.Y("raw_value:Q", title=col),
                color=zone_color,
                detail="zone_id:N",
                opacity=alt.Opacity(
                    "구분:N", scale=alt.Scale(domain=[IN_SEQ, NOT_IN_SEQ], range=[0.9, 0.35]), legend=None
                ),
            )
        )
        points = (
            alt.Chart(col_df)
            .mark_point(filled=True)
            .encode(
                x="timestamp:T",
                y="raw_value:Q",
                color=zone_color,
                size=alt.Size(
                    "구분:N",
                    scale=alt.Scale(domain=[IN_SEQ, NOT_IN_SEQ], range=[110, 25]),
                    legend=alt.Legend(title="구분"),
                ),
                opacity=alt.Opacity(
                    "구분:N", scale=alt.Scale(domain=[IN_SEQ, NOT_IN_SEQ], range=[0.95, 0.4]), legend=None
                ),
                tooltip=["timestamp:T", "raw_value:Q", "zone_id:N", "table:N", "구분:N", "bin_token:N"],
            )
        )
        labeled = col_df[col_df["bin_token"] != ""]
        layers = [line, points]
        # 라벨이 너무 많으면 서로 겹쳐 오히려 안 보이므로, 보이는 구간 기준 라벨이 40개
        # 넘으면 화면 텍스트는 생략하고 tooltip으로만 확인하게 한다(위 tooltip에 bin_token 포함).
        if not labeled.empty and len(labeled) <= 40:
            text = (
                alt.Chart(labeled)
                .mark_text(dy=-12, fontSize=9, color="#6b7062")
                .encode(x="timestamp:T", y="raw_value:Q", text="bin_token:N")
            )
            layers.append(text)
        st.altair_chart(
            alt.layer(*layers).properties(height=260, title=col).interactive(),
            use_container_width=True,
        )
    if plot_cols and not view_df.empty:
        max_labeled_in_view = max(
            (int((view_df[view_df["column"] == c]["bin_token"] != "").sum()) for c in plot_cols),
            default=0,
        )
        if max_labeled_in_view > 40:
            st.caption("라벨이 많아 화면 텍스트는 생략했다 — 점에 마우스를 올리면 tooltip으로 bin 값을 볼 수 있다.")

    matched_n = int((long_df["bin_token"] != "").sum())
    in_seq_n = int((long_df["구분"] == IN_SEQ).sum())
    st.caption(
        f"[전체 구간 기준] 시퀀스 이벤트(진한 점): {in_seq_n}/{len(long_df)}개 raw 행 — 나머지는 그 사이/전후의 raw "
        f"중간값(옅은 점, 이 시퀀스에 토큰화되어 들어가지 않음). 라벨 매칭: {matched_n}/{len(long_df)}개 — "
        f"feature 토큰 이름을 소문자화해 raw 컬럼명과 매칭(예: `EC_SENSOR`→`ec_sensor`)하고, **그 이벤트의 "
        f"정확한 토큰화 시각과 raw 행 시각이 일치할 때만** 라벨을 붙인다(그래서 진한 점이라도 라벨이 없을 "
        f"수 있음 — 같은 bin 라벨이 서로 다른 raw 값에 잘못 찍히는 걸 막기 위함). 매칭 안 된 값은 라벨 없이 "
        f"표시됨(actuator on/off류처럼 bin이 아니라 상태값인 경우 등). 위 그래프는 슬라이더로 선택한 구간만 그린다."
    )


def render_sequence_trace_tab(index_df: pd.DataFrame) -> None:
    st.subheader("1. 원시데이터 ↔ 토큰 ↔ 시퀀스 추적")
    st.caption(
        "narrative가 raw 이벤트를 골라 시퀀스로 만든 결과를 그대로 보여준다 — 왼쪽은 그 시퀀스의 토큰 "
        "(역할별 색), 오른쪽은 각 이벤트가 나온 원본 raw CSV 행(E_environment/R_rootzone/A_actuator/G_growth)."
    )
    if index_df.empty:
        st.warning("인덱스가 없다 — 먼저 `scripts/online2_v2/build_pipeline_explorer_index.py`를 실행하세요.")
        return

    catalog_full = load_narrative_catalog_full()
    narratives = sorted(index_df["narrative_id"].unique())
    narrative_id = st.selectbox(
        "narrative_id",
        narratives,
        format_func=lambda nid: narrative_label(nid, {k: v[:2] for k, v in catalog_full.items()}),
        key="trace_narrative",
    )
    cat_row = catalog_full.get(narrative_id)
    if cat_row:
        name, purpose, category, expected_pattern, agronomic_interpretation = cat_row
        st.info(
            f"**{narrative_id} — {name}** ({category})\n\n"
            f"- **목적**: {purpose}\n"
            f"- **기대 패턴**: {expected_pattern}\n"
            f"- **해석**: {agronomic_interpretation}"
        )
    else:
        st.caption(f"{narrative_id}: 카탈로그에 설명이 없음.")

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
    raw_match_window = st.slider(
        "raw 행 매칭 허용 오차 (이벤트 시각 ± N시간, 0=정확히 일치만)",
        0, 24, 0,
        key="trace_raw_match_window",
        help=(
            "이벤트 시각과 raw CSV의 timestamp가 초 단위까지 완전히 같아야만 찾던 걸(0시간), "
            "코퍼스 빌드 시점과 raw 스냅샷 사이 미세한 시간차를 허용하도록 넓힐 수 있다 — "
            "growth 계열 서사는 실측상 최대 1일 정도 어긋나 있는 경우가 있었다. "
            "단, G_growth는 이 창을 넓혀도 못 찾는다(그 raw 파일의 시각 컬럼 이름 자체가 "
            "timestamp가 아니라 observation_date라 애초에 검색 대상이 아님)."
        ),
    )
    for _, row in seq_rows.iterrows():
        with st.expander(
            f"event_position={row['event_position']}  event_id={row['event_id']}  "
            f"({row['event_kind']}, {row['START_DATE']})",
            expanded=(row["event_position"] == seq_rows["event_position"].max()),
        ):
            col_tok, col_raw = st.columns([3, 2])
            with col_tok:
                st.markdown("**토큰 (색 = role)**")
                entries = da.parse_token_trace(
                    row["SENTENCE"], row["measurement_group_ids"], row["token_roles"],
                    row.get("source_file_ids", ""), row.get("source_row_ids", ""),
                )
                html_parts = []
                for e in entries:
                    color = ROLE_COLORS.get(e.role, "#374151")
                    prov = " · provenance" if e.source_row_id else ""
                    html_parts.append(
                        f'<span title="role={e.role} group={e.group_id}{prov}" '
                        f'style="background:{color}22;border:1px solid {color};border-radius:4px;'
                        f'padding:1px 4px;margin:2px;display:inline-block;font-size:0.85em;">{e.token}</span>'
                    )
                st.markdown(" ".join(html_parts), unsafe_allow_html=True)
                role_legend = ", ".join(f"{r}" for r in sorted({e.role for e in entries}))
                st.caption(f"등장한 role: {role_legend}")
            with col_raw:
                st.markdown("**원본 raw CSV 행**")
                # 정확한 계보(source_file_id/source_row_id, cell_occurrences.parquet 유래)가
                # 토큰에 있으면 그걸로 직접 지목한다 -- farm/zone/timestamp 재검색이 아니라서
                # tz 변환·raw 스냅샷 시간차 문제가 없다. 없을 때만(구코퍼스, 또는 이 이벤트가
                # 애초에 provenance 없는 토큰뿐일 때) 기존 fuzzy 검색으로 넘어간다.
                sfids = [e.source_file_id for e in entries]
                srids = [e.source_row_id for e in entries]
                raw, drift = da.lookup_raw_rows_for_event(sfids, srids)
                exact_used = bool(raw)
                if not raw:
                    raw = da.lookup_raw_rows(
                        row["farm_id"], str(row["zone_id"]), str(row["START_DATE"]),
                        window_hours=raw_match_window,
                    )
                if exact_used:
                    st.caption(
                        "✓ 정확한 계보(source_row_id)로 조회함 — 재검색 아님."
                        + (" ⚠ raw 파일이 build 시점과 달라짐(checksum 불일치)" if drift else "")
                    )
                elif raw:
                    st.caption("△ 이 이벤트엔 계보 정보가 없어 farm/zone/시각으로 재검색함(fuzzy).")
                if not raw:
                    st.caption(
                        f"(이 이벤트 시각/zone에 ±{raw_match_window}시간 이내로도 일치하는 raw 행을 못 찾음 — "
                        "G_growth는 timestamp 컬럼이 없어(observation_date) 애초에 검색 대상이 아니고, "
                        "그 외 테이블은 코퍼스 빌드 시점과 raw 스냅샷 사이 시간차가 이 창보다 클 수 있음 — "
                        "위 슬라이더로 허용 오차를 늘려보세요)"
                    )
                for table, df in raw.items():
                    st.caption(f"{table}{f' ({len(df)}행)' if raw_match_window > 0 else ''}")
                    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    render_property_trace_charts(
        seq_rows.rename(columns={"START_DATE": "timestamp"}),
        raw_match_window=raw_match_window,
        key_prefix=f"trace_{seq_label}",
    )


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

    run_names = da.discover_all_run_names()
    run_name = st.selectbox("체크포인트(run)", run_names, format_func=run_select_label, key="encoding_run")

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

    run_names = da.discover_all_run_names()
    run_name = st.selectbox("체크포인트(run)", run_names, format_func=run_select_label, key="sim_run")
    position = st.selectbox("지점(position)", POSITIONS, key="sim_position")

    catalog = load_narrative_catalog()
    seq_choices = index_df[["sequence_id", "farm_id", "narrative_id"]].drop_duplicates(subset=["sequence_id"])
    seq_ids = seq_choices["sequence_id"].tolist()
    seq_label_by_id = {
        r["sequence_id"]: f"{r['sequence_id']} ({narrative_label(r['narrative_id'], catalog)}, farm={r['farm_id']})"
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

    st.markdown("---")
    st.markdown("**Concept Space 시각화**")
    st.caption(
        "여러 narrative_id(개념)의 시퀀스를 한 표현 공간에 2D로 투영해 개념별 군집을 본다. "
        "두 시퀀스만 비교하는 위 유사도와 달리, 다수 개념이 실제로 서로 갈려 있는지를 한눈에 확인하는 용도."
    )
    all_narratives = sorted(index_df["narrative_id"].dropna().unique().tolist())
    cs_narratives = st.multiselect(
        "narrative_id 선택",
        all_narratives,
        default=all_narratives[: min(8, len(all_narratives))],
        format_func=lambda nid: narrative_label(nid, catalog),
        key="cs_narratives",
    )
    if cs_narratives:
        with st.expander("선택한 narrative의 목적(purpose) 보기"):
            for nid in cs_narratives:
                name, purpose = catalog.get(nid, ("", ""))
                st.markdown(f"- **{nid} ({name})**: {purpose}" if name else f"- **{nid}**: (카탈로그에 없음)")
    cs_max_per = st.slider("narrative당 최대 시퀀스 수", 2, 15, 5, key="cs_max_per")
    cs_method = st.radio("투영 방법", ["PCA", "UMAP"], horizontal=True, key="cs_method")
    cs_run_name = st.selectbox("체크포인트(run)", run_names, format_func=run_select_label, key="cs_run")
    cs_position = st.selectbox("지점(position)", POSITIONS, key="cs_position")

    if st.button("Concept Space 그리기", key="cs_button"):
        if not cs_narratives:
            st.warning("narrative_id를 하나 이상 선택하세요.")
        else:
            sub = index_df[index_df["narrative_id"].isin(cs_narratives)]
            rep_rows = sub.sort_values("event_position").groupby("sequence_id").tail(1)
            sampled = rep_rows.groupby("narrative_id").head(cs_max_per)

            vecs: list[np.ndarray] = []
            seq_ids_sampled: list[str] = []
            narr_sampled: list[str] = []
            with st.spinner(f"{len(sampled)}개 시퀀스 인코딩 중..."):
                for _, row in sampled.iterrows():
                    v = encode_sequence(cs_run_name, row["farm_id"], row["event_id"])
                    if v is None:
                        continue
                    vecs.append(v[cs_position])
                    seq_ids_sampled.append(row["sequence_id"])
                    narr_sampled.append(row["narrative_id"])

            if len(vecs) < 4:
                st.error(f"인코딩된 시퀀스가 너무 적다({len(vecs)}개) — 투영하려면 최소 4개 필요.")
            else:
                activations = np.stack(vecs)
                method_params = {}
                if cs_method == "UMAP":
                    method_params["n_neighbors"] = max(2, min(15, len(vecs) - 1))
                artifact = project_activations(
                    activations,
                    seq_ids_sampled,
                    method=cs_method,
                    checkpoint_id=cs_run_name,
                    layer=cs_position,
                    color_label_source="narrative_id",
                    seed=0,
                    trustworthiness_n_neighbors=max(2, min(5, len(vecs) - 1)),
                    method_params=method_params,
                )
                st.caption(f"⚠ {PROJECTION_DISCLAIMER}")
                st.caption(
                    f"neighborhood_preservation(trustworthiness)={artifact.metadata.neighborhood_preservation:.3f} "
                    f"— 1.0에 가까울수록 2D 투영이 원공간 이웃관계를 잘 보존"
                )
                plot_df = pd.DataFrame(
                    {
                        "x": [c[0] for c in artifact.coordinates],
                        "y": [c[1] for c in artifact.coordinates],
                        "narrative_id": narr_sampled,
                        "sequence_id": seq_ids_sampled,
                    }
                )
                scatter_with_tooltip(
                    plot_df, x="x", y="y", color="narrative_id",
                    tooltip=["sequence_id", "narrative_id", "x", "y"],
                )
                st.caption(f"{len(vecs)}개 시퀀스, {len(set(narr_sampled))}개 narrative_id 표시됨.")


def render_vocab_concept_space_tab() -> None:
    st.subheader("Vocab Concept Space — 등록된 token 단위")
    st.caption(
        "life2vec 사전학습 vocab(RegistryVocabulary)에 등록된 모든 token의 고정 임베딩 벡터를 2D로 투영한다. "
        "3번 탭의 시퀀스 투영과 달리 문맥(contextual activation)이 아니라, 문맥과 무관하게 고정된 "
        "vocab embedding table 자체의 값이다 — 계획서 §8.3의 'Token embedding'(고정) vs "
        "'Contextual activation'(문맥별) 구분에서 전자에 해당."
    )
    run_names = da.discover_all_run_names()
    vc_run_name = st.selectbox("체크포인트(run)", run_names, format_func=run_select_label, key="vc_run")
    # vocab도 run마다 다를 수 있다(2026-07-26 이후 run은 v2_build_expanded_full1517의
    # 3404-vocab, 그 이전 run은 v2_build의 728-vocab) -- 그래서 run 선택 다음에 로드.
    vocab, _ = load_vocab_and_abspos(vc_run_name)
    vocab_df = vocab.vocab()  # columns: ID, TOKEN, CATEGORY
    st.caption(f"vocab 크기: {len(vocab_df)}개 token, {vocab_df['CATEGORY'].nunique()}개 CATEGORY")
    all_categories = sorted(vocab_df["CATEGORY"].unique().tolist())
    vc_categories = st.multiselect(
        "CATEGORY 필터 (비우면 전체 표시)", all_categories, default=[], key="vc_categories"
    )
    vc_method = st.radio("투영 방법", ["PCA", "UMAP"], horizontal=True, key="vc_method")

    if st.button("어휘 Concept Space 그리기", key="vc_button"):
        filtered = vocab_df if not vc_categories else vocab_df[vocab_df["CATEGORY"].isin(vc_categories)]
        if len(filtered) < 4:
            st.error(f"선택된 token이 너무 적다({len(filtered)}개) — 투영하려면 최소 4개 필요.")
        else:
            model, _device = load_encoder_cached(vc_run_name)
            weight = model.transformer.embedding.token.weight.detach().cpu().numpy()
            ids = filtered["ID"].tolist()
            activations = weight[ids]
            tokens = filtered["TOKEN"].tolist()
            categories = filtered["CATEGORY"].tolist()

            method_params = {}
            if vc_method == "UMAP":
                method_params["n_neighbors"] = max(2, min(15, len(activations) - 1))
            artifact = project_activations(
                activations,
                tokens,
                method=vc_method,
                checkpoint_id=vc_run_name,
                layer="token_embedding",
                color_label_source="vocab_registry_category",
                seed=0,
                trustworthiness_n_neighbors=max(2, min(5, len(activations) - 1)),
                method_params=method_params,
            )
            st.caption(f"⚠ {PROJECTION_DISCLAIMER}")
            st.caption(
                f"neighborhood_preservation(trustworthiness)={artifact.metadata.neighborhood_preservation:.3f}, "
                f"n_tokens={len(activations)}"
            )
            plot_df = pd.DataFrame(
                {
                    "x": [c[0] for c in artifact.coordinates],
                    "y": [c[1] for c in artifact.coordinates],
                    "CATEGORY": categories,
                    "TOKEN": tokens,
                }
            )
            scatter_with_tooltip(
                plot_df, x="x", y="y", color="CATEGORY", tooltip=["TOKEN", "CATEGORY", "x", "y"], size=40
            )
            st.dataframe(
                plot_df[["TOKEN", "CATEGORY", "x", "y"]], use_container_width=True, hide_index=True
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


@st.cache_data(ttl=120)
def discover_finetune_runs() -> list[dict]:
    """finetune run 디렉토리 자동 감지 -- OOF representation npz가 있는 것만, 최신순.
    사용자 요청(2026-07-24): Entity Summary에서 run 디렉토리를 직접 타이핑하지 않고
    모델을 골라 쓸 수 있어야 함."""
    import json

    root = _REPO_ROOT / "outputs/online2/v2_finetune_v03/runs"
    if not root.exists():
        return []
    out = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        npz_files = list(d.glob("*_oof_representations.npz"))
        if not npz_files:
            continue
        mtime = max(f.stat().st_mtime for f in npz_files)
        macro_f1 = None
        summary_path = d / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                f1s = [
                    r["best"].get("val_fine_macro_f1")
                    for r in summary.get("results", [])
                    if r.get("best") and r["best"].get("val_fine_macro_f1") is not None
                ]
                if f1s:
                    macro_f1 = float(np.mean(f1s))
            except (json.JSONDecodeError, KeyError):
                pass
        ts = pd.Timestamp(mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        f1_part = f", macro_f1={macro_f1:.3f}" if macro_f1 is not None else ""
        out.append({"run_dir": str(d), "label": f"{d.name} ({ts}{f1_part})", "mtime": mtime})
    out.sort(key=lambda r: -r["mtime"])
    return out


@st.cache_data
def load_finetune_oof(run_dir_str: str) -> pd.DataFrame:
    run_dir = Path(run_dir_str)
    rows = []
    seen: set[str] = set()
    for path in sorted(run_dir.glob("*_oof_representations.npz")):
        split_id = path.stem.replace("_oof_representations", "")
        data = np.load(path, allow_pickle=True)
        for cid, y, zp, hb in zip(data["case_id"], data["y_abnormal"], data["z_proj"], data["h_binary"]):
            cid = str(cid)
            if cid in seen:
                continue
            seen.add(cid)
            rows.append(
                {"case_id": cid, "y_abnormal": float(y), "z_proj": zp, "h_binary": hb, "split_id": split_id}
            )
    return pd.DataFrame(rows)


@st.cache_data
def load_finetune_labels() -> pd.DataFrame:
    path = _REPO_ROOT / "outputs/online2/v2_finetune/labels_example_score90.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_case_manifest() -> pd.DataFrame:
    path = _REPO_ROOT / "outputs/online2/v2_finetune/case_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["case_id"] = df["case_id"].astype(str)
    df["farm_id"] = df["farm_id"].astype(str)
    return df


@st.cache_resource
def load_registry_vocab_for_run(run_name: str) -> RegistryVocabulary:
    build_dir = da.build_dir_for_run(run_name)
    return RegistryVocabulary(
        registry_path=str(build_dir / "life2vec_token_registry_v2.json"), registry_version="v2"
    )


@st.cache_data
def load_case_events_cached(
    events_path_str: str, farm_id: str, period_start: str, period_end: str, include_image: bool
) -> pd.DataFrame:
    return da.load_case_events_full(
        Path(events_path_str), farm_id, period_start, period_end, include_image=include_image
    )


@st.cache_data
def load_case_events_raw_cached(
    events_path_str: str, farm_id: str, period_start: str, period_end: str, include_image: bool = False
) -> list:
    """`token_ixg_for_event_v03`가 요구하는 `list[CaseEvent]` 형태(DataFrame이 아님) --
    Stage A 캐싱이 실제로 쓰는 것과 동일한 이벤트 선정/순서(`case_events_from_frame`)를
    그대로 재사용한다. `include_image`는 호출부가 명시해야 한다 -- 실제 finetune
    학습(Stage A 캐싱)은 항상 False를 쓰지만, 이 함수를 tab7(브라우징용, IMAGE 포함
    토글이 있음)과도 공유하므로 하드코딩하면 case_df와 인덱스가 어긋난다."""
    events_df = load_events_frame(Path(events_path_str), {farm_id})
    return case_events_from_frame(
        events_df, farm_id, pd.Timestamp(period_start), pd.Timestamp(period_end),
        include_image=include_image,
    )


@st.cache_data
def load_separation_probe_report(run_dir_str: str) -> dict | None:
    path = Path(run_dir_str) / "normal_abnormal_separation_probe.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_all_labeled_event_means(emb_dir_str: str, labels_json: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All events' `event_mean` (layer5-equivalent, same position the persisted SAE was
    trained on) for every labeled case, stacked with a parallel y_abnormal array and
    case_id array. Reused as-is -- no new encoder forward pass, just reading the same
    cache `llm_evidence`/TCAV already read."""
    import json

    labels = json.loads(labels_json)  # {case_id: diagnosis_normalized}
    emb_dir = Path(emb_dir_str)
    means: list[np.ndarray] = []
    y_abn: list[float] = []
    case_ids: list[str] = []
    for case_id, diag in labels.items():
        path = emb_dir / f"{case_id}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["event_mean"])
        m = np.stack([np.asarray(x, dtype=np.float32) for x in df["event_mean"]])
        means.append(m)
        y = 0.0 if diag == "정상_운영" else 1.0
        y_abn.extend([y] * len(m))
        case_ids.extend([case_id] * len(m))
    return np.concatenate(means, axis=0), np.array(y_abn), np.array(case_ids)


@st.cache_data
def load_event_token_rows(events_path_str: str, event_ids: tuple[str, ...]) -> dict:
    """SENTENCE + measurement_group_ids + token_roles + 시간/위치 메타(observation_timestamp,
    farm_id, zone_id) for a handful of event_ids (top-k salient events only) -- same
    pyarrow-filter pattern as llm_evidence/evidence_bundle.py::lookup_event_sentences,
    extended with the columns da.parse_token_trace + raw-data lookup need. 2026-07-24:
    사용자 지적("이벤트 정보에 시간 정보가 누락") 반영 -- 절대 시각/farm/zone을 항상 같이 반환."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(events_path_str, format="parquet")
    available = set(dataset.schema.names)
    cols = [
        "event_id",
        "SENTENCE",
        "measurement_group_ids",
        "token_roles",
        "observation_timestamp",
        "farm_id",
        "zone_id",
    ]
    has_provenance = "source_file_ids" in available and "source_row_ids" in available
    if has_provenance:
        cols += ["source_file_ids", "source_row_ids"]
    table = dataset.to_table(
        columns=cols,
        filter=ds.field("event_id").isin(list(set(event_ids))),
    )
    df = table.to_pandas()
    return {
        row["event_id"]: {
            "sentence": row["SENTENCE"],
            "measurement_group_ids": row["measurement_group_ids"],
            "token_roles": row["token_roles"],
            "source_file_ids": row["source_file_ids"] if has_provenance else "",
            "source_row_ids": row["source_row_ids"] if has_provenance else "",
            "observation_timestamp": row["observation_timestamp"],
            "farm_id": row["farm_id"],
            "zone_id": row["zone_id"],
        }
        for _, row in df.iterrows()
    }


def render_finetune_case_tokenization_tab() -> None:
    st.subheader("7. Finetune 케이스 시퀀스 ↔ 토큰(사전학습 vocab) ↔ raw")
    st.caption(
        "finetune(정상/비정상 진단)에 실제로 들어가는 케이스별 이벤트 시퀀스가, 선택한 run의 "
        "**사전학습 vocab**(`life2vec_token_registry_v2.json`) 기준으로 어떻게 토큰화되는지 그대로 보여준다. "
        "tab1(narrative 매칭)과 달리 여기선 `case_manifest.csv`의 farm_id+기간으로 케이스를 정의하고, "
        "Stage A 캐싱(`cache_stage_a_event_embeddings.py`)이 실제로 쓰는 것과 동일한 이벤트 목록/순서를 "
        "그대로 재사용한다(별도로 다시 구현하지 않음 — 로직이 갈라질 위험 방지). "
        "**토큰은 이벤트(행) 단위로 매칭되는 게 아니라, 그 행의 특정 feature 셀 하나에 대응한다** — "
        "`FEATURE|value` 토큰 하나가 raw CSV의 한 행에서 그 feature 컬럼 값 하나만 가리킨다."
    )
    manifest = load_case_manifest()
    if manifest.empty:
        st.warning("`outputs/online2/v2_finetune/case_manifest.csv`가 없다.")
        return

    run_names = da.discover_all_run_names()
    run_name = st.selectbox(
        "체크포인트(run) — 이 run이 학습된 코퍼스의 vocab을 기준으로 토큰화를 확인",
        run_names, format_func=run_select_label, key="finetune_tok_run",
    )
    build_dir = da.build_dir_for_run(run_name)
    events_path = build_dir / "events_tokenized_v2.parquet"
    if not events_path.exists():
        st.error(f"이 run의 이벤트 파일을 못 찾음: {events_path}")
        return
    vocab = load_registry_vocab_for_run(run_name)
    unk_id = int(vocab.token2index["[UNK]"])
    st.caption(f"코퍼스: `{build_dir.name}` (vocab_size={vocab.size()})")

    labels_df = load_finetune_labels()
    diag_by_case = (
        dict(zip(labels_df["case_id"].astype(str), labels_df["diagnosis_normalized"]))
        if not labels_df.empty else {}
    )
    case_ids = manifest["case_id"].tolist()
    case_id = st.selectbox(
        "case_id",
        case_ids,
        format_func=lambda c: c + (f"  [{diag_by_case[c]}]" if c in diag_by_case else ""),
        key="finetune_tok_case",
    )
    case_row = manifest[manifest["case_id"] == case_id].iloc[0]
    include_image = st.checkbox(
        "IMAGE 이벤트 포함 (Stage A 캐싱 기본값=제외)", value=False, key="finetune_tok_include_image"
    )

    with st.spinner("케이스 이벤트 목록 로딩 중..."):
        case_df = load_case_events_cached(
            str(events_path), str(case_row["farm_id"]), str(case_row["period_start"]),
            str(case_row["period_end"]), include_image,
        )
    if case_df.empty:
        st.warning("이 케이스 기간에 이벤트가 없다.")
        return

    total_tokens = 0
    unk_tokens = 0
    unk_counter: dict[str, int] = {}
    n_tokens_col: list[int] = []
    n_unk_col: list[int] = []
    for _, row in case_df.iterrows():
        toks = row["SENTENCE"].split()
        n_unk = 0
        for t in toks:
            if int(vocab.token2index.get(t, unk_id)) == unk_id:
                n_unk += 1
                unk_counter[t] = unk_counter.get(t, 0) + 1
        # Stage A 캐싱은 이 SENTENCE 앞에 이벤트마다 ZONE_LOCAL|i 토큰을 하나 더 붙인다
        # (zone이 있는 이벤트만 -- IMAGE/INTERPRETATION처럼 zone이 없으면 안 붙음). 정확한
        # 로컬 인덱스 i는 그 이벤트가 어느 윈도우에 포함되는지에 따라 달라져(윈도우마다
        # 그 안에 실제로 등장하는 zone들만 다시 번호를 매김) 여기서 고정값으로 보여줄 수는
        # 없지만, ZONE_LOCAL|*는 항상 vocab에 있는 토큰이라(UNK 되지 않음) 개수만은 정확히
        # 반영한다.
        n_zone_tok = 1 if zone_present(str(row["zone_id"])) else 0
        total_tokens += len(toks) + n_zone_tok
        unk_tokens += n_unk
        n_tokens_col.append(len(toks) + n_zone_tok)
        n_unk_col.append(n_unk)
    case_df = case_df.assign(n_tokens=n_tokens_col, n_unk=n_unk_col)

    c1, c2, c3 = st.columns(3)
    c1.metric("이벤트 수", len(case_df))
    c2.metric("전체 토큰 수", total_tokens)
    c3.metric(
        "UNK 비율 (이 run vocab 기준)",
        f"{(unk_tokens / max(1, total_tokens)):.2%}",
        delta=f"{unk_tokens}개",
        delta_color="inverse",
    )
    if unk_counter:
        top_unk = sorted(unk_counter.items(), key=lambda kv: -kv[1])[:10]
        st.caption(
            "가장 흔한 UNK 원본 토큰(이 run의 사전학습 vocab에 없어 `[UNK]`로 대체됨): "
            + ", ".join(f"`{t}`×{n}" for t, n in top_unk)
        )

    st.markdown("---")
    st.markdown(f"**이벤트 목록** ({len(case_df)}개, 시간순 — Stage A와 동일한 순서)")
    st.dataframe(
        case_df[["event_position", "timestamp", "event_kind", "view", "zone_id", "n_tokens", "n_unk"]],
        use_container_width=True, hide_index=True, height=240,
    )

    idx = st.number_input(
        "상세히 볼 이벤트 위치 (event_position)",
        min_value=0, max_value=len(case_df) - 1, value=0, key="finetune_tok_event_idx",
    )
    raw_match_window = st.slider(
        "raw 행 매칭 허용 오차 (이벤트 시각 ± N시간, 0=정확히 일치만)",
        0, 24, 0, key="finetune_tok_raw_window",
    )
    row = case_df.iloc[int(idx)]
    st.markdown(
        f"**event_position={row['event_position']}  event_id={row['event_id']}**  "
        f"({row['event_kind']}, {row['timestamp']}, view={row['view']}, zone={row['zone_id']})"
    )
    col_tok, col_raw = st.columns([3, 2])
    with col_tok:
        st.markdown("**토큰 (색 = role, 빨간 테두리 = 이 run vocab에 없어 `[UNK]`로 대체됨)**")
        entries = da.parse_token_trace(
            row["SENTENCE"], row["measurement_group_ids"], row["token_roles"],
            row.get("source_file_ids", ""), row.get("source_row_ids", ""),
        )
        html_parts = []
        # SENTENCE/token_roles는 corpus 원본이라 ZONE_LOCAL을 담고 있지 않다(사전학습·
        # finetune Stage A 둘 다 인코딩 직전에 즉석으로 붙임) -- 실제로 어떤 토큰이 붙는지
        # 이 이벤트를 Stage A 타겟으로 삼아 윈도우를 구성해보고 그대로 계산해서 보여준다
        # (점선 테두리로 "corpus에 저장된 토큰이 아니라 계산된 예시"임을 표시).
        if zone_present(str(row["zone_id"])):
            zone_events = load_case_events_raw_cached(
                str(events_path), str(case_row["farm_id"]), str(case_row["period_start"]),
                str(case_row["period_end"]), include_image,
            )
            id_to_idx = {ev.event_id: i for i, ev in enumerate(zone_events)}
            t_idx = id_to_idx.get(row["event_id"])
            zone_tok = None
            if t_idx is not None:
                window = stage_a_module.construct_target_window(zone_events, t_idx, max_length=1024)
                selected = [zone_events[i] for i in window.event_indices]
                zones_in_window = sorted({ev.zone for ev in selected if zone_present(ev.zone)})
                zone_local = {z: f"ZONE_LOCAL|{i}" for i, z in enumerate(zones_in_window)}
                zone_tok = zone_local.get(zone_events[t_idx].zone)
            if zone_tok:
                st.caption(
                    f"실제 인코딩 시 이 앞에 `{zone_tok}`가 붙는다(점선 테두리 칩 — Stage A 기본 "
                    "윈도우 크기 1024토큰 기준으로 지금 계산한 값. 이 이벤트가 다른 윈도우에 포함되면 "
                    "그 안의 zone 구성에 따라 인덱스가 달라질 수 있다 — vocab에 항상 있어 UNK는 안 됨)."
                )
                html_parts.append(
                    f'<span title="실제 인코딩 시 이 이벤트 앞에 붙는 토큰(계산됨, corpus 원본 아님)" '
                    f'style="background:{ROLE_COLORS.get("context", "#0891b2")}22;'
                    f'border:2px dashed {ROLE_COLORS.get("context", "#0891b2")};border-radius:4px;'
                    f'padding:1px 4px;margin:2px;display:inline-block;font-size:0.85em;">{zone_tok}</span>'
                )
        for e in entries:
            color = ROLE_COLORS.get(e.role, "#374151")
            is_unk = int(vocab.token2index.get(e.token, unk_id)) == unk_id
            border = "2px solid #ef4444" if is_unk else f"1px solid {color}"
            prov = " · provenance" if e.source_row_id else ""
            unk_tag = " · UNK!" if is_unk else ""
            html_parts.append(
                f'<span title="role={e.role} group={e.group_id}{prov}{unk_tag}" '
                f'style="background:{color}22;border:{border};border-radius:4px;'
                f'padding:1px 4px;margin:2px;display:inline-block;font-size:0.85em;">{e.token}</span>'
            )
        st.markdown(" ".join(html_parts), unsafe_allow_html=True)
        role_legend = ", ".join(sorted({e.role for e in entries}))
        st.caption(f"등장한 role: {role_legend}")
    with col_raw:
        st.markdown("**원본 raw CSV 행**")
        sfids = [e.source_file_id for e in entries]
        srids = [e.source_row_id for e in entries]
        raw, drift = da.lookup_raw_rows_for_event(sfids, srids)
        exact_used = bool(raw)
        if not raw:
            raw = da.lookup_raw_rows(
                str(row["farm_id"]), str(row["zone_id"]), str(row["timestamp"]),
                window_hours=raw_match_window,
            )
        if exact_used:
            st.caption(
                "✓ 정확한 계보(source_row_id)로 조회함 — 재검색 아님."
                + (" ⚠ raw 파일이 build 시점과 달라짐(checksum 불일치)" if drift else "")
            )
        elif raw:
            st.caption("△ 이 이벤트엔 계보 정보가 없어 farm/zone/시각으로 재검색함(fuzzy).")
        if not raw:
            st.caption(f"(±{raw_match_window}시간 이내로도 일치하는 raw 행을 못 찾음)")
        for table, df in raw.items():
            st.caption(f"{table}{f' ({len(df)}행)' if raw_match_window > 0 else ''}")
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    render_property_trace_charts(case_df, raw_match_window=raw_match_window, key_prefix=f"finetune_{case_id}")


def render_entity_summary_tab() -> None:
    st.subheader("5. Entity Summary — 정상/비정상 개념 공간 + TCAV")
    st.caption(
        "finetune 진단 케이스의 out-of-fold(OOF) 표현을 person_space.ipynb(원본 life2vec)와 같은 "
        "'개체별 표현 공간 위치' 방식으로 본다. 반드시 OOF 표현만 쓴다 — in-sample이면 분리가 "
        "자명하게 좋아 보여 무의미하다."
    )

    discovered = discover_finetune_runs()
    if discovered:
        options = [d["run_dir"] for d in discovered]
        label_by_dir = {d["run_dir"]: d["label"] for d in discovered}
        picked = st.selectbox(
            "모델(finetune run) 선택",
            options,
            format_func=lambda p: label_by_dir[p],
            key="entity_run_select",
        )
        use_manual = st.checkbox("직접 경로 입력", value=False, key="entity_run_manual_toggle")
        run_dir_str = (
            st.text_input("run 디렉토리 직접 입력", value=picked, key="entity_run_dir")
            if use_manual
            else picked
        )
    else:
        default_run_dir = _REPO_ROOT / "outputs/online2/v2_finetune_v03/runs/probe_oof_3fold"
        st.info("outputs/online2/v2_finetune_v03/runs/ 아래에서 자동 감지된 run이 없다 — 경로를 직접 입력.")
        run_dir_str = st.text_input("finetune OOF run 디렉토리", value=str(default_run_dir), key="entity_run_dir")
    oof_df = load_finetune_oof(run_dir_str)
    labels_df = load_finetune_labels()
    if oof_df.empty or labels_df.empty:
        st.warning(
            f"OOF representation(`*_oof_representations.npz`)이 없다 — "
            f"{run_dir_str}에서 `run_diagnosis_finetune_v03.py --mode cv_repeated`를 먼저 실행해야 한다."
        )
        return

    diag_by_case = labels_df.set_index(labels_df["case_id"].astype(str))["diagnosis_normalized"].to_dict()
    oof_df["diagnosis"] = oof_df["case_id"].map(diag_by_case).fillna("UNKNOWN")

    n = len(oof_df)
    n_abn = int(oof_df["y_abnormal"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("케이스 수 (OOF)", n)
    col2.metric("비정상", n_abn)
    col3.metric("정상", n - n_abn)

    probe_report = load_separation_probe_report(run_dir_str)
    if probe_report:
        st.markdown("**정상/비정상 선형 분리 실측 (cross-validated linear probe)**")
        prow = st.columns(len(probe_report))
        for col, (rep_name, r) in zip(prow, probe_report.items()):
            col.metric(
                f"{rep_name} AUROC",
                f"{r['cv_auroc_mean']:.3f} ± {r['cv_auroc_std']:.3f}",
                help=f"실루엣={r['silhouette']:.3f}, n={r['metadata']['n_samples']}",
            )
        st.caption(probe_report[next(iter(probe_report))]["disclaimer"])
    else:
        st.info(
            "`scripts/online2_v2/v03/probe_normal_abnormal_separation_v03.py`를 먼저 돌리면 "
            "여기에 AUROC/실루엣 실측치가 나온다."
        )

    rep_choice = st.selectbox("투영할 표현", ["h_binary", "z_proj"], key="entity_rep_choice")
    reps = np.stack(oof_df[rep_choice].tolist())

    ckpt_options = sorted(str(p) for p in Path(run_dir_str).glob("*fold*best*.pt"))
    ckpt_path = None
    if ckpt_options:
        ckpt_path = st.selectbox(
            "체크포인트 (예측/근거 + TCAV 공용)",
            ckpt_options,
            format_func=lambda p: f"{Path(p).name} ({pd.Timestamp(Path(p).stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M')} 저장됨)",
            key="tcav_ckpt",
        )
    else:
        st.info(
            "이 run 디렉토리에 fold 체크포인트가 없다 — '케이스 요약 · 예측/근거'의 예측/근거와 "
            "'TCAV' 하위 탭은 계산할 수 없다(개념 공간/SAE 분석은 계속 볼 수 있음)."
        )

    sub_concept, sub_sae, sub_case, sub_tcav = st.tabs(
        ["개념 공간 + Split검증", "SAE 분석", "케이스 요약 · 예측/근거", "TCAV"]
    )

    with sub_concept:
        artifact = project_activations(
            reps,
            oof_df["case_id"].tolist(),
            method="PCA",
            checkpoint_id=Path(run_dir_str).name,
            layer=rep_choice,
            color_label_source="y_abnormal (label_map.json)",
            seed=0,
        )
        st.caption(f"⚠ {PROJECTION_DISCLAIMER}")
        st.caption(
            f"neighborhood_preservation(trustworthiness)={artifact.metadata.neighborhood_preservation:.3f} "
            f"(1.0에 가까울수록 3D 투영이 원공간 이웃관계를 잘 보존)"
        )
        oof_indexed = oof_df.set_index("case_id")
        plot_df = pd.DataFrame(
            {
                "x": [c[0] for c in artifact.coordinates],
                "y": [c[1] for c in artifact.coordinates],
                "case_id": artifact.point_ids,
                "diagnosis": [oof_indexed.loc[cid, "diagnosis"] for cid in artifact.point_ids],
                "정상/비정상": ["비정상" if oof_indexed.loc[cid, "y_abnormal"] > 0 else "정상" for cid in artifact.point_ids],
            }
        )
        scatter_with_tooltip(
            plot_df, x="x", y="y", color="정상/비정상",
            tooltip=["case_id", "diagnosis", "정상/비정상", "x", "y"],
        )

        st.markdown("**Split 검증 — fold(val) 겹침 확인**")
        st.caption(
            "각 케이스가 held-out(val)으로 들어간 fold를 라벨 삼아, 이 표현 공간에서 다른 fold와 "
            "얼마나 섞여 있는지 본다(`representation_explorer/overlap_check.py` 재사용, 계획서 §10 — "
            "참고용 진단이지 leakage 판정 도구가 아니다). 비율이 높다고 곧바로 누수는 아니고 "
            "낮다고 누수가 없다는 보장도 아니다 — 실제 leakage 판정은 CF1S의 "
            "`core_fold_provenance`/`case_fold_provenance` 계약이 한다."
        )
        if "split_id" not in oof_df.columns or oof_df["split_id"].isna().all():
            st.info("이 run에 split_id 정보가 없다(오래된 npz일 수 있음).")
        else:
            split_by_case = dict(zip(oof_df["case_id"], oof_df["split_id"]))
            try:
                overlap = split_overlap_in_projection_space(
                    artifact, split_by_case, k=min(5, len(oof_df) - 1)
                )
                overlap_df = pd.DataFrame(
                    [
                        {
                            "fold(val)": name,
                            "n": int(stats["n"]),
                            "다른 fold 이웃 비율": stats["mean_other_split_neighbor_fraction"],
                        }
                        for name, stats in overlap.items()
                    ]
                ).sort_values("fold(val)")
                st.dataframe(overlap_df, use_container_width=True, hide_index=True)
                st.caption(
                    "이웃 비율이 낮을수록(0에 가까울수록) 그 fold의 케이스들이 이 표현 공간에서 "
                    "다른 fold와 잘 안 섞여 있다는 뜻 — 각 fold가 대체로 구별되는 표현을 갖는지의 참고 지표."
                )
            except ValueError as exc:
                st.warning(f"split overlap 계산 불가: {exc}")

    with sub_sae:
        st.markdown("**SAE 기반 분석 — 정상/비정상 활성 패턴 대조**")
        st.caption(
            "3번 탭에서 6-arm 재학습 산출물로 이미 학습·저장해 둔 SAE(`pipeline_explorer_saes/`)를 "
            "그대로 재사용한다 — 새 SAE를 학습하지 않는다. `event_mean`(finetune 캐시)이 SAE가 학습된 "
            "layer5 위치와 같은 표현이라(둘 다 frozen encoder의 동일 checkpoint 출력) 바로 인코딩 가능하다. "
            "정상 라벨 이벤트와 비정상 라벨 이벤트 각각에서 SAE feature별 평균 활성값을 비교해, "
            "어떤 feature가 두 그룹을 가장 잘 가르는지 본다."
        )
        sae_run_name = st.selectbox("SAE run", ["original", "control", "sop_balanced"], key="sae_run")
        sae = load_persisted_sae_cached(sae_run_name, "layer5")
        if sae is None:
            st.info(f"저장된 SAE가 없다({sae_run_name}/layer5).")
        elif st.button("SAE 활성 패턴 대조", key="sae_contrast_button"):
            import json as _json

            diag_by_case_json = _json.dumps(
                dict(zip(labels_df["case_id"].astype(str), labels_df["diagnosis_normalized"]))
            )
            with st.spinner("전체 라벨 케이스의 이벤트를 SAE로 인코딩하는 중..."):
                means, y_abn, _case_ids = load_all_labeled_event_means(
                    str(_REPO_ROOT / "outputs/online2/v2_finetune_v02/event_embeddings"), diag_by_case_json
                )
                with torch.no_grad():
                    codes = sae.encode(torch.from_numpy(means)).numpy()  # [N, dict_size]

            is_abn = y_abn > 0
            mean_normal = codes[~is_abn].mean(axis=0)
            mean_abnormal = codes[is_abn].mean(axis=0)
            fire_rate_normal = (codes[~is_abn] > 0).mean(axis=0)
            fire_rate_abnormal = (codes[is_abn] > 0).mean(axis=0)
            dead_ratio = float((codes == 0).all(axis=0).mean())

            s1, s2, s3 = st.columns(3)
            s1.metric("인코딩된 이벤트 수", f"{codes.shape[0]:,}", help=f"정상 {int((~is_abn).sum()):,} / 비정상 {int(is_abn.sum()):,}")
            s2.metric("dict_size", codes.shape[1])
            s3.metric("dead_feature_ratio", f"{dead_ratio:.3f}")

            diff_df = pd.DataFrame(
                {
                    "feature": range(codes.shape[1]),
                    "정상 평균활성": mean_normal,
                    "비정상 평균활성": mean_abnormal,
                    "차이(비정상-정상)": mean_abnormal - mean_normal,
                    "정상 발화율": fire_rate_normal,
                    "비정상 발화율": fire_rate_abnormal,
                }
            ).sort_values("차이(비정상-정상)", key=abs, ascending=False)
            st.markdown("_그룹을 가장 잘 가르는 feature 상위 15개_")
            st.dataframe(diff_df.head(15), use_container_width=True, hide_index=True)
            st.bar_chart(diff_df.head(15).set_index("feature")[["정상 평균활성", "비정상 평균활성"]])
            st.caption(
                "⚠ 이 표/차트는 `E1_ASSOCIATED`(정상·비정상과 통계적으로 연관) 수준까지다 — 계획서 §9.7 증거 "
                "등급 체계에서 `E3_INTERVENTION`(ablation/steering으로 재현) 이전까지는 이 feature가 "
                "'정상/비정상을 나타내는 개념'이라고 확정하지 않는다. dead_feature_ratio가 여전히 "
                "높다면(§18 원래 문제) 그 자체가 유효한 실측 결과다 — 감추지 않는다."
            )

    with sub_case:
        st.markdown("**개체(케이스) 요약**")
        case_id = st.selectbox("케이스 선택", oof_df["case_id"].tolist(), key="entity_case_select")
        sel = oof_df[oof_df["case_id"] == case_id].iloc[0]
        st.write(f"진단: **{sel['diagnosis']}** (y_abnormal={sel['y_abnormal']:.0f})")
        nearest = nearest_labeled_examples(
            sel[rep_choice],
            reps,
            oof_df["case_id"].tolist(),
            oof_df["diagnosis"].tolist(),
            k=5,
            exclude_case_id=case_id,
        )
        st.dataframe(pd.DataFrame(nearest), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**예측 및 근거 — 예측 결과 · 정답 · 주요 이벤트 · 주요 이벤트 내 주요 토큰**")
        if not ckpt_path:
            st.info("이 run에 체크포인트가 없어 예측/근거를 계산할 수 없다.")
        else:
            st.caption(
                "선택한 케이스를 실제로 forward pass해서 모델 예측(p_abnormal, fine 확률)을 정답과 나란히 보여주고, "
                "`risk_saliency.py::event_ixg_abnormal_margin`으로 뽑은 saliency 상위/하위 이벤트를 보여준다. "
                "**주요 이벤트 안의 토큰별 색은 이제 근사가 아니라 실제 gradient 기반 IxG다** — 그 이벤트만 grad "
                "활성화 상태로 다시 인코딩해서(`token_saliency.py::token_ixg_for_event_v03`) objective에서 그 "
                "이벤트의 입력 토큰 임베딩까지 역전파한 값이다(캐시된 pooled 벡터를 우회 — 진짜 objective→토큰 "
                "gradient)."
            )
            if st.button("예측+근거 계산", key="evidence_button"):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                with st.spinner("forward pass + saliency 계산 중..."):
                    ev_model, ev_meta = load_fold_model_v03(Path(ckpt_path), device)
                    ev_normal_id = int(ev_meta["normal_class_id"])
                    ev_label_map = load_label_map(_REPO_ROOT / "outputs/online2/v2_finetune/label_map.json")
                    ev_label_names = list(ev_label_map["labels"])

                    # 이 finetune run이 실제로 학습에 쓴 emb_dir(config.json)에서, 그걸
                    # 인코딩한 사전학습 run(vocab/encoder)을 역추적한다 -- 하드코딩된
                    # v2_finetune_v02/v2_build를 고른 run과 무관하게 항상 참조하던 문제 수정.
                    ev_emb_dir = da.finetune_run_emb_dir(Path(run_dir_str))
                    ev_encoder_run = da.encoder_run_for_emb_dir(ev_emb_dir)
                    ev_build_dir = da.build_dir_for_run(ev_encoder_run)
                    ev_events_path = ev_build_dir / "events_tokenized_v2.parquet"

                    ev_ds = DiagnosisEventDatasetV03(
                        [case_id],
                        labels_df=labels_df,
                        label_map=ev_label_map,
                        emb_dir=ev_emb_dir,
                        interpretation_bank=None,
                        semantic_dim=192,
                        max_events=4096,
                        require_complete=True,
                        normal_class_id=ev_normal_id,
                    )
                    ev_sample = ev_ds[0]
                    ev_batch = collate_diagnosis_batch_v03([ev_sample], max_events=4096)
                    ev_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ev_batch.items()}
                    evidence = build_case_evidence(
                        ev_model,
                        ev_sample,
                        ev_batch,
                        normal_class_id=ev_normal_id,
                        label_names=ev_label_names,
                        events_path=ev_events_path,
                        k_salient=3,
                    )

                    # 실제 토큰 단위 IxG -- 주요 이벤트(비정상/정상 방향 각각)만 다시
                    # grad 활성화 상태로 인코딩한다(전체 케이스가 아니라 몇 개 이벤트뿐이라
                    # 인터랙티브하게 가능).
                    case_row_ev = labels_df[labels_df["case_id"].astype(str) == case_id].iloc[0]
                    ixg_events = load_case_events_raw_cached(
                        str(ev_events_path), str(case_row_ev["farm_id"]),
                        str(case_row_ev["period_start"]), str(case_row_ev["period_end"]),
                    )
                    ev_event_id_to_idx = {ev.event_id: i for i, ev in enumerate(ixg_events)}
                    ev_vocab, ev_abspos_ref = load_vocab_and_abspos(ev_encoder_run)
                    ev_encoder, _enc_device = load_encoder_cached(ev_encoder_run)
                    ixg_case_t0 = ixg_events[0].timestamp

                    token_ixg_by_event: dict[str, list] = {}
                    salient_ids = [r["event_id"] for r in evidence["top_abnormal_events"] if r["event_id"]]
                    salient_ids += [r["event_id"] for r in evidence["top_normal_events"][:2] if r["event_id"]]
                    for eid in salient_ids:
                        t_idx = ev_event_id_to_idx.get(eid)
                        if t_idx is None:
                            continue
                        try:
                            ixg_result = token_ixg_for_event_v03(
                                encoder=ev_encoder, model=ev_model, stage_a_mod=stage_a_module,
                                events=ixg_events, target_idx=t_idx, case_t0=ixg_case_t0,
                                abspos_reference=ev_abspos_ref, vocab=ev_vocab, batch=ev_batch,
                                event_index=t_idx, normal_class_id=ev_normal_id, device=device,
                                use_binary=True, max_length=1024,
                            )
                            token_ixg_by_event[eid] = list(
                                zip(ixg_result.tokens, ixg_result.scores.tolist(), ixg_result.token_types)
                            )
                        except Exception as exc:  # noqa: BLE001 — one event's IxG failing shouldn't block the rest
                            token_ixg_by_event[eid] = [("(token IxG 계산 실패: " + str(exc) + ")", 0.0, "other")]

                # session_state, not a local var: the button's True state does not survive a
                # rerun triggered by the sliders/multiselects rendered below, so the (expensive)
                # forward pass must be cached here and the display kept outside `if st.button`.
                st.session_state["evidence_result"] = evidence
                st.session_state["evidence_case_id"] = case_id
                st.session_state["evidence_label_names"] = ev_label_names
                st.session_state["evidence_token_ixg"] = token_ixg_by_event

            if (
                st.session_state.get("evidence_case_id") == case_id
                and st.session_state.get("evidence_result") is not None
            ):
                evidence = st.session_state["evidence_result"]
                ev_label_names = st.session_state["evidence_label_names"]
                token_ixg_by_event = st.session_state.get("evidence_token_ixg", {})
                disp_emb_dir = da.finetune_run_emb_dir(Path(run_dir_str))
                disp_events_path = da.build_dir_for_run(da.encoder_run_for_emb_dir(disp_emb_dir)) / "events_tokenized_v2.parquet"
                pred_id = int(np.argmax([evidence["fine_probs"][n] for n in ev_label_names]))
                pred_label = ev_label_names[pred_id]
                gt_label = sel["diagnosis"]
                c1, c2, c3 = st.columns(3)
                c1.metric("예측", pred_label, delta="일치" if pred_label == gt_label else "불일치")
                c2.metric("정답", gt_label)
                c3.metric("p_abnormal (모델)", f"{evidence['p_abnormal']:.3f}")
                st.caption(evidence["image_evidence_note"])

                st.markdown("_주요 이벤트 (abnormal-risk를 가장 높인 순)_")
                raw_window_hours = st.slider(
                    "원본 raw 그래프 문맥 범위 (이벤트 시각 ± N시간)", 3, 48, 24, key="evidence_raw_window"
                )
                for row in evidence["top_abnormal_events"]:
                    with st.expander(
                        f"t={row['case_age_hours']:.1f}h (케이스 시작 후)  saliency={row['saliency_score']:.4f}  "
                        f"event_id={row['event_id']}"
                    ):
                        if row["event_id"]:
                            token_rows = load_event_token_rows(str(disp_events_path), (row["event_id"],))
                            hit = token_rows.get(row["event_id"])
                            if hit:
                                obs_ts = hit["observation_timestamp"]
                                farm_id_ev = str(hit["farm_id"])
                                zone_id_ev = str(hit["zone_id"])
                                st.markdown(
                                    f"**시각**: {obs_ts}  ·  **farm**: {farm_id_ev}  ·  **zone**: {zone_id_ev}  ·  "
                                    f"**케이스 경과**: {row['case_age_hours']:.1f}h"
                                )
                                ixg_toks = token_ixg_by_event.get(row["event_id"])
                                if ixg_toks:
                                    st.markdown(
                                        "**토큰별 IxG (빨강=비정상 쪽으로 밈, 파랑=정상 쪽으로 밈, 진할수록 |score| 큼; "
                                        "테두리 색 = role)**"
                                    )
                                    max_abs = max((abs(s) for _, s, _ in ixg_toks), default=0.0) or 1.0
                                    html_parts = []
                                    for tok, score, ttype in ixg_toks:
                                        role_color = ROLE_COLORS.get(ttype, "#374151")
                                        intensity = min(1.0, abs(score) / max_abs)
                                        bg = f"rgba(220,38,38,{0.12 + 0.55*intensity})" if score > 0 else (
                                            f"rgba(37,99,235,{0.12 + 0.55*intensity})" if score < 0 else "transparent"
                                        )
                                        html_parts.append(
                                            f'<span title="type={ttype} score={score:+.2e}" '
                                            f'style="background:{bg};border:1px solid {role_color};border-radius:4px;'
                                            f'padding:1px 4px;margin:2px;display:inline-block;font-size:0.85em;">{tok}</span>'
                                        )
                                    st.markdown(" ".join(html_parts), unsafe_allow_html=True)
                                else:
                                    st.caption("이 이벤트의 token IxG가 계산되지 않음(role만 아래에 표시).")
                                entries = da.parse_token_trace(
                                    hit["sentence"], hit["measurement_group_ids"], hit["token_roles"],
                                    hit.get("source_file_ids", ""), hit.get("source_row_ids", ""),
                                )
                                html_parts = [
                                    f'<span title="role={e.role}" style="background:{ROLE_COLORS.get(e.role, "#374151")}22;'
                                    f'border:1px solid {ROLE_COLORS.get(e.role, "#374151")};border-radius:4px;'
                                    f'padding:1px 4px;margin:2px;display:inline-block;font-size:0.85em;">{e.token}</span>'
                                    for e in entries
                                ]
                                st.markdown("**토큰 (색 = role, 참고용)**")
                                st.markdown(" ".join(html_parts), unsafe_allow_html=True)

                                # 정확한 계보(source_row_id)가 있으면 그 이벤트가 실제로
                                # 어떤 raw 값에서 나왔는지 재검색 없이 바로 보여준다 -- 아래
                                # 그래프(윈도우 컨텍스트)와는 별개로, "이 토큰의 진짜 값"만 콕
                                # 집어 확인하고 싶을 때 쓴다.
                                sfids = [e.source_file_id for e in entries]
                                srids = [e.source_row_id for e in entries]
                                exact_rows, drift = da.lookup_raw_rows_for_event(sfids, srids)
                                if exact_rows:
                                    st.markdown("**정확한 raw 값 (계보로 직접 조회, 재검색 아님)**")
                                    if drift:
                                        st.caption("⚠ raw 파일이 build 시점과 달라짐(checksum 불일치) — 아래 값은 build 당시 값.")
                                    for table_name, tdf in exact_rows.items():
                                        st.caption(table_name)
                                        st.dataframe(tdf, use_container_width=True, hide_index=True)

                                st.markdown("**원본 raw 데이터 (그래프, 문맥 창)**")
                                raw_tables = da.lookup_raw_rows(
                                    farm_id_ev, zone_id_ev, str(obs_ts), window_hours=raw_window_hours
                                )
                                if not raw_tables:
                                    st.caption("이 시각/zone에 일치하는 raw 데이터를 못 찾음.")
                                else:
                                    event_ts = pd.Timestamp(obs_ts)
                                    if event_ts.tzinfo is not None:
                                        event_ts = event_ts.tz_localize(None)  # match tz-naive raw CSV timestamps
                                    for table_name, tdf in raw_tables.items():
                                        numeric_cols = [
                                            c
                                            for c in tdf.columns
                                            if pd.api.types.is_numeric_dtype(tdf[c])
                                            and c not in ("farm_id", "zone_id")
                                        ]
                                        if not numeric_cols or "timestamp" not in tdf.columns:
                                            continue
                                        plot_cols = st.multiselect(
                                            f"{table_name} — 그래프로 볼 컬럼",
                                            numeric_cols,
                                            default=numeric_cols[:3],
                                            key=f"evidence_plot_cols_{row['event_id']}_{table_name}",
                                        )
                                        for col in plot_cols:
                                            line = (
                                                alt.Chart(tdf)
                                                .mark_line(point=True)
                                                .encode(
                                                    x=alt.X("timestamp:T", title="시각"),
                                                    y=alt.Y(f"{col}:Q", title=col),
                                                    tooltip=["timestamp:T", f"{col}:Q"],
                                                )
                                            )
                                            marker = (
                                                alt.Chart(pd.DataFrame({"timestamp": [event_ts]}))
                                                .mark_rule(color="#e4572e", strokeDash=[4, 4])
                                                .encode(x="timestamp:T")
                                            )
                                            st.altair_chart(
                                                (line + marker).properties(height=140, title=f"{table_name}.{col}"),
                                                use_container_width=True,
                                            )
                                        st.caption(f"점선 = 이 이벤트의 정확한 시각({event_ts})")
                            else:
                                st.caption("이 event_id를 events_tokenized_v2.parquet에서 못 찾음.")

                if evidence["top_normal_events"]:
                    st.markdown("_정상 방향으로 기여한 이벤트_")
                    for row in evidence["top_normal_events"][:2]:
                        st.caption(
                            f"t={row['case_age_hours']:.1f}h saliency={row['saliency_score']:.4f}: "
                            f"{(row['sentence'] or '')[:150]}"
                        )

                st.info(
                    "이 케이스의 개체 공간 위치는 위쪽 '정상/비정상 개념 공간' 산점도에서, "
                    "사전학습 encoder 자체의 concept space(narrative 시퀀스 기준)는 **3번 탭 'Concept Space 시각화'**, "
                    "vocab token 단위 concept space는 **6번 탭 'Vocab Concept Space'**에서 같이 확인할 수 있다 — "
                    "이 셋은 서로 다른 표현 공간(finetune OOF vs 사전학습 시퀀스 vs vocab embedding table)이라 "
                    "하나로 합치지 않고 탭을 분리해 뒀다."
                )

    with sub_tcav:
        st.markdown("**TCAV — narrative 개념과 abnormal-risk 판단 방향 (버튼으로 즉시 계산)**")
        if not ckpt_path:
            st.info("이 run에 체크포인트가 없어 TCAV를 계산할 수 없다.")
        else:
            st.caption(
                "concept_tcav/README.md의 A05 실측(sign_mean=0.267)과 같은 방식 — 다른 narrative_id로 반복 검증할 때 쓴다. "
                "계산 대상은 선택한 체크포인트가 학습에 쓰지 않은 val 케이스로 제한된다(split_manifest.json 필요)."
            )
            narrative_id = st.text_input("narrative_id", value="A05", key="tcav_narrative")
            if st.button("TCAV 계산", key="tcav_button"):
                split_manifest_path = Path(run_dir_str) / "split_manifest.json"
                if not split_manifest_path.exists():
                    st.error(f"{split_manifest_path} 없음")
                    return
                import json

                splits = json.loads(split_manifest_path.read_text(encoding="utf-8"))
                fold_tag = Path(ckpt_path).stem  # e.g. "r0_fold0_best"
                fold_num = next((int(c) for c in fold_tag if c.isdigit()), None)
                matching = [s for s in splits if int(s["fold"]) == fold_num] if fold_num is not None else []
                if not matching:
                    st.error(f"{fold_tag}에 대응하는 split을 split_manifest.json에서 못 찾음")
                    return
                val_cases = matching[0]["val"]

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                with st.spinner(f"{len(val_cases)}개 val 케이스로 TCAV 계산 중..."):
                    model, meta = load_fold_model_v03(Path(ckpt_path), device)
                    label_map = load_label_map(_REPO_ROOT / "outputs/online2/v2_finetune/label_map.json")
                    normal_id = normal_class_id_from_map(label_map)
                    ds = DiagnosisEventDatasetV03(
                        val_cases,
                        labels_df=labels_df,
                        label_map=label_map,
                        emb_dir=_REPO_ROOT / "outputs/online2/v2_finetune_v02/event_embeddings",
                        interpretation_bank=None,
                        semantic_dim=192,
                        max_events=4096,
                        require_complete=True,
                        normal_class_id=normal_id,
                    )
                    farm_ids = {str(labels_df.set_index("case_id").loc[c, "farm_id"]) for c in val_cases}
                    try:
                        cavs, score = run_narrative_tcav(
                            model,
                            ds,
                            collate_fn=lambda xs: collate_diagnosis_batch_v03(xs, max_events=4096),
                            training_events_path=_REPO_ROOT / "outputs/online2/v2_build/training_events_v2.parquet",
                            narrative_id=narrative_id,
                            farm_ids=farm_ids,
                            normal_class_id=normal_id,
                            device=device,
                            use_binary=True,
                            n_bootstraps=50,
                            seed=0,
                        )
                    except ValueError as exc:
                        st.error(f"TCAV 계산 실패: {exc}")
                        return
                c1, c2, c3 = st.columns(3)
                c1.metric("TCAV sign_mean", f"{score.sign_mean:.3f}", help="0.5=무연관, 낮을수록 반대방향 정렬")
                c2.metric("concept 이벤트 수", cavs.n_concept)
                c3.metric("random 이벤트 수", cavs.n_random)
                st.caption(
                    "이 값은 E2_PREDICTIVE 수준 실측치다 — ablation/steering으로 재현되기 전까지 인과적 주장을 하지 않는다 "
                    "(계획서 §9.7 증거 등급)."
                )


def main() -> None:
    st.set_page_config(page_title="Pipeline Explorer", layout="wide")
    st.title("Pipeline Explorer — 원시데이터 → 토큰 → 시퀀스 → encoder → SAE")

    index_df = load_index()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1. 원시↔토큰↔시퀀스", "2. 인코딩 트레이스", "3. 개념/시퀀스 유사도", "4. 학습 버전 브라우저",
        "5. Entity Summary", "6. Vocab Concept Space", "7. Finetune 케이스 토큰화",
    ])
    with tab1:
        render_sequence_trace_tab(index_df)
    with tab2:
        render_encoding_trace_tab()
    with tab3:
        render_similarity_tab(index_df)
    with tab4:
        render_run_browser_tab()
    with tab5:
        render_entity_summary_tab()
    with tab6:
        render_vocab_concept_space_tab()
    with tab7:
        render_finetune_case_tokenization_tab()


if __name__ == "__main__":
    main()
