# Excerpt from scripts/online2_v2/cache_stage_a_event_embeddings.py
# NOTE: DINO fuse is NOT here; see 01_dino_fuse/stage_a_image_encode.py


# --- case_events_from_frame: IMAGE excluded unless --include-image (lines 231-274) ---
def case_events_from_frame(
    df_all: pd.DataFrame,
    farm_id: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    *,
    include_image: bool,
) -> list[CaseEvent]:
    df = df_all[df_all["farm_id"].astype(str) == str(farm_id)].copy()
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end) + pd.Timedelta(days=1)
    df = df[(df["_ts"] >= start) & (df["_ts"] < end)].copy()

    exclude = {"INTERPRETATION"}
    if not include_image:
        exclude.add("IMAGE")
    df = df[~df["event_kind"].astype(str).isin(exclude)].copy()
    if (df["event_kind"].astype(str) == "INTERPRETATION").any():
        raise AssertionError("INTERPRETATION events leaked into Stage A input")

    df["_view"] = [parse_view(s, k) for s, k in zip(df["SENTENCE"], df["event_kind"])]
    df["_view_rank"] = df["_view"].map(lambda v: VIEW_ORDER.get(v, 99))
    df["_zone_rank"] = pd.to_numeric(df["zone_id"], errors="coerce").fillna(999)
    df = df.sort_values(
        ["_ts", "same_time_group_id", "_zone_rank", "_view_rank", "event_id"]
    ).reset_index(drop=True)

    out: list[CaseEvent] = []
    for i, row in df.iterrows():
        toks = sentence_to_tokens(row["SENTENCE"])
        out.append(
            CaseEvent(
                event_id=str(row["event_id"]),
                event_order=int(i),
                same_time_group_id=str(row["same_time_group_id"] or ""),
                timestamp=pd.Timestamp(row["_ts"]),
                view=str(row["_view"]),
                zone=str(row["zone_id"]),
                event_kind=str(row["event_kind"]),
                sentence_tokens=toks,
                token_count=len(toks),
            )
        )
    return out

# --- encode_batch_pool: no DINO — span mean/max only (lines 471-494) ---
@torch.no_grad()
def encode_batch_pool(
    model: TransformerEncoder,
    xs: list[torch.Tensor],
    masks: list[torch.Tensor],
    spans: list[tuple[int, int]],
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batch encode windows and pool each target span."""
    x = torch.stack(xs, dim=0).to(device)  # (B, 4, L)
    mask = torch.stack(masks, dim=0).to(device).long()  # (B, L) True=valid
    hidden = model.transformer.forward_finetuning(x=x, padding_mask=mask)  # (B, L, H)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for b, (s, e) in enumerate(spans):
        span = hidden[b, s:e, :]
        if span.numel() == 0:
            h = hidden.shape[-1]
            z = np.zeros(h, dtype=np.float32)
            out.append((z, z))
            continue
        mean_v = span.mean(dim=0).float().cpu().numpy().astype(np.float32)
        max_v = span.max(dim=0).values.float().cpu().numpy().astype(np.float32)
        out.append((mean_v, max_v))
    return out
