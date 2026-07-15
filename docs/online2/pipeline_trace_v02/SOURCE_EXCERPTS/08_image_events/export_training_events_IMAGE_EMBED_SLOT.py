# Excerpt from export_training_events_v2_event_grain.py


# --- IMAGE sentence → IMAGE_EMBED_SLOT roles / embedding_status (lines 35-90) ---
    if isinstance(value, list):
        return list(value)
    return list(json.loads(str(value)))


def _modality_ref(event_kind: str, sentence: str) -> str:
    kind = str(event_kind or "")
    if kind == "IMAGE" or "[IMAGE_SLOT]" in sentence:
        return json.dumps(["IMAGE_EMBED_SLOT"])
    if kind == "INTERPRETATION" or "[TEXT_SLOT]" in sentence:
        return json.dumps(["TEXT_EMBED_SLOT"])
    return "[]"


def load_event_lookup(events_path: Path) -> dict[str, dict[str, Any]]:
    """Load ~222k events into a dict (small vs sequences)."""
    cols = [
        "event_id",
        "same_time_group_id",
        "event_kind",
        "observation_timestamp",
        "farm_id",
        "zone_id",
        "SENTENCE",
        "measurement_group_ids",
        "token_roles",
        "embedding_status",
    ]
    table = pq.read_table(events_path, columns=cols)
    df = table.to_pandas()
    lookup: dict[str, dict[str, Any]] = {}
    for rec in df.itertuples(index=False):
        lookup[str(rec.event_id)] = {
            "same_time_group_id": str(rec.same_time_group_id or ""),
            "event_kind": str(rec.event_kind or ""),
            "observation_timestamp": rec.observation_timestamp,
            "farm_id": str(rec.farm_id or ""),
            "zone_id": str(rec.zone_id or ""),
            "SENTENCE": str(rec.SENTENCE or ""),
            "measurement_group_ids": rec.measurement_group_ids
            if rec.measurement_group_ids is not None
            else "[]",
            "token_roles": rec.token_roles if rec.token_roles is not None else "[]",
            "embedding_status": str(rec.embedding_status or ""),
        }
    del df, table
    return lookup


def expand_sequence_rows(
    rec: Any,
    event_lookup: dict[str, dict[str, Any]],
    build_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Expand one sequence record into event-grain training rows."""
    stats = Counter()
