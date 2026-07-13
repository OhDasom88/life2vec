#!/usr/bin/env python3
"""Offline Stage A: target-centered contextual event embeddings.

For each case / each target event:
  1. Build a ≤1024-token window centered on the target (left + target + right)
  2. Run frozen V2 encoder (best.ckpt)
  3. Pool only the target event token span (masked mean + elementwise max)
  4. Write parquet cache under outputs/online2/v2_finetune/event_embeddings/

Design: outputs/online2/v2_finetune/EVENT_POOLING_FINETUNE_ARCH.md

Examples:
  # window-only
  python scripts/online2_v2/cache_stage_a_event_embeddings.py \\
    --case-id F814133_2025-03-24_2025-04-06 --dry-run

  # 1-case smoke (real encode)
  python scripts/online2_v2/cache_stage_a_event_embeddings.py \\
    --case-id F814133_2025-03-24_2025-04-06 --batch-targets 8 --device cuda

  # all example cases (skip finished)
  python scripts/online2_v2/cache_stage_a_event_embeddings.py \\
    --skip-existing --batch-targets 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_new.types import Background
from src.data_new.vocabulary import RegistryVocabulary
from src.transformer.models import TransformerEncoder

VIEW_RE = re.compile(r"VIEW\|(\w+)")
VIEW_ORDER = {
    "ENVIRONMENT": 0,
    "ROOTZONE": 1,
    "ACTUATOR": 2,
    "GROWTH": 3,
    "IMAGE": 4,
    "INTERPRETATION": 5,
}

# Matches encode_batch background contract (maps to [UNK] in V2 vocab).
DEFAULT_BACKGROUND = Background(origin="UNK", gender="U", birth_month=1, birth_year=2000)


@dataclass
class CaseEvent:
    event_id: str
    event_order: int
    same_time_group_id: str
    timestamp: pd.Timestamp
    view: str
    zone: str
    event_kind: str
    sentence_tokens: list[str]
    token_count: int


@dataclass
class WindowSpec:
    event_indices: list[int]
    target_local_idx: int
    target_token_start: int
    target_token_end: int
    context_token_count: int
    left_context_count: int
    right_context_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt",
    )
    p.add_argument(
        "--events",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/case_manifest.csv",
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json",
    )
    p.add_argument(
        "--abspos-reference",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/abspos_reference.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/event_embeddings",
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--case-id", type=str, default=None)
    p.add_argument("--limit-cases", type=int, default=None)
    p.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="Encode only first N events per case (smoke)",
    )
    p.add_argument("--include-image", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true", help="Build windows only; skip encoder")
    p.add_argument("--batch-targets", type=int, default=8, help="Targets per encoder forward")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip case if complete embedding parquet already exists",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if parquet exists",
    )
    return p.parse_args()


def load_abspos_reference(path: Path) -> pd.Timestamp:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ts = pd.Timestamp(payload["reference_timestamp"])
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def parse_view(sentence: str, event_kind: str) -> str:
    m = VIEW_RE.search(sentence or "")
    if m:
        return m.group(1)
    kind = str(event_kind or "")
    if kind == "LOCAL_OBSERVATION":
        return "ENVIRONMENT"
    if kind in VIEW_ORDER:
        return kind
    return kind or "UNKNOWN"


def sentence_to_tokens(sentence: str) -> list[str]:
    return [t for t in str(sentence or "").split(" ") if t]


def cache_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["event_mean", "event_max", "event_id"])
    except Exception:
        return False
    if df.empty:
        return False
    if "event_mean" not in df.columns or "event_max" not in df.columns:
        return False
    sample = df["event_mean"].iloc[0]
    if sample is None:
        return False
    if isinstance(sample, float) and np.isnan(sample):
        return False
    return True


def load_frozen_encoder(ckpt_path: Path, vocab: RegistryVocabulary, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(ckpt["hparams"])
    # Drop non-serializable / unused keys that break Module init.
    raw.pop("vocabulary", None)
    hparams = {
        **raw,
        "vocabulary": vocab,
        "vocab_size": vocab.size(),
        "training_task": raw.get("training_task", "mlm"),
        "epsilon": float(raw.get("epsilon", 1e-6)),
    }
    model = TransformerEncoder(hparams)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[warn] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] unexpected keys (first 5): {unexpected[:5]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, hparams, ckpt.get("abspos_reference")


def load_events_frame(events_path: Path, farm_ids: set[str]) -> pd.DataFrame:
    cols = [
        "event_id",
        "same_time_group_id",
        "event_kind",
        "observation_timestamp",
        "farm_id",
        "zone_id",
        "SENTENCE",
    ]
    table = pq.read_table(events_path, columns=cols)
    df = table.to_pandas()
    df = df[df["farm_id"].astype(str).isin(farm_ids)].copy()
    df["_ts"] = pd.to_datetime(df["observation_timestamp"], utc=True).dt.tz_localize(None)
    return df


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


def prefix_token_budget() -> int:
    return 1 + len(Background.get_sentence(DEFAULT_BACKGROUND)) + 1


def construct_target_window(
    events: list[CaseEvent],
    target_idx: int,
    *,
    max_length: int = 1024,
) -> WindowSpec:
    prefix = prefix_token_budget()
    budget = max_length - prefix
    target = events[target_idx]
    if target.token_count > budget:
        return WindowSpec(
            event_indices=[target_idx],
            target_local_idx=0,
            target_token_start=prefix,
            target_token_end=prefix + budget,
            context_token_count=prefix + budget,
            left_context_count=0,
            right_context_count=0,
        )

    def stg_id(i: int) -> str:
        return events[i].same_time_group_id or f"__solo_{i}"

    included: set[int] = {target_idx}
    used = target.token_count

    for i, ev in enumerate(events):
        if i == target_idx:
            continue
        if stg_id(i) == stg_id(target_idx) and used + ev.token_count <= budget:
            included.add(i)
            used += ev.token_count

    def try_add(i: int) -> bool:
        nonlocal used
        if i in included:
            return False
        if used + events[i].token_count > budget:
            return False
        included.add(i)
        used += events[i].token_count
        return True

    left_cursor = target_idx - 1
    right_cursor = target_idx + 1
    while left_cursor >= 0 or right_cursor < len(events):
        progressed = False

        if left_cursor >= 0:
            gid = stg_id(left_cursor)
            block: list[int] = []
            j = left_cursor
            while j >= 0 and stg_id(j) == gid:
                block.append(j)
                j -= 1
            missing = [k for k in block if k not in included]
            if missing:
                block_cost = sum(events[k].token_count for k in missing)
                if used + block_cost <= budget:
                    for k in missing:
                        included.add(k)
                    used += block_cost
                    left_cursor = j
                    progressed = True
                else:
                    for k in sorted(missing, key=lambda x: abs(x - target_idx)):
                        if not try_add(k):
                            break
                    left_cursor = -1
                    progressed = True
            else:
                left_cursor = j
                progressed = True

        if right_cursor < len(events):
            gid = stg_id(right_cursor)
            block = []
            j = right_cursor
            while j < len(events) and stg_id(j) == gid:
                block.append(j)
                j += 1
            missing = [k for k in block if k not in included]
            if missing:
                block_cost = sum(events[k].token_count for k in missing)
                if used + block_cost <= budget:
                    for k in missing:
                        included.add(k)
                    used += block_cost
                    right_cursor = j
                    progressed = True
                else:
                    for k in sorted(missing, key=lambda x: abs(x - target_idx)):
                        if not try_add(k):
                            break
                    right_cursor = len(events)
                    progressed = True
            else:
                right_cursor = j
                progressed = True

        if not progressed:
            break

    ordered = sorted(included)
    target_local = ordered.index(target_idx)
    spans: list[tuple[int, int]] = []
    cursor = prefix
    for i in ordered:
        n = events[i].token_count
        spans.append((cursor, cursor + n))
        cursor += n

    t_start, t_end = spans[target_local]
    return WindowSpec(
        event_indices=ordered,
        target_local_idx=target_local,
        target_token_start=t_start,
        target_token_end=t_end,
        context_token_count=cursor,
        left_context_count=t_start - prefix,
        right_context_count=cursor - t_end,
    )


def window_to_tensors(
    events: list[CaseEvent],
    window: WindowSpec,
    *,
    vocab: RegistryVocabulary,
    abspos_reference: pd.Timestamp,
    case_t0: pd.Timestamp,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (4,L) input_ids, (L,) padding_mask True=valid, and realized target end."""
    selected = [events[i] for i in window.event_indices]
    start_hours = [
        int((ev.timestamp - abspos_reference) / pd.Timedelta(hours=1)) for ev in selected
    ]
    ages = [float((ev.timestamp - case_t0) / pd.Timedelta(hours=1)) for ev in selected]
    segments = [((i % 3) + 1) for i in range(len(selected))]
    abspos = [h + 1 for h in start_hours]

    bg = Background.get_sentence(DEFAULT_BACKGROUND)
    flat_tokens: list[str] = ["[CLS]", *bg, "[SEP]"]
    flat_abspos: list[int] = [abspos[0]] * (1 + len(bg) + 1)
    flat_age: list[float] = [ages[0]] * (1 + len(bg) + 1)
    flat_seg: list[int] = [1] * (1 + len(bg) + 1)

    for si, ev in enumerate(selected):
        flat_tokens.extend(ev.sentence_tokens)
        flat_abspos.extend([abspos[si]] * ev.token_count)
        flat_age.extend([ages[si]] * ev.token_count)
        flat_seg.extend([segments[si]] * ev.token_count)

    # Target span must match construct_target_window accounting.
    prefix = 1 + len(bg) + 1
    assert prefix == prefix_token_budget()
    rebuilt_start = prefix + sum(
        events[i].token_count for i in window.event_indices[: window.target_local_idx]
    )
    rebuilt_end = rebuilt_start + events[window.event_indices[window.target_local_idx]].token_count
    assert rebuilt_start == window.target_token_start, (
        rebuilt_start,
        window.target_token_start,
    )
    assert rebuilt_end == window.target_token_end, (rebuilt_end, window.target_token_end)

    L = min(len(flat_tokens), max_length)
    flat_tokens = flat_tokens[:L]
    flat_abspos = flat_abspos[:L]
    flat_age = flat_age[:L]
    flat_seg = flat_seg[:L]

    unk = int(vocab.token2index["[UNK]"])
    ids = [int(vocab.token2index.get(t, unk)) for t in flat_tokens]
    pad_len = max_length - L
    x = torch.tensor(
        [
            ids + [0] * pad_len,
            flat_abspos + [0] * pad_len,
            [int(a) for a in flat_age] + [0] * pad_len,
            flat_seg + [0] * pad_len,
        ],
        dtype=torch.long,
    )
    padding_mask = torch.tensor([True] * L + [False] * pad_len, dtype=torch.bool)
    assert window.target_token_end <= L, (window.target_token_end, L)
    return x, padding_mask, L


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


def process_case(
    case_row: pd.Series,
    *,
    events_df: pd.DataFrame,
    model: Optional[TransformerEncoder],
    vocab: Optional[RegistryVocabulary],
    abspos_reference: pd.Timestamp,
    max_length: int,
    include_image: bool,
    device: torch.device,
    dry_run: bool,
    batch_targets: int,
    limit_events: Optional[int],
) -> pd.DataFrame:
    farm_id = str(case_row["farm_id"])
    case_id = str(case_row["case_id"])
    period_start = pd.Timestamp(case_row["period_start"])
    period_end = pd.Timestamp(case_row["period_end"])

    events = case_events_from_frame(
        events_df,
        farm_id,
        period_start,
        period_end,
        include_image=include_image,
    )
    if not events:
        print(f"[warn] {case_id}: no events")
        return pd.DataFrame()

    if limit_events is not None:
        events = events[: int(limit_events)]

    case_t0 = events[0].timestamp
    rows: list[dict[str, Any]] = []

    if dry_run or model is None or vocab is None:
        for ti, ev in enumerate(events):
            window = construct_target_window(events, ti, max_length=max_length)
            rows.append(
                {
                    "case_id": case_id,
                    "event_id": ev.event_id,
                    "event_order": ev.event_order,
                    "same_time_group_id": ev.same_time_group_id,
                    "timestamp": ev.timestamp.isoformat(),
                    "view": ev.view,
                    "zone": ev.zone,
                    "target_token_start": window.target_token_start,
                    "target_token_end": window.target_token_end,
                    "context_token_count": window.context_token_count,
                    "left_context_count": window.left_context_count,
                    "right_context_count": window.right_context_count,
                    "n_events_in_window": len(window.event_indices),
                    "case_age_hours": float(
                        (ev.timestamp - case_t0) / pd.Timedelta(hours=1)
                    ),
                    "local_hour": int(ev.timestamp.hour),
                }
            )
        return pd.DataFrame(rows)

    pending_x: list[torch.Tensor] = []
    pending_mask: list[torch.Tensor] = []
    pending_span: list[tuple[int, int]] = []
    pending_meta: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal pending_x, pending_mask, pending_span, pending_meta
        if not pending_x:
            return
        pooled = encode_batch_pool(model, pending_x, pending_mask, pending_span, device)
        for meta, (mean_v, max_v) in zip(pending_meta, pooled):
            if not np.isfinite(mean_v).all() or not np.isfinite(max_v).all():
                raise RuntimeError(f"non-finite embedding for {meta['event_id']}")
            meta["event_mean"] = mean_v.tolist()
            meta["event_max"] = max_v.tolist()
            meta["event_embedding"] = None
            rows.append(meta)
        pending_x, pending_mask, pending_span, pending_meta = [], [], [], []

    for ti, ev in enumerate(events):
        window = construct_target_window(events, ti, max_length=max_length)
        meta = {
            "case_id": case_id,
            "event_id": ev.event_id,
            "event_order": ev.event_order,
            "same_time_group_id": ev.same_time_group_id,
            "timestamp": ev.timestamp.isoformat(),
            "view": ev.view,
            "zone": ev.zone,
            "target_token_start": window.target_token_start,
            "target_token_end": window.target_token_end,
            "context_token_count": window.context_token_count,
            "left_context_count": window.left_context_count,
            "right_context_count": window.right_context_count,
            "n_events_in_window": len(window.event_indices),
            "case_age_hours": float((ev.timestamp - case_t0) / pd.Timedelta(hours=1)),
            "local_hour": int(ev.timestamp.hour),
        }
        x, mask, _L = window_to_tensors(
            events,
            window,
            vocab=vocab,
            abspos_reference=abspos_reference,
            case_t0=case_t0,
            max_length=max_length,
        )
        pending_x.append(x)
        pending_mask.append(mask)
        pending_span.append((window.target_token_start, window.target_token_end))
        pending_meta.append(meta)
        if len(pending_x) >= batch_targets:
            flush()
        if (ti + 1) % 200 == 0:
            print(f"  {case_id}: {ti + 1}/{len(events)} events", flush=True)

    flush()
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if args.case_id:
        manifest = manifest[manifest["case_id"].astype(str) == args.case_id]
    if args.limit_cases is not None:
        manifest = manifest.head(args.limit_cases)
    if manifest.empty:
        raise SystemExit("no cases selected")

    abspos_reference = load_abspos_reference(args.abspos_reference)
    print(
        f"abspos_reference={abspos_reference} cases={len(manifest)} "
        f"dry_run={args.dry_run} batch_targets={args.batch_targets}",
        flush=True,
    )

    farm_ids = set(manifest["farm_id"].astype(str))
    print(f"loading events for {len(farm_ids)} farms…", flush=True)
    events_df = load_events_frame(args.events, farm_ids)
    print(f"events rows (farm-filtered)={len(events_df)}", flush=True)

    model = None
    vocab = None
    if not args.dry_run:
        vocab = RegistryVocabulary(
            registry_path=str(args.vocab),
            registry_version="v2",
        )
        model, hparams, _ = load_frozen_encoder(
            args.ckpt, vocab, torch.device(args.device)
        )
        print(
            f"encoder H={hparams.get('hidden_size')} L={hparams.get('max_length')} "
            f"encoders={hparams.get('n_encoders')} device={args.device}",
            flush=True,
        )

    for _, case_row in manifest.iterrows():
        case_id = str(case_row["case_id"])
        out_path = args.out_dir / f"{case_id}.parquet"
        if (
            not args.force
            and not args.dry_run
            and args.skip_existing
            and cache_is_complete(out_path)
            and args.limit_events is None
        ):
            print(f"=== {case_id} SKIP (exists) ===", flush=True)
            continue

        print(f"=== {case_id} ===", flush=True)
        df = process_case(
            case_row,
            events_df=events_df,
            model=model,
            vocab=vocab,
            abspos_reference=abspos_reference,
            max_length=args.max_length,
            include_image=args.include_image,
            device=torch.device(args.device),
            dry_run=args.dry_run,
            batch_targets=max(1, int(args.batch_targets)),
            limit_events=args.limit_events,
        )
        if args.dry_run:
            meta_path = args.out_dir / f"{case_id}.window_meta.parquet"
            df.to_parquet(meta_path, index=False)
            print(f"wrote {meta_path} rows={len(df)}", flush=True)
        else:
            # Write via temp then rename for crash-safe resume.
            tmp = out_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(out_path)
            means = np.asarray(df["event_mean"].iloc[0], dtype=np.float32)
            print(
                f"wrote {out_path} rows={len(df)} H={means.shape[0]} "
                f"mean_norm={float(np.linalg.norm(means)):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
