#!/usr/bin/env python3
"""Stage A cache with IMAGE events + DINO attached (and optional slot injection).

Does NOT modify scripts/online2_v2/cache_stage_a_event_embeddings.py.

Modes:
  1) attach-only (default after base encode): call v0.1 cache with --include-image
     into v02 out-dir, then attach raw dino_vec by event_id.
  2) --inject-slots: re-encode with IMAGE_SLOT ← SharedImageAdapter(DINO) residual
     before encoder layers (folds should prefer attach + fold-trainable adapter).

Examples:
  # 1-case with IMAGE + dino columns (uses frozen encoder, no slot bake)
  python scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py \\
    --case-id F814133_2025-03-24_2025-04-06 --device cuda

  # slot injection bake (optional experiment)
  python scripts/online2_v2/v02/cache_stage_a_image_slots_v02.py \\
    --case-id F814133_2025-03-24_2025-04-06 --inject-slots --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.image_index import (  # noqa: E402
    default_dino_paths,
    load_dino_index,
)
from src.online2.v2.finetune_v02.stage_a_image_encode import (  # noqa: E402
    encode_batch_pool_with_dino_slots,
    make_stage_a_image_adapter,
)
from src.online2.v2.finetune_v02.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION  # noqa: E402

V01_CACHE = ROOT / "scripts/online2_v2/cache_stage_a_event_embeddings.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / "event_embeddings",
    )
    p.add_argument("--case-id", type=str, default=None)
    p.add_argument("--limit-cases", type=int, default=None)
    p.add_argument("--limit-events", type=int, default=None)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Case manifest CSV (default: v0.1 example case_manifest; pass problem manifest for problem_set)",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-targets", type=int, default=8)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--inject-slots",
        action="store_true",
        help="Bake IMAGE_SLOT←DINO residual into Stage A encode (fold adapter prefers off)",
    )
    p.add_argument("--attach-only", action="store_true", help="Skip encode; only attach dino_vec")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_v01():
    spec = importlib.util.spec_from_file_location("cache_stage_a_v01", V01_CACHE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attach_dino_columns(parquet_path: Path, dino_index: dict) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    vecs = []
    n_hit = 0
    for eid, view in zip(df["event_id"].astype(str), df.get("view", pd.Series([""] * len(df)))):
        v = dino_index.get(str(eid))
        if v is None:
            vecs.append(None)
        else:
            vecs.append(v.tolist())
            n_hit += 1
    df["dino_vec"] = vecs
    df["has_dino"] = [v is not None for v in vecs]
    df.to_parquet(parquet_path, index=False)
    return {
        "path": str(parquet_path),
        "n_events": int(len(df)),
        "n_dino": int(n_hit),
        "n_image_view": int((df["view"].astype(str) == "IMAGE").sum())
        if "view" in df.columns
        else None,
    }


def run_v01_include_image(args: argparse.Namespace, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(V01_CACHE),
        "--include-image",
        "--out-dir",
        str(out_dir),
        "--device",
        str(args.device),
        "--batch-targets",
        str(args.batch_targets),
    ]
    if args.manifest is not None:
        cmd += ["--manifest", str(args.manifest)]
    if args.case_id:
        cmd += ["--case-id", args.case_id]
    if args.limit_cases is not None:
        cmd += ["--limit-cases", str(args.limit_cases)]
    if args.limit_events is not None:
        cmd += ["--limit-events", str(args.limit_events)]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    print({"spawn_v01_include_image": cmd}, flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def inject_reencode_case(
    *,
    case_id: str,
    mod: Any,
    dino_index: dict,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    """Re-encode one case with slot injection (fold-train adapters prefer attach-only)."""
    from src.data_new.vocabulary import RegistryVocabulary

    device = torch.device(args.device)
    manifest = pd.read_csv(ROOT / "outputs/online2/v2_finetune/case_manifest.csv")
    row = manifest[manifest["case_id"].astype(str) == case_id]
    if row.empty:
        raise KeyError(case_id)
    case_row = row.iloc[0]
    vocab = RegistryVocabulary(
        str(ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json")
    )
    model, hparams, _ = mod.load_frozen_encoder(
        ROOT / "outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt",
        vocab,
        device,
    )
    hidden = int(hparams.get("hidden_size", 384))
    adapter = make_stage_a_image_adapter(hidden).to(device).eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    abspos_ref = mod.load_abspos_reference(
        ROOT / "outputs/online2/v2_build/abspos_reference.json"
    )
    events_df = mod.load_events_frame(
        ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet",
        {str(case_row["farm_id"])},
    )
    events = mod.case_events_from_frame(
        events_df,
        str(case_row["farm_id"]),
        pd.Timestamp(case_row["period_start"]),
        pd.Timestamp(case_row["period_end"]),
        include_image=True,
    )
    if args.limit_events:
        events = events[: int(args.limit_events)]
    case_t0 = events[0].timestamp
    max_length = 1024

    rows = []
    pending_x, pending_mask, pending_span = [], [], []
    pending_tokens, pending_dino, pending_meta = [], [], []

    def flush():
        nonlocal pending_x, pending_mask, pending_span, pending_tokens, pending_dino, pending_meta
        if not pending_x:
            return
        pooled = encode_batch_pool_with_dino_slots(
            model,
            pending_x,
            pending_mask,
            pending_span,
            flat_token_lists=pending_tokens,
            slot_dino=pending_dino,
            adapter=adapter,
            device=device,
        )
        for meta, (mean_v, max_v) in zip(pending_meta, pooled):
            meta["event_mean"] = mean_v.tolist()
            meta["event_max"] = max_v.tolist()
            meta["event_embedding"] = None
            rows.append(meta)
        pending_x, pending_mask, pending_span = [], [], []
        pending_tokens, pending_dino, pending_meta = [], [], []

    for ti, ev in enumerate(events):
        window = mod.construct_target_window(events, ti, max_length=max_length)
        x, mask, _L = mod.window_to_tensors(
            events,
            window,
            vocab=vocab,
            abspos_reference=abspos_ref,
            case_t0=case_t0,
            max_length=max_length,
        )
        # rebuild flat tokens for slot indexing (same order as window_to_tensors)
        from src.data_new.types import Background

        bg = Background.get_sentence(mod.DEFAULT_BACKGROUND)
        flat = ["[CLS]", *bg, "[SEP]"]
        for i in window.event_indices:
            flat.extend(events[i].sentence_tokens)
        flat = flat[:max_length]
        dino = dino_index.get(ev.event_id) if ev.view == "IMAGE" else None
        # also inject context IMAGE slots: pass target dino only when target is IMAGE;
        # for context images, encode_batch injects when dino provided — supply target's
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
            "dino_vec": dino.tolist() if dino is not None else None,
            "has_dino": dino is not None,
            "slot_injection": True,
        }
        pending_x.append(x)
        pending_mask.append(mask)
        pending_span.append((window.target_token_start, window.target_token_end))
        pending_tokens.append(flat)
        pending_dino.append(dino)
        pending_meta.append(meta)
        if len(pending_x) >= int(args.batch_targets):
            flush()
    flush()

    out_path = out_dir / f"{case_id}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return {"case_id": case_id, "n_events": len(rows), "path": str(out_path), "inject_slots": True}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path, npy_path = default_dino_paths(args.root)
    dino_index = load_dino_index(meta_path=meta_path, npy_path=npy_path)

    report: dict[str, Any] = {
        "finetune_version": FINETUNE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "n_dino_index": len(dino_index),
        "inject_slots": bool(args.inject_slots),
        "cases": [],
    }

    if args.attach_only:
        paths = (
            [out_dir / f"{args.case_id}.parquet"]
            if args.case_id
            else sorted(out_dir.glob("*.parquet"))
        )
        for path in paths:
            if path.name.startswith("_"):
                continue
            report["cases"].append(attach_dino_columns(path, dino_index))
    elif args.inject_slots:
        mod = _load_v01()
        man_path = args.manifest or (ROOT / "outputs/online2/v2_finetune/case_manifest.csv")
        manifest = pd.read_csv(man_path)
        if args.case_id:
            ids = [args.case_id]
        else:
            ids = manifest["case_id"].astype(str).tolist()
            if args.limit_cases:
                ids = ids[: int(args.limit_cases)]
        for cid in ids:
            report["cases"].append(
                inject_reencode_case(
                    case_id=cid,
                    mod=mod,
                    dino_index=dino_index,
                    args=args,
                    out_dir=out_dir,
                )
            )
    else:
        # Recommended: v0.1 encode with --include-image, then attach raw DINO
        run_v01_include_image(args, out_dir)
        paths = (
            [out_dir / f"{args.case_id}.parquet"]
            if args.case_id
            else sorted(out_dir.glob("*.parquet"))
        )
        for path in paths:
            if not path.exists() or path.name.startswith("_"):
                continue
            report["cases"].append(attach_dino_columns(path, dino_index))

    (out_dir / "_IMAGE_STAGE_A_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
