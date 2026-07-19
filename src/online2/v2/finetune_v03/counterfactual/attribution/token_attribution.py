"""P0-B attribution via fuse-aware event IxG + role-weighted token distribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03
from src.online2.v2.finetune_v03.risk_saliency import (
    event_ixg_abnormal_margin,
    risk_from_binary_logit,
)

from ..evaluation.case_batch import load_case_batch, tensor_batch_only
from ..io_utils import write_parquet
from .aggregation import infer_token_role
from .fold_consensus import percentile_normalize, robust_scale


def _limit_gpu(fraction: float = 0.4) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)


def compute_event_attribution_for_case(
    *,
    case_id: str,
    ckpt_paths: Sequence[Path],
    embeddings_dir: Path,
    labels_path: Path,
    label_map_path: Optional[Path] = None,
    device: str = "cuda",
    gpu_fraction: float = 0.4,
    use_binary: bool = True,
    normalization: str = "robust",
    global_trace: Optional[Any] = None,
    trace_context: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    _limit_gpu(gpu_fraction)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    labels_path = Path(labels_path)
    lmp = Path(label_map_path) if label_map_path else labels_path.parent / "label_map.json"
    if not lmp.exists():
        lmp = Path("/home/dasom/life2vec/outputs/online2/v2_finetune/label_map.json")
    batch_cpu, side = load_case_batch(
        case_id=case_id,
        embeddings_dir=Path(embeddings_dir),
        labels_path=labels_path,
        label_map_path=lmp,
    )
    normal_id = int(batch_cpu["_normal_class_id"])
    eids = [str(x) for x in batch_cpu.get("_event_ids") or side["event_id"].astype(str).tolist()]
    rows = []
    for fold_i, ckpt in enumerate(ckpt_paths):
        from ..cf1s.core_contract import sha256_file
        from ..cf1s.core_trace import TraceKind

        context = dict(trace_context or {})
        trace_phase = str(context.get("phase") or "ATTRIBUTION")
        checkpoint_sha = sha256_file(Path(ckpt))
        load_invocation = f"attribution-load::{case_id}::{fold_i}::{checkpoint_sha[:12]}"
        if global_trace is not None:
            global_trace.begin_operation(
                kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
                invocation_id=load_invocation,
                phase=trace_phase,
                scope="search",
                allow=True,
                case_id=case_id,
                fold_id=fold_i,
                checkpoint_sha=checkpoint_sha,
            )
            global_trace.start(
                kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
                invocation_id=load_invocation,
                phase=trace_phase,
                scope="search",
                case_id=case_id,
                fold_id=fold_i,
                checkpoint_sha=checkpoint_sha,
            )
        try:
            model, meta = load_fold_model_v03(Path(ckpt), device=dev)
        except Exception as exc:
            if global_trace is not None:
                global_trace.fail(
                    kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
                    invocation_id=load_invocation,
                    phase=trace_phase,
                    scope="search",
                    failure_code=type(exc).__name__,
                    case_id=case_id,
                    fold_id=fold_i,
                    checkpoint_sha=checkpoint_sha,
                )
            raise
        if global_trace is not None:
            global_trace.complete(
                kind=TraceKind.ATTRIBUTION_MODEL_LOAD,
                invocation_id=load_invocation,
                phase=trace_phase,
                scope="search",
                case_id=case_id,
                fold_id=fold_i,
                checkpoint_sha=checkpoint_sha,
            )
        if meta.get("normal_class_id") is not None:
            normal_id = int(meta["normal_class_id"])
        batch = {k: v.to(dev) for k, v in tensor_batch_only(batch_cpu).items()}
        forward_invocation = (
            f"attribution-forward::{case_id}::{fold_i}::{checkpoint_sha[:12]}"
        )
        if global_trace is not None:
            global_trace.begin_operation(
                kind=TraceKind.ATTRIBUTION_FORWARD,
                invocation_id=forward_invocation,
                phase=trace_phase,
                scope="search",
                allow=True,
                case_id=case_id,
                fold_id=fold_i,
                checkpoint_sha=checkpoint_sha,
                parent_invocation_id=load_invocation,
            )
            global_trace.start(
                kind=TraceKind.ATTRIBUTION_FORWARD,
                invocation_id=forward_invocation,
                phase=trace_phase,
                scope="search",
                case_id=case_id,
                fold_id=fold_i,
                checkpoint_sha=checkpoint_sha,
                parent_invocation_id=load_invocation,
            )
        try:
            scores = event_ixg_abnormal_margin(
                model,
                batch,
                normal_class_id=normal_id,
                use_binary=use_binary and hasattr(model, "binary_head"),
            )
        except Exception as exc:
            if global_trace is not None:
                global_trace.fail(
                    kind=TraceKind.ATTRIBUTION_FORWARD,
                    invocation_id=forward_invocation,
                    phase=trace_phase,
                    scope="search",
                    failure_code=type(exc).__name__,
                    case_id=case_id,
                    fold_id=fold_i,
                    checkpoint_sha=checkpoint_sha,
                    parent_invocation_id=load_invocation,
                )
            raise
        if global_trace is not None:
            global_trace.complete(
                kind=TraceKind.ATTRIBUTION_FORWARD,
                invocation_id=forward_invocation,
                phase=trace_phase,
                scope="search",
                case_id=case_id,
                fold_id=fold_i,
                checkpoint_sha=checkpoint_sha,
                parent_invocation_id=load_invocation,
            )
        scores_np = scores.detach().float().cpu().numpy()
        mask = batch["padding_mask"][0].detach().cpu().numpy().astype(bool)
        if normalization == "percentile":
            norm = percentile_normalize(scores_np[mask])
        else:
            norm = robust_scale(scores_np[mask])
        norm_full = np.zeros_like(scores_np)
        norm_full[mask] = norm
        for i, eid in enumerate(eids):
            if i >= len(scores_np) or not mask[i]:
                continue
            meta_row = side.iloc[i] if i < len(side) else None
            rows.append(
                {
                    "case_id": case_id,
                    "fold_id": int(fold_i),
                    "event_id": str(eid),
                    "event_index": int(i),
                    "same_time_group_id": str(meta_row["same_time_group_id"]) if meta_row is not None else "",
                    "view": str(meta_row["view"]) if meta_row is not None else "",
                    "zone": str(meta_row["zone"]) if meta_row is not None else "",
                    "timestamp": str(meta_row["timestamp"]) if meta_row is not None else "",
                    "signed_attribution": float(scores_np[i]),
                    "absolute_attribution": float(abs(scores_np[i])),
                    "normalized_signed_attribution": float(norm_full[i]),
                    "normalized_absolute_attribution": float(abs(norm_full[i])),
                    "normalization_method": normalization,
                    "attribution_method": "event_ixg_fuse_aware",
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def distribute_event_scores_to_tokens(
    event_attr: pd.DataFrame,
    token_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Join event scores onto tokens; weight later in MG aggregation via roles."""
    if token_rows is None or len(token_rows) == 0:
        return pd.DataFrame()
    tok = token_rows.copy()
    if "token_role" not in tok.columns:
        tok["token_role"] = tok["token_string"].map(infer_token_role)
    merged = tok.merge(
        event_attr[
            [
                "case_id",
                "fold_id",
                "event_id",
                "signed_attribution",
                "absolute_attribution",
                "normalized_signed_attribution",
                "normalized_absolute_attribution",
                "normalization_method",
            ]
        ],
        on=["case_id", "fold_id", "event_id"],
        how="inner",
    )
    merged["attribution_method"] = "event_ixg_role_distributed"
    return merged


def synthetic_tokens_from_events(
    event_attr: pd.DataFrame,
    *,
    feature: str = "inside_temp_c",
    measurement_group_id: str = "mg0",
) -> pd.DataFrame:
    """Fallback token rows for smoke when event token parquet join is unavailable."""
    rows = []
    for _, r in event_attr.iterrows():
        for idx, (tok, role) in enumerate(
            [
                (f"FEATURE|{feature}", "feature_identity"),
                ("VALUE_ABS|ABS_B10", "abs_value"),
                ("VALUE_GLOBAL_REL|G_B03", "global_value"),
            ]
        ):
            rows.append(
                {
                    "case_id": r["case_id"],
                    "fold_id": r["fold_id"],
                    "event_id": r["event_id"],
                    "measurement_group_id": measurement_group_id,
                    "same_time_group_id": "",
                    "token_index": idx,
                    "token_string": tok,
                    "token_role": role,
                    "feature": feature,
                }
            )
    return pd.DataFrame(rows)
