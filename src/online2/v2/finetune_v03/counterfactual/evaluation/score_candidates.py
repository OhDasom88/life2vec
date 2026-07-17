"""Score Path A / Path B candidates with frozen diagnosis critic (fuse-aware)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import torch

from .case_batch import load_case_batch, tensor_batch_only
from .diagnosis_critic import ensemble_delta_r
from .perturbation import interpolate_event_indices, noop_batch, zero_event_indices
from ..candidates.path_a_generator import choose_direction_hint
from ..candidates.path_b_b0_generator import select_best_operation
from ..grounding.actuator_span import apply_operation_to_event_ids
from ..io_utils import write_jsonl
from ..manifest import load_manifest


def _label_map_path(cfg: Mapping[str, Any]) -> Path:
    p = Path(cfg.get("label_map_path") or Path(cfg["labels_path"]).parent / "label_map.json")
    if not p.exists():
        p = Path("/home/dasom/life2vec/outputs/online2/v2_finetune/label_map.json")
    return p


def score_path_a_results(
    cfg: Mapping[str, Any],
    paths,
    rows: List[Dict[str, Any]],
    *,
    max_candidates: int = 12,
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    man = load_manifest(paths.manifests / "diagnosis_model_manifest.json")
    ckpts = [Path(p) for p in man["checkpoint_paths"]]
    batch, side = load_case_batch(
        case_id=cfg["case_id"],
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=_label_map_path(cfg),
    )
    base = tensor_batch_only(batch)
    normal_id = int(batch["_normal_class_id"])
    eid_to_idx = {str(r.event_id): int(r.event_index) for r in side.itertuples()}

    # group by locus feature for direction probe
    scored: List[Dict[str, Any]] = []
    # limit GPU work
    work = rows[: int(max_candidates)]
    # direction probe aggregates per feature
    probe_cache: Dict[str, Dict[str, float]] = {}

    for row in work:
        locus = row.get("locus") or {}
        eids = locus.get("event_ids") or []
        idxs = []
        for eid in eids:
            if str(eid) in eid_to_idx:
                idxs.append(eid_to_idx[str(eid)])
        # fallback: top ENV event by order 0 if synthetic ids
        if not idxs:
            env = side[side["view"].astype(str).str.upper().str.contains("ENV", na=False)]
            if len(env):
                idxs = [int(env.iloc[0]["event_index"])]

        side_name = str(row.get("probe_side") or "lower")
        alpha = float(cfg.get("path_a_embed_alpha", 0.35))
        toward = "zero" if side_name == "lower" else "amplify"
        after = interpolate_event_indices(base, idxs, alpha=alpha, toward=toward)
        stats = ensemble_delta_r(
            ckpts,
            base,
            after,
            normal_class_id=normal_id,
            device=str(cfg.get("device", "cuda")),
            gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
            use_binary=True,
        )
        feat = str(row.get("feature"))
        probe_cache.setdefault(feat, {})
        if side_name == "lower":
            probe_cache[feat]["delta_r_lower"] = float(stats["delta_r"])
        else:
            probe_cache[feat]["delta_r_upper"] = float(stats["delta_r"])

        out = dict(row)
        out.update(
            {
                "risk_before": stats["risk_before"],
                "risk_after": stats["risk_after"],
                "delta_r": stats["delta_r"],
                "delta_r_folds": stats["delta_r_folds"],
                "folds_improved": stats["folds_improved"],
                "stage_a_reencode_mode": "cached_event_embedding_perturbation",
                "critic_event_indices": idxs,
                "route_before": "KNOWN",
                "route_after": "KNOWN" if stats["delta_r"] <= 0.05 else "KNOWN",
            }
        )
        # reject if risk increases a lot
        if stats["delta_r"] > 0.05:
            out["operational_eligibility"] = "BLOCKED"
            out["reason_codes"] = list(out.get("reason_codes") or []) + ["ERR_DELTA_R_WORSE"]
        scored.append(out)

    # attach direction hints
    for out in scored:
        feat = str(out.get("feature"))
        d = probe_cache.get(feat) or {}
        if "delta_r_lower" in d and "delta_r_upper" in d:
            out["direction_hint"] = choose_direction_hint(d["delta_r_lower"], d["delta_r_upper"])
        elif "delta_r_lower" in d or "delta_r_upper" in d:
            # partial probe
            out["direction_hint"] = {
                "value": "decrease" if float(out.get("delta_r", 0)) < 0 else "increase",
                "source": "adjacent_raw_perturbation",
                "status": "proposal_only",
                "delta_r_lower": d.get("delta_r_lower"),
                "delta_r_upper": d.get("delta_r_upper"),
            }

    # append unscored remainder without critic
    scored_ids = {id(r) for r in work}
    for row in rows[len(work) :]:
        out = dict(row)
        out["delta_r"] = None
        out["stage_a_reencode_mode"] = "not_scored_budget"
        scored.append(out)

    write_jsonl(paths.results / "path_a_cf_results.jsonl", scored)
    write_jsonl(paths.artifacts / "path_a_content_candidates.jsonl", scored)
    return scored


def score_path_b_results(
    cfg: Mapping[str, Any],
    paths,
    candidates: List[Dict[str, Any]],
    spans: pd.DataFrame,
    *,
    max_spans: int = 5,
    max_ops_per_span: int = 6,
) -> List[Dict[str, Any]]:
    if not candidates:
        write_jsonl(paths.results / "path_b_b0_results.jsonl", [])
        return []
    man = load_manifest(paths.manifests / "diagnosis_model_manifest.json")
    ckpts = [Path(p) for p in man["checkpoint_paths"]]
    batch, side = load_case_batch(
        case_id=cfg["case_id"],
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=_label_map_path(cfg),
    )
    base = tensor_batch_only(batch)
    normal_id = int(batch["_normal_class_id"])
    eid_to_idx = {str(r.event_id): int(r.event_index) for r in side.itertuples()}

    # group candidates by span_id
    by_span: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_span.setdefault(str(c.get("span_id")), []).append(c)

    span_ids = list(by_span.keys())[: int(max_spans)]
    all_scored_cands: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for sid in span_ids:
        ops = by_span[sid][: int(max_ops_per_span)]
        # ensure NO_OP first
        ops = sorted(ops, key=lambda o: 0 if o.get("operation") == "NO_OP" else 1)
        scored_ops = []
        sp_rows = spans[spans["span_id"].astype(str) == str(sid)] if len(spans) else pd.DataFrame()
        aligned_idx = []
        if len(sp_rows) and "aligned_event_indices" in sp_rows.columns:
            raw = sp_rows.iloc[0]["aligned_event_indices"]
            if raw is None:
                aligned_idx = []
            elif hasattr(raw, "tolist"):
                aligned_idx = [int(x) for x in raw.tolist()]
            else:
                aligned_idx = [int(x) for x in list(raw)]
        for op in ops:
            keep, drop = apply_operation_to_event_ids(
                (sp_rows.iloc[0]["source_event_ids"] if len(sp_rows) else []),
                op["operation"],
                int(op.get("cut_events") or 0),
            )
            # map drop event ids → indices; prefer aligned indices truncated
            drop_idx = []
            for eid in drop:
                if str(eid) in eid_to_idx:
                    drop_idx.append(eid_to_idx[str(eid)])
            if not drop_idx and aligned_idx and op.get("operation") != "NO_OP":
                cut = int(op.get("cut_events") or 0)
                if op.get("operation") == "truncate_end":
                    drop_idx = aligned_idx[-cut:] if cut else []
                elif op.get("operation") == "truncate_start":
                    drop_idx = aligned_idx[:cut] if cut else []
                elif op.get("operation") == "clear_span":
                    drop_idx = list(aligned_idx)

            if op.get("operation") == "NO_OP":
                after = noop_batch(base)
            else:
                after = zero_event_indices(base, drop_idx)

            stats = ensemble_delta_r(
                ckpts,
                base,
                after,
                normal_class_id=normal_id,
                device=str(cfg.get("device", "cuda")),
                gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
                use_binary=True,
            )
            out = dict(op)
            out.update(
                {
                    "model_space_delta_r": stats["model_space_delta_r"],
                    "risk_before": stats["risk_before"],
                    "risk_after": stats["risk_after"],
                    "delta_r_folds": stats["delta_r_folds"],
                    "folds_improved": stats["folds_improved"],
                    "critic_event_indices": drop_idx,
                    "stage_a_reencode_mode": "cached_event_embedding_perturbation",
                    "limitation": (op.get("limitation") or [])
                    + [
                        "classifier-oriented hypothesis evaluation",
                        "not physical response validation",
                    ],
                }
            )
            if stats["model_space_delta_r"] > 0.05:
                out["operational_eligibility"] = "BLOCKED"
            scored_ops.append(out)
            all_scored_cands.append(out)
        best = select_best_operation(scored_ops)
        best_rows.append(dict(best) if isinstance(best, dict) else {"status": "NO_VALID_OPERATION"})

    write_jsonl(paths.artifacts / "path_b_operation_candidates.jsonl", all_scored_cands)
    write_jsonl(paths.results / "path_b_b0_results.jsonl", best_rows)
    return best_rows


def noop_sanity_check(
    cfg: Mapping[str, Any],
    paths,
) -> Dict[str, Any]:
    man = load_manifest(paths.manifests / "diagnosis_model_manifest.json")
    ckpts = [Path(p) for p in man["checkpoint_paths"]]
    batch, _side = load_case_batch(
        case_id=cfg["case_id"],
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=_label_map_path(cfg),
    )
    base = tensor_batch_only(batch)
    stats = ensemble_delta_r(
        ckpts,
        base,
        noop_batch(base),
        normal_class_id=int(batch["_normal_class_id"]),
        device=str(cfg.get("device", "cuda")),
        gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
    )
    ok = abs(float(stats["delta_r"])) < 1e-5
    return {"ok": ok, "delta_r": stats["delta_r"], "risk_before": stats["risk_before"]}
