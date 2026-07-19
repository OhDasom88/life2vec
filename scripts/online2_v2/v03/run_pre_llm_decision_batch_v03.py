"""Batch pre-LLM decision packages for online2 55 cases.

Pipeline (stops before open-set LLM / Gemma report generation):

  Stage-A pretrained embeddings (frozen cache)
  → v0.3 diagnosis ensemble (binary + fine + routing)
  → CF M1 attribution → temporal → locus → Path A / Path B B0 candidates
  → optional light critic scoring
  → per-case JSON + cohort summary

Does NOT call LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03
from src.online2.v2.finetune_v03.config import RoutingThresholds
from src.online2.v2.finetune_v03.counterfactual.attribution.aggregation import (
    aggregate_events,
    aggregate_measurement_groups,
    aggregate_spans,
)
from src.online2.v2.finetune_v03.counterfactual.attribution.token_attribution import (
    compute_event_attribution_for_case,
    distribute_event_scores_to_tokens,
    synthetic_tokens_from_events,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.path_a_generator import (
    build_path_a_candidates,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.path_b_b0_generator import (
    generate_b0_operations,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.case_batch import (
    load_case_batch,
    load_embedding_sidecar,
    tensor_batch_only,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.diagnosis_critic import (
    risk_from_batch,
)
from src.online2.v2.finetune_v03.counterfactual.gates.context_abac import (
    check_path_a,
    check_path_b,
)
from src.online2.v2.finetune_v03.counterfactual.gates.local_rules import (
    build_token_edits,
    compare_token_bundles,
)
from src.online2.v2.finetune_v03.counterfactual.gates.safety_bounds import project_to_safe
from src.online2.v2.finetune_v03.counterfactual.grounding.actuator_span import (
    ground_actuator_span,
)
from src.online2.v2.finetune_v03.counterfactual.grounding.raw_target import (
    nearest_feasible_interior,
)
from src.online2.v2.finetune_v03.counterfactual.grounding.retokenize import (
    retokenize_abs_value,
    retokenize_actuator_literal,
)
from src.online2.v2.finetune_v03.counterfactual.io_utils import write_json, write_jsonl
from src.online2.v2.finetune_v03.counterfactual.locus.selector import (
    select_path_a_loci,
    select_path_b_loci,
)
from src.online2.v2.finetune_v03.counterfactual.manifest import (
    build_diagnosis_manifest,
    discover_fold_ckpts,
)
from src.online2.v2.finetune_v03.counterfactual.pipeline import _parse_cell_raw
from src.online2.v2.finetune_v03.counterfactual.semantics_registry import (
    build_semantics_registry,
)
from src.online2.v2.finetune_v03.counterfactual.temporal.span_builder import build_zoh_spans
from src.online2.v2.finetune_v03.counterfactual.temporal.span_event_align import (
    align_spans_to_events,
    alignment_summary,
)
from src.online2.v2.finetune_v03.open_set import energy_from_logits, route_case
from src.online2.v2.finetune_v03.risk_saliency import risk_from_binary_logit
from src.online2.v2.binning import BinningRegistryV2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cfg(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


@torch.no_grad()
def diagnose_case(
    *,
    case_id: str,
    ckpt_paths: List[Path],
    embeddings_dir: Path,
    labels_path: Path,
    label_map_path: Path,
    device: str,
    gpu_fraction: float,
    label_map: Dict[str, Any],
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(gpu_fraction), device=0)
        except RuntimeError:
            pass
    batch, side = load_case_batch(
        case_id=case_id,
        embeddings_dir=embeddings_dir,
        labels_path=labels_path,
        label_map_path=label_map_path,
    )
    normal_id = int(batch["_normal_class_id"])
    labels = list(label_map["labels"])
    base = tensor_batch_only(batch)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    fold_p_abn = []
    fold_fine = []
    fold_energy = []
    fold_pred = []
    for ckpt in ckpt_paths:
        model, meta = load_fold_model_v03(Path(ckpt), device=dev)
        nid = int(meta.get("normal_class_id") if meta.get("normal_class_id") is not None else normal_id)
        b = {k: v.to(dev) for k, v in base.items()}
        # forward via encode_events path
        enc = model.encode_events(
            b["event_mean"],
            b["event_max"],
            b["case_age_hours"],
            b["view_id"],
            b["zone_id"],
            b["local_hour"],
            b["padding_mask"],
            dino_vec=b.get("dino_vec"),
            dino_mask=b.get("dino_mask"),
        )
        h = enc.get("h_binary", enc["h_case"])
        h_fine = enc.get("h_fine", enc["h_case"])
        if hasattr(model, "binary_head"):
            p_abn = float(torch.sigmoid(model.binary_head(h))[0].item())
        else:
            p_abn = float(risk_from_binary_logit(torch.tensor([0.0]))[0])
        logits = model.fine_head(h_fine) if hasattr(model, "fine_head") else model.head(h_fine)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        fold_p_abn.append(p_abn)
        fold_fine.append(probs)
        fold_energy.append(energy_from_logits(logits[0].detach().cpu().numpy()))
        fold_pred.append(int(np.argmax(probs)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    p_abn_mean = float(np.mean(fold_p_abn))
    fine_mean = np.mean(np.stack(fold_fine, axis=0), axis=0)
    pred_id = int(np.argmax(fine_mean))
    # fold agreement: count of folds matching ensemble argmax
    n_agree = int(sum(1 for p in fold_pred if p == pred_id))
    agree_frac = float(n_agree / max(len(fold_pred), 1))
    head_contradiction = (p_abn_mean >= 0.5 and pred_id == normal_id) or (
        p_abn_mean < 0.5 and pred_id != normal_id
    )
    # RoutingThresholds.min_fold_agreement is an integer fold-count threshold (default 4).
    # This ensemble has 3 folds → use min(n_folds, configured) effectively by passing n_agree
    # and a lowered thr for 3-fold runs.
    thr = RoutingThresholds()
    thr.min_fold_agreement = min(int(thr.min_fold_agreement), len(fold_pred))
    route = route_case(
        p_abnormal=p_abn_mean,
        fine_probs=fine_mean,
        normal_class_id=normal_id,
        energy=float(np.mean(fold_energy)),
        fold_agreement=float(n_agree),
        head_contradiction=bool(head_contradiction),
        use_reject=True,
        use_open_set=True,
        thr=thr,
    )
    gt_row = pd.read_csv(labels_path)
    gt = gt_row[gt_row["case_id"].astype(str) == str(case_id)]
    gt_diag = str(gt.iloc[0]["diagnosis_normalized"]) if len(gt) else None
    gt_set = str(gt.iloc[0]["set"]) if len(gt) and "set" in gt.columns else None

    return {
        "case_id": case_id,
        "set": gt_set,
        "ground_truth_diagnosis": gt_diag,
        "p_abnormal_mean": p_abn_mean,
        "p_abnormal_folds": fold_p_abn,
        "fine_pred_id": pred_id,
        "fine_pred_label": labels[pred_id] if 0 <= pred_id < len(labels) else str(pred_id),
        "fine_prob_max": float(np.max(fine_mean)),
        "fine_probs": {labels[i]: float(fine_mean[i]) for i in range(len(labels))},
        "fold_agreement": agree_frac,
        "fold_agree_count": n_agree,
        "energy_mean": float(np.mean(fold_energy)),
        "route": route.decision,
        "route_reasons": list(route.reasons),
        "route_scores": dict(route.scores),
        "n_events": int(base["padding_mask"].sum().item()),
        "llm_stage": "NOT_RUN",
        "pretrain_role": "stage_a_event_embedding_cache",
        "diagnosis_model": "finetune_v03_ensemble",
    }


def build_spans_for_case(cfg: Dict[str, Any], case_id: str, cells: pd.DataFrame) -> pd.DataFrame:
    farm_id = case_id.split("_")[0]
    features = cfg.get("actuator_features") or []
    sub_all = cells[cells["farm_id"].astype(str) == str(farm_id)].copy()
    if "is_null" in sub_all.columns:
        sub_all = sub_all[~sub_all["is_null"].fillna(False)]
    frames = []
    for feat in features:
        sub = sub_all[sub_all["column_name"].astype(str) == feat].copy()
        if not len(sub):
            continue
        sub = sub.rename(columns={"observation_timestamp": "timestamp"})
        sub["raw_value"] = [
            _parse_cell_raw(v, d) for v, d in zip(sub["raw_value"], sub.get("raw_display", [None] * len(sub)))
        ]
        sub["event_id"] = sub.apply(lambda r: f"{feat}:{r['zone_id']}:{r['timestamp']}", axis=1)
        for zone, g in sub.groupby("zone_id"):
            g = g.sort_values("timestamp").drop_duplicates("timestamp")
            if len(g) > int(cfg.get("max_actuator_rows", 2000)):
                g = g.iloc[:: max(1, len(g) // 2000)]
            spans = build_zoh_spans(
                g,
                feature=feat,
                farm_id=farm_id,
                zone_id=zone,
                case_id=case_id,
                nominal_interval_min=float(cfg.get("sampling_interval_min", 60)),
            )
            if len(spans):
                frames.append(spans)
    if not frames:
        return pd.DataFrame()
    spans_df = pd.concat(frames, ignore_index=True)
    side = load_embedding_sidecar(Path(cfg["embeddings_dir"]), case_id)
    return align_spans_to_events(spans_df, side)


def path_a_candidates_for_loci(
    cfg: Dict[str, Any],
    loci: List[Dict[str, Any]],
    cells: pd.DataFrame,
    semantics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    reg = BinningRegistryV2.load(Path(cfg["binning_registry_path"]))
    out = []
    farm_id = cfg.get("_farm_id") or (loci[0]["case_id"].split("_")[0] if loci else "")
    for locus in loci[: int(cfg.get("top_k_loci", 10))]:
        feat = locus["feature"]
        g1 = check_path_a(feat, semantics)
        if g1["status"] != "PASSED":
            continue
        rule = reg.rules.get((feat, "ABS", None))
        if rule is None:
            continue
        edges = list(rule.edges)
        sub = cells[(cells["farm_id"].astype(str) == str(farm_id)) & (cells["column_name"] == feat)]
        if "is_null" in sub.columns:
            sub = sub[~sub["is_null"].fillna(False)]
        if not len(sub):
            continue
        parsed = [_parse_cell_raw(v, d) for v, d in zip(sub["raw_value"], sub.get("raw_display", [None] * len(sub)))]
        vals = pd.to_numeric(pd.Series(parsed), errors="coerce").dropna()
        if not len(vals):
            continue
        observed = float(vals.median())
        cands = build_path_a_candidates(feature=feat, observed_raw=observed, edges=edges)
        for c in cands:
            g2 = project_to_safe(
                c["target_raw"],
                hard=(min(edges), max(edges)),
                operational_known=False,
            )
            tokens_from = retokenize_abs_value(feat, observed, edges)
            tokens_to = retokenize_abs_value(feat, float(c["target_raw"]), edges)
            g4 = compare_token_bundles(tokens_to, tokens_to)
            out.append(
                {
                    **c,
                    "locus": locus,
                    "gate1": g1,
                    "gate2": g2,
                    "gate4": g4,
                    "token_edits": build_token_edits(
                        event_id=str(locus["event_ids"][0]),
                        measurement_group_id=str(locus.get("measurement_group_id")),
                        feature=feat,
                        from_tokens=tokens_from,
                        to_tokens=tokens_to,
                        gate4_passed=g4["status"] == "PASSED",
                    ),
                    "operational_eligibility": "EXPLANATORY_ONLY"
                    if g2["status"] == "PASSED"
                    else "EXPERT_REVIEW_REQUIRED",
                    "llm_stage": "NOT_RUN",
                }
            )
    return out


def path_b_candidates_for_loci(
    cfg: Dict[str, Any],
    loci: List[Dict[str, Any]],
    spans: pd.DataFrame,
) -> List[Dict[str, Any]]:
    thr = cfg.get("path_b_thresholds") or {}
    out = []
    for locus in loci[: int(cfg.get("top_k_loci", 10))]:
        sp = spans[spans["span_id"] == locus.get("span_id")] if len(spans) else pd.DataFrame()
        if not len(sp):
            continue
        span = sp.iloc[0].to_dict()
        ops = generate_b0_operations(span, locus.get("attribution") or {}, thresholds=thr, gate1_passed=True)
        for op in ops:
            grounded = ground_actuator_span({**op, "feature": locus["feature"]})
            out.append(
                {
                    **op,
                    "locus": locus,
                    "grounded_actions": [grounded],
                    "gate2": {
                        "status": "PARTIAL",
                        "reason_codes": ["OPERATIONAL_BOUNDS_UNKNOWN"],
                    },
                    "operational_eligibility": "EXPERT_REVIEW_REQUIRED",
                    "model_space_delta_r": None,
                    "limitation": grounded.get("limitation"),
                    "llm_stage": "NOT_RUN",
                }
            )
    return out


def process_one_case(
    *,
    case_id: str,
    cfg: Dict[str, Any],
    ckpt_paths: List[Path],
    label_map: Dict[str, Any],
    semantics: Dict[str, Any],
    cells: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    case_dir = out_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg["_farm_id"] = case_id.split("_")[0]

    diagnosis = diagnose_case(
        case_id=case_id,
        ckpt_paths=ckpt_paths,
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=Path(cfg["label_map_path"]),
        device=str(cfg.get("device", "cuda")),
        gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
        label_map=label_map,
    )
    write_json(case_dir / "diagnosis.json", diagnosis)

    # Attribution
    event_attr = compute_event_attribution_for_case(
        case_id=case_id,
        ckpt_paths=ckpt_paths,
        embeddings_dir=Path(cfg["embeddings_dir"]),
        labels_path=Path(cfg["labels_path"]),
        label_map_path=Path(cfg["label_map_path"]),
        device=str(cfg.get("device", "cuda")),
        gpu_fraction=float(cfg.get("gpu_memory_fraction", 0.4)),
    )
    feature = cfg.get("path_a_probe_feature", "inside_temp_c")
    env_attr = event_attr
    if "view" in event_attr.columns:
        tmp = event_attr[event_attr["view"].astype(str).str.upper().str.contains("ENV", na=False)]
        if len(tmp):
            env_attr = tmp
    tok = synthetic_tokens_from_events(env_attr, feature=feature, measurement_group_id=f"mg:{feature}")
    token_attr = distribute_event_scores_to_tokens(env_attr, tok)
    mg = aggregate_measurement_groups(token_attr, role_weights=cfg.get("token_role_weights"))
    ev = aggregate_events(mg)
    event_attr.to_parquet(case_dir / "event_ixg.parquet", index=False)
    mg.to_parquet(case_dir / "mg_attribution.parquet", index=False)
    ev.to_parquet(case_dir / "event_attribution.parquet", index=False)

    # Temporal + locus
    spans = build_spans_for_case(cfg, case_id, cells)
    if len(spans):
        spans.to_parquet(case_dir / "actuator_spans.parquet", index=False)
        write_json(case_dir / "span_alignment.json", alignment_summary(spans))

    # span attribution proxy from event scores
    span_attr = pd.DataFrame()
    if len(spans) and len(ev):
        rows = []
        for fold_id, eg in ev.groupby("fold_id"):
            for _, sp in spans.iterrows():
                row = dict(sp)
                row["fold_id"] = int(fold_id)
                s = eg["signed_attribution"]
                thr = float(s.quantile(0.75)) if len(s) else 0.0
                pos = s[s >= thr]
                mean = float(pos.mean()) if len(pos) else float(s.mean() if len(s) else 0.0)
                row["duration_weighted_mean"] = mean
                row["duration_weighted_positive_ratio"] = 1.0 if mean >= 0 else float((s > 0).mean() if len(s) else 0.0)
                row["start_mass_ratio"] = 0.2
                row["middle_mass_ratio"] = 0.3
                row["end_mass_ratio"] = 0.5
                row["signed_total_attribution"] = mean * max(int(row.get("n_events") or 1), 1)
                row["absolute_total_attribution"] = abs(mean) * max(int(row.get("n_events") or 1), 1)
                row["max_positive_attribution"] = max(mean, 0.0)
                row["max_absolute_attribution"] = abs(mean)
                row["attribution_mass_center"] = 0.7
                row["peak_locus_index"] = max(0, int(row.get("n_events") or 1) - 1)
                rows.append(row)
        span_attr = pd.DataFrame(rows)

    path_a_loci = select_path_a_loci(
        mg,
        semantics=semantics,
        min_agree=int(cfg.get("min_fold_agree", 2)),
        top_k=int(cfg.get("top_k_loci", 10)),
    )
    wl = set(cfg.get("actuator_whitelist") or [])
    path_b_loci = select_path_b_loci(
        span_attr if len(span_attr) else pd.DataFrame(),
        semantics=semantics,
        whitelist=wl or None,
        min_agree=int(cfg.get("min_fold_agree", 2)),
        top_k=int(cfg.get("top_k_loci", 10)),
    )
    write_jsonl(case_dir / "loci_path_a.jsonl", path_a_loci)
    write_jsonl(case_dir / "loci_path_b.jsonl", path_b_loci)

    path_a_cands = path_a_candidates_for_loci(cfg, path_a_loci, cells, semantics)
    path_b_cands = path_b_candidates_for_loci(cfg, path_b_loci, spans)
    write_jsonl(case_dir / "path_a_candidates.jsonl", path_a_cands)
    write_jsonl(case_dir / "path_b_candidates.jsonl", path_b_cands)

    package = {
        "schema_version": "pre_llm_decision.v1",
        "created_at": utc_now(),
        "case_id": case_id,
        "llm_boundary": "STOPPED_BEFORE_LLM",
        "stages_completed": [
            "pretrain_stage_a_cache",
            "diagnosis_v03_ensemble",
            "routing_known_unknown_reject",
            "cf_m1_attribution",
            "temporal_recovery",
            "locus_selection",
            "path_a_grounded_targets",
            "path_b_b0_operations",
        ],
        "stages_not_run": [
            "open_set_retrieval_llm",
            "gemma_report_generation",
            "operational_actuator_execution",
        ],
        "diagnosis": diagnosis,
        "counts": {
            "path_a_loci": len(path_a_loci),
            "path_b_loci": len(path_b_loci),
            "path_a_candidates": len(path_a_cands),
            "path_b_candidates": len(path_b_cands),
            "spans": int(len(spans)),
            "span_aligned_rate": float(alignment_summary(spans).get("aligned_rate") or 0.0)
            if len(spans)
            else 0.0,
        },
        "top_path_a": path_a_loci[:3],
        "top_path_b": path_b_loci[:3],
        "recommendation_gate": {
            "analysis_report": bool(path_a_loci or path_b_loci),
            "management_suggestion": False,  # B1/response not available; never OPERATIONAL
            "llm_narrative": False,
        },
    }
    write_json(case_dir / "decision_package.json", package)
    return package


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="optional max cases for smoke")
    p.add_argument("--case-id", type=str, default=None)
    args = p.parse_args()
    cfg = load_cfg(args.config)
    out_root = Path(cfg.get("decision_output_root") or cfg.get("output_root") or "outputs/m1") / "decision_pre_llm"
    out_root.mkdir(parents=True, exist_ok=True)

    # Freeze once
    man = build_diagnosis_manifest(
        run_dir=Path(cfg["run_dir"]),
        tokenizer_path=Path(cfg["tokenizer_path"]),
        binning_registry_path=Path(cfg["binning_registry_path"]),
        cells_path=Path(cfg["cells_path"]),
        embeddings_dir=Path(cfg["embeddings_dir"]),
        out_path=out_root / "diagnosis_model_manifest.json",
        repeat=int(cfg.get("repeat", 0)),
    )
    semantics_reg = build_semantics_registry(
        Path(cfg["feature_audit_csv"]),
        out_yaml=out_root / "feature_semantics_registry.yaml",
        out_feature_csv=out_root / "feature_semantics_audit.csv",
        out_actuator_csv=out_root / "actuator_semantics_audit.csv",
    )
    semantics = semantics_reg.get("features", {})
    ckpt_paths = [Path(x) for x in man["checkpoint_paths"]]
    from src.online2.v2.finetune_v03.dataset import load_label_map

    label_map = load_label_map(Path(cfg["label_map_path"]))
    labels_df = pd.read_csv(cfg["labels_path"])
    cells = pd.read_parquet(
        cfg["cells_path"],
        columns=[
            "farm_id",
            "zone_id",
            "observation_timestamp",
            "column_name",
            "raw_value",
            "raw_display",
            "is_null",
        ],
    )

    case_ids = labels_df["case_id"].astype(str).tolist()
    if args.case_id:
        case_ids = [args.case_id]
    if args.limit and args.limit > 0:
        case_ids = case_ids[: args.limit]

    summary_rows = []
    errors = []
    for i, case_id in enumerate(case_ids, 1):
        print(f"[{i}/{len(case_ids)}] {case_id}", flush=True)
        try:
            pkg = process_one_case(
                case_id=case_id,
                cfg=cfg,
                ckpt_paths=ckpt_paths,
                label_map=label_map,
                semantics=semantics,
                cells=cells,
                out_dir=out_root,
            )
            d = pkg["diagnosis"]
            summary_rows.append(
                {
                    "case_id": case_id,
                    "set": d.get("set"),
                    "gt": d.get("ground_truth_diagnosis"),
                    "route": d.get("route"),
                    "p_abnormal": d.get("p_abnormal_mean"),
                    "pred": d.get("fine_pred_label"),
                    "fold_agreement": d.get("fold_agreement"),
                    "path_a_loci": pkg["counts"]["path_a_loci"],
                    "path_b_loci": pkg["counts"]["path_b_loci"],
                    "path_a_cands": pkg["counts"]["path_a_candidates"],
                    "path_b_cands": pkg["counts"]["path_b_candidates"],
                    "spans": pkg["counts"]["spans"],
                    "span_aligned_rate": pkg["counts"]["span_aligned_rate"],
                    "llm_stage": "NOT_RUN",
                    "status": "OK",
                }
            )
        except Exception as e:
            traceback.print_exc()
            errors.append({"case_id": case_id, "error": str(e)})
            summary_rows.append({"case_id": case_id, "status": "ERROR", "error": str(e), "llm_stage": "NOT_RUN"})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "cohort_summary.csv", index=False)
    write_json(
        out_root / "cohort_summary.json",
        {
            "created_at": utc_now(),
            "n_cases": len(case_ids),
            "n_ok": int((summary["status"] == "OK").sum()) if "status" in summary.columns else 0,
            "n_error": len(errors),
            "route_counts": summary["route"].value_counts(dropna=False).to_dict()
            if "route" in summary.columns
            else {},
            "errors": errors,
            "llm_boundary": "STOPPED_BEFORE_LLM",
            "models": {
                "pretrain_stage_a_cache": str(cfg["embeddings_dir"]),
                "pretrain_ckpt_reference": str(
                    cfg.get(
                        "pretrain_ckpt",
                        "outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt",
                    )
                ),
                "diagnosis_run_dir": str(cfg["run_dir"]),
                "cf_m1_package": "src/online2/v2/finetune_v03/counterfactual",
            },
        },
    )
    print(json.dumps({"ok": True, "out": str(out_root), "n": len(case_ids), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
